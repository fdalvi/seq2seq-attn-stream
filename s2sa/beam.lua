require 'nn'
require 'string'
require 'nngraph'

require 's2sa.models'
require 's2sa.data'
require 's2sa.scorer'

path = require 'pl.path'
stringx = require 'pl.stringx'

local sent_id = 0
local opt = {}
local cmd = torch.CmdLine()
local prev_dec_state = nil

-- file location
cmd:option('-model', 'seq2seq_lstm_attn.t7.', [[Path to model .t7 file]])
cmd:option('-src_file', '', [[Source sequence to decode (one line per sequence)]])
cmd:option('-targ_file', '', [[True target sequence (optional)]])
cmd:option('-output_file', 'pred.txt', [[Path to output the predictions (each line will be the decoded sequence]])
cmd:option('-src_dict', 'data/demo.src.dict', [[Path to source vocabulary (*.src.dict file)]])
cmd:option('-targ_dict', 'data/demo.targ.dict', [[Path to target vocabulary (*.targ.dict file)]])
cmd:option('-feature_dict_prefix', 'data/demo', [[Prefix of the path to features vocabularies (*.feature_N.dict files)]])
cmd:option('-char_dict', 'data/demo.char.dict', [[If using chars, path to character vocabulary (*.char.dict file)]])
cmd:option('-stream', 0, [[Enable Stream Decoding]])

-- beam search options
cmd:option('-beam', 5, [[Beam size]])
cmd:option('-keep_dec_state', 0, [[Keep Decoder state from previous life]])
cmd:option('-policy', 'wue', [[Policy]])
cmd:option('-max_sent_l', 250, [[Maximum sentence length. If any sequences in srcfile are longer than this then it will error out]])
cmd:option('-simple', 0, [[If = 1, output prediction is simply the first time the top of the beam
                         ends with an end-of-sentence token. If = 0, the model considers all
                         hypotheses that have been generated so far that ends with end-of-sentence
                         token and takes the highest scoring of all of them.]])
cmd:option('-replace_unk', 0, [[Replace the generated UNK tokens with the source token that
                              had the highest attention weight. If srctarg_dict is provided,
                              it will lookup the identified source token and give the corresponding
                              target token. If it is not provided (or the identified source token
                              does not exist in the table) then it will copy the source token]])
cmd:option('-srctarg_dict', 'data/en-de.dict', [[Path to source-target dictionary to replace UNK
                                               tokens. See README.md for the format this file should be in]])
cmd:option('-score_gold', 0, [[If = 1, score the log likelihood of the gold as well]])
cmd:option('-n_best', 1, [[If > 1, it will also output an n_best list of decoded sentences]])
cmd:option('-gpuid', -1, [[ID of the GPU to use (-1 = use CPU)]])
cmd:option('-gpuid2', -1, [[Second GPU ID]])
cmd:option('-cudnn', 0, [[If using character model, this should be = 1 if the character model was trained using cudnn]])

cmd:option('-rescore', '', [[use specified metric to select best translation in the beam, available: bleu, gleu]])
cmd:option('-rescore_param', 4, [[parameter for the scoring metric, for BLEU is corresponding to n_gram ]])

-- nwords policy arguments
cmd:option('-nwords_start', 8, [[nwords policy start delay]])
cmd:option('-nwords_read', 4, [[nwords policy read delay]])
cmd:option('-nwords_write', 4, [[nwords policy write delay]])

-- agent policy arguments
cmd:option('-agent_model', '', [[trained agent]])
cmd:option('-agent_vocab', '', [[path to agent_vocab]])

-- extra information
cmd:option('-delay_file', '', [[If != '', every output word has an associated token as to how many source words were waiting on saved in delay_file]])

function copy(orig)
  local orig_type = type(orig)
  local copy
  if orig_type == 'table' then
    copy = {}
    for orig_key, orig_value in pairs(orig) do
      copy[orig_key] = orig_value
    end
  else
    copy = orig
  end
  return copy
end

local function append_table(dst, src)
  for i = 1, #src do
    table.insert(dst, src[i])
  end
end

local StateAll = torch.class("StateAll")

function StateAll.initial(start)
  return {start}
end

function StateAll.advance(state, token)
  local new_state = copy(state)
  table.insert(new_state, token)
  return new_state
end

function StateAll.disallow(out)
  local bad = {1, 3} -- 1 is PAD, 3 is BOS
  for j = 1, #bad do
    out[bad[j]] = -1e9
  end
end

function StateAll.same(state1, state2)
  for i = 2, #state1 do
    if state1[i] ~= state2[i] then
      return false
    end
  end
  return true
end

function StateAll.next(state)
  return state[#state]
end

function StateAll.heuristic(state)
  return 0
end

function StateAll.print(state)
  for i = 1, #state do
    io.write(state[i] .. " ")
  end
  print()
end

-- Convert a flat index to a row-column tuple.
function flat_to_rc(v, flat_index)
  local row = math.floor((flat_index - 1) / v:size(2)) + 1
  return row, (flat_index - 1) % v:size(2) + 1
end

function typeof(var)
    local _type = type(var);
    if(_type ~= "table" and _type ~= "userdata") then
        return _type;
    end
    local _meta = getmetatable(var);
    if(_meta ~= nil and _meta._NAME ~= nil) then
        return _meta._NAME;
    else
        return _type;
    end
end

-- orig is any table of CudaTensors
function deepcopy(orig)
  new_copy = {}
  for i = 1, #orig do
    -- table.insert(rnn_state_dec, init_fwd_dec[i]:zero())
    table.insert(new_copy, orig[i]:clone())
  end
  return new_copy
end

function generate_beam(model, initial, K, max_sent_l, source, source_features, gold)
  --reset decoder initial states
  if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
    cutorch.setDevice(opt.gpuid)
  end
  local n = max_sent_l
  -- Backpointer table.
  local prev_ks = torch.LongTensor(n, K):fill(1)
  -- Current States.
  local next_ys = torch.LongTensor(n, K):fill(1)
  -- Current Scores.
  local scores = torch.FloatTensor(n, K)
  scores:zero()
  local source_l = math.min(source:size(1), opt.max_sent_l)
  local attn_argmax = {} -- store attn weights
  attn_argmax[1] = {}

  local states = {} -- store predicted word idx
  states[1] = {}
  for k = 1, 1 do
    table.insert(states[1], initial)
    table.insert(attn_argmax[1], initial)
    next_ys[1][k] = State.next(initial)
  end

  local source_input
  if model_opt.use_chars_enc == 1 then
    source_input = source:view(source_l, 1, source:size(2)):contiguous()
  else
    source_input = source:view(source_l, 1)
  end

  local rnn_state_enc = {}
  for i = 1, #init_fwd_enc do
    table.insert(rnn_state_enc, init_fwd_enc[i]:zero())
  end
  local context = context_proto[{{}, {1,source_l}}]:clone() -- 1 x source_l x rnn_size
  -- print("Contexts:",context:sum())
  for t = 1, source_l do
    local encoder_input = {source_input[t]}
    if model_opt.num_source_features > 0 then
      append_table(encoder_input, source_features[t])
    end
    append_table(encoder_input, rnn_state_enc)
    local out = model[1]:forward(encoder_input)
    rnn_state_enc = out
    context[{{},t}]:copy(out[#out])
    -- print("Contexts:",context:sum())
  end

  rnn_state_dec = {}

  if prev_dec_state == nil or opt.keep_dec_state == 0 then
    prev_dec_state = deepcopy(init_fwd_dec)
  end
  -- print(typeof(prev_dec_state[1]))
  -- print(prev_dec_state[1][1][1])
  -- print(init_fwd_dec)
  for i = 1, #init_fwd_dec do
    -- table.insert(rnn_state_dec, init_fwd_dec[i]:zero())
    table.insert(rnn_state_dec, prev_dec_state[i])
  end

  if model_opt.init_dec == 1 then
    for L = 1, model_opt.num_layers do
      rnn_state_dec[L*2-1+model_opt.input_feed]:copy(
        rnn_state_enc[L*2-1]:expand(K, model_opt.rnn_size))
      rnn_state_dec[L*2+model_opt.input_feed]:copy(
        rnn_state_enc[L*2]:expand(K, model_opt.rnn_size))
    end
  end

  if model_opt.brnn == 1 then
    for i = 1, #rnn_state_enc do
      rnn_state_enc[i]:zero()
    end
    for t = source_l, 1, -1 do
      local encoder_input = {source_input[t]}
      if model_opt.num_source_features > 0 then
        append_table(encoder_input, source_features[t])
      end
      append_table(encoder_input, rnn_state_enc)
      local out = model[4]:forward(encoder_input)
      rnn_state_enc = out
      context[{{},t}]:add(out[#out])
    end
    if model_opt.init_dec == 1 then
      for L = 1, model_opt.num_layers do
        rnn_state_dec[L*2-1+model_opt.input_feed]:add(
          rnn_state_enc[L*2-1]:expand(K, model_opt.rnn_size))
        rnn_state_dec[L*2+model_opt.input_feed]:add(
          rnn_state_enc[L*2]:expand(K, model_opt.rnn_size))
      end
    end
  end
  if opt.score_gold == 1 then
    rnn_state_dec_gold = {}
    for i = 1, #rnn_state_dec do
      table.insert(rnn_state_dec_gold, rnn_state_dec[i][{{1}}]:clone())
    end
  end

  context = context:expand(K, source_l, model_opt.rnn_size)

  if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
    cutorch.setDevice(opt.gpuid2)
    local context2 = context_proto2[{{1, K}, {1, source_l}}]
    context2:copy(context)
    context = context2
  end

  out_float = torch.FloatTensor()

  local i = 1
  local done = false
  local max_score = -1e9
  local found_eos = false

  while (not done) and (i < n) do
    i = i+1
    states[i] = {}
    attn_argmax[i] = {}
    local decoder_input1
    if model_opt.use_chars_dec == 1 then
      decoder_input1 = word2charidx_targ:index(1, next_ys:narrow(1,i-1,1):squeeze())
    else
      decoder_input1 = next_ys:narrow(1,i-1,1):squeeze()
      if opt.beam == 1 then
        decoder_input1 = torch.LongTensor({decoder_input1})
      end
    end
    -- local decsum = 0
    -- for idx = 1,#rnn_state_dec do
      -- decsum = decsum + rnn_state_dec[idx]:sum()
    -- end
    -- print("Decoder input:", decoder_input1:sum(), context:sum(), decsum)
    local decoder_input
    if model_opt.attn == 1 then
      decoder_input = {decoder_input1, context, table.unpack(rnn_state_dec)}
    else
      decoder_input = {decoder_input1, context[{{}, source_l}], table.unpack(rnn_state_dec)}
    end

    local out_decoder = model[2]:forward(decoder_input)
    local out_pred_idx = #out_decoder
    if model_opt.guided_alignment == 1 then
        out_pred_idx = #out_decoder - 1
    end
    local out = model[3]:forward(out_decoder[out_pred_idx]) -- K x vocab_size

    rnn_state_dec = {} -- to be modified later
    if model_opt.input_feed == 1 then
      table.insert(rnn_state_dec, out_decoder[out_pred_idx])
    end
    for j = 1, out_pred_idx - 1 do
      table.insert(rnn_state_dec, out_decoder[j])
    end
    out_float:resize(out:size()):copy(out)
    for k = 1, K do
      State.disallow(out_float:select(1, k))
      out_float[k]:add(scores[i-1][k])
    end
    -- All the scores available.

    local flat_out = out_float:view(-1)
    if i == 2 then
      flat_out = out_float[1] -- all outputs same for first batch
    end

    if model_opt.start_symbol == 1 then
      decoder_softmax.output[{{},1}]:zero()
      decoder_softmax.output[{{},source_l}]:zero()
    end

    for k = 1, K do
      while true do
        local score, index = flat_out:max(1)
        local score = score[1]
        local prev_k, y_i = flat_to_rc(out_float, index[1])
        states[i][k] = State.advance(states[i-1][prev_k], y_i)
        local diff = true
        for k2 = 1, k-1 do
          if State.same(states[i][k2], states[i][k]) then
            diff = false
          end
        end

        if i < 2 or diff then
          if model_opt.attn == 1 then
            max_attn, max_index = decoder_softmax.output[prev_k]:max(1)
            attn_argmax[i][k] = State.advance(attn_argmax[i-1][prev_k],max_index[1])
          end
          prev_ks[i][k] = prev_k
          next_ys[i][k] = y_i
          scores[i][k] = score
          flat_out[index[1]] = -1e9
          break -- move on to next k
        end
        flat_out[index[1]] = -1e9
      end
    end

    -- for k = 1,K do
    --   io.write(scores[i][k],"\t")
    -- end
    -- io.write("\n")


    for j = 1, #rnn_state_dec do
      rnn_state_dec[j]:copy(rnn_state_dec[j]:index(1, prev_ks[i]))
    end
    end_hyp = states[i][1]
    end_score = scores[i][1]
    if model_opt.attn == 1 then
      end_attn_argmax = attn_argmax[i][1]
    end
    if end_hyp[#end_hyp] == END then
      done = true
      found_eos = true
    else
      for k = 1, K do
        local possible_hyp = states[i][k]
        if possible_hyp[#possible_hyp] == END then
          found_eos = true
          if scores[i][k] > max_score then
            max_hyp = possible_hyp
            max_score = scores[i][k]
            if model_opt.attn == 1 then
              max_attn_argmax = attn_argmax[i][k]
            end
          end
        end
      end
    end
  end

  prev_dec_state = deepcopy(rnn_state_dec)

  local best_mscore = -1e9
  local mscore_hyp
  local mscore_scores
  local mscore_attn_argmax
  local gold_table
  if opt.rescore ~= '' then
    gold_table = gold[{{2, gold:size(1)-1}}]:totable()
    for k = 1, K do
        local hyp={}
        for _,v in pairs(states[i][k]) do
          if v == END then break; end
          table.insert(hyp,v)
        end
        table.insert(hyp, END)
        local score_k = scorers[opt.rescore](hyp, gold_table, opt.rescore_param)
        if score_k > best_mscore then
          mscore_hyp = hyp
          mscore_scores = scores[i][k]
          best_mscore = score_k
          mscore_attn_argmax = attn_argmax[i][k]
        end
    end
  end

  local gold_score = 0
  if opt.score_gold == 1 then
    rnn_state_dec = {}
    for i = 1, #init_fwd_dec do
      table.insert(rnn_state_dec, init_fwd_dec[i][{{1}}]:zero())
    end
    if model_opt.init_dec == 1 then
      rnn_state_dec = rnn_state_dec_gold
    end
    local target_l = gold:size(1)
    for t = 2, target_l do
      local decoder_input1
      if model_opt.use_chars_dec == 1 then
        decoder_input1 = word2charidx_targ:index(1, gold[{{t-1}}])
      else
        decoder_input1 = gold[{{t-1}}]
      end
      local decoder_input
      if model_opt.attn == 1 then
        decoder_input = {decoder_input1, context[{{1}}], table.unpack(rnn_state_dec)}
      else
        decoder_input = {decoder_input1, context[{{1}, source_l}], table.unpack(rnn_state_dec)}
      end
      local out_decoder = model[2]:forward(decoder_input)
      local out = model[3]:forward(out_decoder[#out_decoder]) -- K x vocab_size
      rnn_state_dec = {} -- to be modified later
      if model_opt.input_feed == 1 then
        table.insert(rnn_state_dec, out_decoder[#out_decoder])
      end
      for j = 1, #out_decoder - 1 do
        table.insert(rnn_state_dec, out_decoder[j])
      end
      gold_score = gold_score + out[1][gold[t]]
    end
  end
  -- print(end_score, max_score)
  if opt.simple == 1 or end_score > max_score or not found_eos then
    max_hyp = end_hyp
    max_score = end_score
    max_attn_argmax = end_attn_argmax
  end

  local best_hyp=states[i]
  local best_scores=scores[i]
  local best_attn_argmax=attn_argmax[i]
  if opt.rescore ~= '' then
    local max_mscore = scorers[opt.rescore](max_hyp, gold_table, opt.rescore_param)
    print('RESCORE MAX '..opt.rescore..': '..max_mscore, 'BEST '..opt.rescore..': '..best_mscore)
    max_hyp=mscore_hyp
    max_score=best_mscore
    max_attn_argmax=mscore_attn_argmax
    best_hyp=mscore_hyp
    best_scores=mscore_scores
    best_attn_argmax=mscore_attn_argmax
  end

  return max_hyp, max_score, max_attn_argmax, gold_score, best_hyp, best_scores, best_attn_argmax
end

local READ = 1
local WRITE = 2
local DEBUG = false

local policy_state = 0
local policy = nil
local agent_model = nil
local agent_vocab = nil
local agent_vocab_size = nil

-- Given a source phrase and current output_phrase, 
-- return best beam and number of words to commit. 
-- If return value is negative, write everything
function n_words_policy(source_phrase, states, attn, scores, prev_ks, next_ys, num_beams, num_committed)
  policy_state = policy_state + 1
  if policy_state % opt.nwords_read == 0 then
    return -1, opt.nwords_write
  else
    return -1, 0
  end
end

local prev_ys

function wait_if_diff_policy(source_phrase, states, attn, scores, prev_ks, next_ys, num_beams, num_committed)
  policy_state = policy_state + 1
  count = 0
  if policy_state == 1 then
    prev_ys = next_ys:clone()
    return -1, 0
  else
    for k = num_committed+1, next_ys:size()[1] do
      count = k
      while ((next_ys[count][1] == prev_ys[count][1]) and count < next_ys:size()[1]) do
        count = count + 1
      end
      prev_ys = next_ys:clone()
      return -1, count - k
    end
  end
  return -1,0
end

-- this function is an implementatin of Cho's wait if worse method
local prev_ys
local prev_scores
function wait_if_worse_policy(source_phrase, states, attn, scores, prev_ks, next_ys, num_beams, num_committed, scores_history, i)
  policy_state = policy_state + 1
  count = 0
  if policy_state == 1 then
    prev_ys = next_ys:clone()
    return -1, 0
  else
    for k = num_committed+1, prev_ys:size()[1] do
      count = k
      while (count < prev_ys:size()[1] and scores_history[count][2][prev_ys[count][1]] >= scores_history[count][1][prev_ys[count][1]]) do
        count = count + 1
      end
      prev_ys = next_ys:clone()
      return 1, count - k
    end
  end
  return -1,0
end

-- this function is a modified form of wait if worse
local prev_ys
local prev_scores

function wait_if_diff_worse_policy(source_phrase, states, attn, scores, prev_ks, next_ys, num_beams, num_committed)
  policy_state = policy_state + 1
  count = 0
  if policy_state == 1 then
    prev_ys = next_ys:clone()
    prev_scores = scores:clone()
    return -1, 0
  else
    for k = num_committed+1, next_ys:size()[1] do
      -- TODO: This for is not needed. fixme.
      count = k
      while ((next_ys[count][1] == prev_ys[count][1]) and count < next_ys:size()[1] and scores[count][1] >= prev_scores[count][1]) do
        count = count + 1
      end
      prev_ys = next_ys:clone()
      prev_scores = scores:clone()
      return -1, count - k
    end
  end
  return -1,0
end

function n_words_constant_policy(source_phrase, states, attn, scores, prev_ks, next_ys, num_beams, num_committed)
  policy_state = policy_state + 1

  -- If we have not reached nwords_start yet
  -- OR we have reached nwords_start but have not read
  --   enough words yet
  if not (
    policy_state == opt.nwords_start 
    or (
      policy_state > opt.nwords_start 
      and (policy_state-opt.nwords_start) % opt.nwords_read == 0
    )
  ) then
    return -1,0
  end

  -- Select the best beam given our nwords_write
  local best_beam = -1
  local max_score = -1e9

  local best_beam_eos = -1
  local max_score_eos = -1e9

  -- Initially, we ignore any beams with EOS, since
  -- our agent is called only with partial source
  -- sentences. If all beams have EOS, we then pick
  -- the one with the best score
  -- Note that the EOS will NOT be commited in the
  -- main loop, since we explicitly know that the
  -- source has not ended
  for k = 1,num_beams do
    local num_commit = opt.nwords_write
    local curr_hyp = nil

    if num_committed+num_commit <= #states then
      curr_hyp = states[num_committed+num_commit][k]
    else
      -- This is the case where we are trying to write
      -- more than we predicted
      curr_hyp = states[#states][k]
    end

    -- If the last predicted word is EOS, we may have
    -- less than nwords_write words to write
    if curr_hyp[#curr_hyp] == END then
      num_commit = #curr_hyp - num_committed
    end
    assert(num_commit >= 0)

    if curr_hyp[#curr_hyp] == END then
      if scores[num_committed+num_commit][k] > max_score_eos then
        max_score_eos = scores[num_committed+num_commit][k]
        best_beam_eos = k
      end
    else
      if scores[num_committed+num_commit][k] > max_score then
        max_score = scores[num_committed+num_commit][k]
        best_beam = k
      end
    end
  end

  if best_beam == -1 then
    best_beam = best_beam_eos
    max_score = max_score_eos
  end

  -- Here we can return num_commit (which is a better estimation
  -- since nwords_write may overflow, but we do this check in the 
  -- main function anyways, so we just return nwords_write)
  if policy_state == opt.nwords_start then
    return best_beam, opt.nwords_write
  elseif policy_state > opt.nwords_start and (policy_state-opt.nwords_start) % opt.nwords_read == 0 then
    return best_beam, opt.nwords_write
  else
    -- We should never come here
    assert(false)
    return best_beam, 0
  end
end

function toOneHot_nobatch(x, num_feats)
  x_onehot = torch.CudaTensor(agent_vocab_size * num_feats):zero()
  for idx = 1, num_feats do
    x_onehot[(idx-1) * agent_vocab_size + x[idx]] = 1
  end
  return x_onehot
end

-- Agent policy
--   policy_state here maintains how many words have been read since
--   the last write. We want to keep an uncommitted context of atleast
--   two source words (hence we should never write if policy_state < 2)
local IS_START = 1
local start_words_so_far = 0
function agent_policy(source_phrase, states, attn, scores, prev_ks, next_ys, num_beams, num_committed)
  MIN_CONTEXT = 2

  if (IS_START == 1) then
    start_words_so_far = start_words_so_far + 1

    if start_words_so_far == 5 then
      IS_START = 0
      -- print("NO MORE START!!!!!")
    end
    return -1, 0
  end

  -- build agent input
  local num_context_words = math.min(5, source_phrase:size(1))
  local agent_input = torch.CudaTensor(5)

  for i=1,(5-num_context_words) do
    agent_input[i] = agent_vocab['<S>']
  end
  local start_idx = source_phrase:size(1) - num_context_words + 1
  for i=start_idx,source_phrase:size(1) do
    local inp_idx = i - start_idx + 5-num_context_words + 1
    if agent_vocab[idx2word_src[source_phrase[i]]] ~= nil then
      agent_input[inp_idx] = agent_vocab[idx2word_src[source_phrase[i]]]
    else
      agent_input[inp_idx] = agent_vocab['<unk>']
    end
  end
  
  -- run agent
  local agent_pred = agent_model:forward(toOneHot_nobatch(agent_input, 5))
  local _, pred_class = torch.max(agent_pred, 1)
  pred_class = pred_class[1]

  local potential_policy_state = policy_state + 1 -- since we have read one more word

  if (DEBUG) then
    print("Old Policy State: ", policy_state)
    print("Potential Policy State: ", potential_policy_state)
    print("Potential Writes: ", potential_policy_state - MIN_CONTEXT)
    print("Agent says: ", pred_class == 1 and "Segment" or "Don't Segment")
  end

  if potential_policy_state > MIN_CONTEXT and pred_class == 1 then
    local to_write = potential_policy_state - MIN_CONTEXT
    policy_state = MIN_CONTEXT

  --   return -1, to_write
  -- else
    -- Select the best beam given our nwords_write
    local best_beam = -1
    local max_score = -1e9

    for k = 1,num_beams do
      local num_commit = to_write
      local curr_hyp = nil
      if num_committed+num_commit <= #states then
        curr_hyp = states[num_committed+num_commit][k]
      else
        curr_hyp = states[#states][k]
      end
      if curr_hyp[#curr_hyp] == END then
        num_commit = #curr_hyp - num_committed
      end
      if scores[num_committed+num_commit][k] > max_score then
        max_score = scores[num_committed+num_commit][k]
        best_beam = k
      end
    end
    
    return best_beam, to_write
  else
    policy_state = potential_policy_state
    return -1, 0
  end
end

function full_source_policy(source_phrase, states, attn, scores, prev_ks, next_ys, num_beams, num_committed)
  return -1, 0
end

function generate_beam_stream(model, initial, K, max_sent_l, source, source_features, gold)
  -- Initialize policy
  if opt.policy == 'wue' then
    policy = full_source_policy
  elseif opt.policy == 'nwords' then
    policy = n_words_policy
    policy_state = 0
  elseif opt.policy == 'nwordsconstant' then
    policy = n_words_constant_policy
    policy_state = 0
  elseif opt.policy == 'agent' then
    policy = agent_policy
    policy_state = 0
    IS_START = 1
    start_words_so_far = 0
  elseif opt.policy == 'wid' then
    policy = wait_if_diff_policy
    policy_state = 0
    assert(K==1) -- policy only works for greedy decoding
  elseif opt.policy == 'widw' then
    policy = wait_if_diff_worse_policy
    policy_state = 0
    assert(K==1) -- policy only works for greedy decoding
  elseif opt.policy == 'wiw' then
    policy = wait_if_worse_policy
    policy_state = 0
  else
    policy = full_source_policy
  end

  -- keeping record of all the scores for two decoder runs
  -- This is useful for some policies that want to compare
  --   scores between two decoder runs such as wid, wiw etc.
  -- scores_history[..][2] always stores scores for the latest
  --   run, and scores_history[..][1] are the scores for
  --   the previous decoder run
  local scores_history = nil

  -- Reset decoder initial states
  if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
    cutorch.setDevice(opt.gpuid)
  end
  local n = max_sent_l
  -- Backpointer table. Stores which beam paths were used
  local prev_ks = torch.LongTensor(n, K):fill(1)
  -- Current States. Stores which words were selected at each timestep
  -- for each beam
  local next_ys = torch.LongTensor(n, K):fill(1)
  -- Current Scores. Stores the scores for the selected word
  -- at each timestep for each beam
  local scores = torch.FloatTensor(n, K)
  scores:zero()

  -- Cap source length to max_sent_l
  local source_l = math.min(source:size(1), opt.max_sent_l)

  -- Initialize Attention structures
  local attn_argmax = {} -- store attn weights
  attn_argmax[1] = {}

  -- states is somewhat redundant structure which globally
  -- stores computation so far
  local states = {} -- store predicted word idx
  states[1] = {}
  for k = 1, 1 do
    table.insert(states[1], initial)
    table.insert(attn_argmax[1], initial)
    next_ys[1][k] = State.next(initial)
  end

  -- Get appropriate view over source (for use_chars_enc == 0, just chaning
  -- from 1D to 2D tensor)
  local source_input
  if model_opt.use_chars_enc == 1 then
    source_input = source:view(source_l, 1, source:size(2)):contiguous()
  else
    source_input = source:view(source_l, 1)
  end

  -- Initialize encoder states (in normal conditions, 
  -- #rnn_state_enc = 4, 2 per LSTM layer, cell and hidden)
  local rnn_state_enc = {}
  for i = 1, #init_fwd_enc do
    table.insert(rnn_state_enc, init_fwd_enc[i]:zero())
  end

  -- Create local context from an already allocated space
  -- for efficiency
  -- 1 x source_l x rnn_size
  local context = context_proto[{{}, {1,source_l}}]:clone()

  -- context will always be in the following state
  -- [ intialized state1, intialized state2, ... intialized stateN, uninitialized states ...]
  -- Here "N" is the number of words READ so far

  -- Initialize RNN decoder
  rnn_state_dec = {}
  local commited_rnn_state_dec = {}
  for i = 1, #init_fwd_dec do
    table.insert(commited_rnn_state_dec, init_fwd_dec[i]:zero())
  end

  -- We cannot init_dec properly yet, so following is incorrect
  -- implementation
  -- if model_opt.init_dec == 1 then
  --   for L = 1, model_opt.num_layers do
  --     commited_rnn_state_dec[L*2-1+model_opt.input_feed]:copy(
  --       rnn_state_enc[L*2-1]:expand(K, model_opt.rnn_size))
  --     commited_rnn_state_dec[L*2+model_opt.input_feed]:copy(
  --       rnn_state_enc[L*2]:expand(K, model_opt.rnn_size))
  --   end
  -- end

  -- Main loop to READ words one by one, translate target everytime 
  -- since last commit, and decide to commit n words
  local source_pointer = 1 -- Always points to the word that is to be read
  local target_pointer = 1 -- Always points to the last "committed" space in target
  local fully_done = false -- boolean to signify if we are done

  -- Local variables for beam
  local max_score = -1e9
  local best_beam = -1

  -- Final stream hypothesis
  local stream_hyp = {}
  table.insert(stream_hyp, START)
  -- target_pointer is 1 since we have already committed START_OF_SENT

  -- Delay for current hypothesis
  local delay_hyp = {}
  -- This will be of the form {6, 6, 6, 7, 7} etc
  -- (How many source words did the ith target word wait for)
  -- Useful to compute AP

  -- TODO: Handle reading end of source
  while true and not fully_done do
    -- READ one word
    -- encoder_input has three things:
    -- current word, source_feats (if any), prev rnn_state
    local encoder_input = {source_input[source_pointer]}

    -- Handle multiple input featues
    if model_opt.num_source_features > 0 then
      append_table(encoder_input, source_features[t])
    end
    append_table(encoder_input, rnn_state_enc)

    -- Do a forward pass for the encoder
    local out = model[1]:forward(encoder_input)

    -- Debug print to print which word we just ready
    if (DEBUG) then print("READ", idx2word_src[source_input[source_pointer][1]]) end

    -- Reset rnn_state_enc with new computed state
    rnn_state_enc = out

    -- Add newly computed encoder state to context
    context[{{},source_pointer}]:copy(out[#out])

    -- "Clone context" K times for each of the K beams
    expanded_context = context:expand(K, source_l, model_opt.rnn_size)

    -- Get view of only initialized context states
    -- i.e. attention should ONLY see context from words read so far
    context_view = expanded_context[{{}, {1,source_pointer}, {}}]

    -- Translate since last commited
    local current_rnn_state_decs = {}

    -- if model_opt.init_dec == 1 then
    --   for L = 1, model_opt.num_layers do
    --     commited_rnn_state_dec[L*2-1+model_opt.input_feed]:copy(
    --       rnn_state_enc[L*2-1]:expand(K, model_opt.rnn_size))
    --     commited_rnn_state_dec[L*2+model_opt.input_feed]:copy(
    --       rnn_state_enc[L*2]:expand(K, model_opt.rnn_size))
    --   end
    -- end

    -- Restore Decoder hidden state to committed state
    -- By doing this, we are essentially "going back in time"
    -- to the moment we predicted the last committed word
    rnn_state_dec = deepcopy(commited_rnn_state_dec)
    
    out_float = torch.FloatTensor()
    local i = target_pointer
    local done = false
    local found_eos = false

    -- n here is max_sent_length
    -- We keep producing words until we are "done" or we have
    -- hit the max number of words we can produce
    max_score = -1e9
    while (not done) and (i < n) do
      -- i+1 points to the first "empty" space in the target sequence
      i = i+1

      -- Clear ith state (i+1, i+2, i+3 have some garbage from
      -- previous run, but we don't need to explicitly clear it
      -- for efficiency)
      states[i] = {}

      -- Clear old state from uncommitted translation
      -- We dont do this explicitly for efficiency
      -- scores[{{i,n},{}}]:zero()
      -- next_ys[{{i,n},{}}]:zero()
      -- prev_ks[{{i,n},{}}]:zero()

      -- Clear ith attention state
      attn_argmax[i] = {}

      -- Build decoder input
      local decoder_input1
      if model_opt.use_chars_dec == 1 then
        decoder_input1 = word2charidx_targ:index(1, next_ys:narrow(1,i-1,1):squeeze())
      else
        decoder_input1 = next_ys:narrow(1,i-1,1):squeeze()
        if opt.beam == 1 then
          decoder_input1 = torch.LongTensor({decoder_input1})
        end
      end
      -- decoder_input1 is previously predicted word(s) per beam

      if (DEBUG) then 
        local decsum = 0
        for idx = 1,#rnn_state_dec do
          decsum = decsum + rnn_state_dec[idx]:sum()
        end
        print("Decoder input:", decoder_input1:sum(), context_view:sum(), decsum)
      end

      -- Build rest of the decoder input:
      -- previous words, attention input, previous decoder states
      -- Remember, decoder includes attention mechanism
      local decoder_input
      if model_opt.attn == 1 then
        decoder_input = {decoder_input1, context_view, table.unpack(rnn_state_dec)}
      else
        decoder_input = {decoder_input1, context_view[{{}, source_pointer}], table.unpack(rnn_state_dec)}
      end

      -- Forward pass on the attention mechanism and decoder
      local out_decoder = model[2]:forward(decoder_input)

      local out_pred_idx = #out_decoder
      if model_opt.guided_alignment == 1 then
          out_pred_idx = #out_decoder - 1
      end

      -- model[3] is the generator model
      local out = model[3]:forward(out_decoder[out_pred_idx])
      -- out is scores per beam [K x vocab_size]

      -- Get new RNN Decoder state
      rnn_state_dec = {} -- to be modified later
      if model_opt.input_feed == 1 then
        table.insert(rnn_state_dec, out_decoder[out_pred_idx])
      end
      for j = 1, out_pred_idx - 1 do
        table.insert(rnn_state_dec, out_decoder[j])
      end

      -- Convert scores to float tensor explicitly
      out_float:resize(out:size()):copy(out)
      -- out_float is also of size K x vocab_size

      for k = 1, K do
        -- Explicitly not allow "PAD" and "START_OF_SENT" to be 
        -- max scores
        -- Add scores from previous hypothesis (adding in logspace)
        State.disallow(out_float:select(1, k))
        out_float[k]:add(scores[i-1][k])
      end

      -- All the scores are now available.

      local flat_out = out_float:view(-1)

      -- Initialize scores_history if it has not been intialized yet
      if scores_history == nil then
        scores_history = torch.FloatTensor(n, 2, out_float:size(2)):fill(2)
        -- set it 2 to be larger than maximum probability all the time
        -- max_sent_length x two time stamps x vocab_size
      end
      scores_history[i][2] = out_float[1]

      -- If we have not predicted anything so far (only START OF SENT)
      -- all beams will have EXACTLY the same scores - so consider only
      -- the first beam (only when i=2)!
      if i == 2 then
        flat_out = out_float[1] -- all outputs same for first batch
      end

      -- Only if you use special start symbol, we don't usually do
      -- this
      if model_opt.start_symbol == 1 then
        decoder_softmax.output[{{},1}]:zero()
        decoder_softmax.output[{{},source_l}]:zero()
      end

      -- For each beam, select the next "best" choice for current timestep
      --            +------------> ----+------>
      --            |                  |
      --            +------------> -+  +------>
      --            |               |
      -- +----------------------->  +--------->
      --            |
      --            +------------> ----------->
      --            |
      --            +------------> ----------->
      -- 
      --    t=0          t=1            t=2
      -- In general, for any beam k, previous decoding path can be
      -- from any other beam
      -- In the above example, at t=2, we discard beam 3 completely,
      -- and use beam 1 twice
      for k = 1, K do
        while true do
          -- Select the maximum scores from scores of ALL prev beams
          -- Q: Are scores across beams comparable? Does score of 
          --     0.89 in beam 1 == score of 0.89 in beam 2?
          -- Potential A: Yes, since they are still over same vocab
          -- This assumption has been made here.
          local score, index = flat_out:max(1)
          local score = score[1]

          -- prev_k = beam that predicted max scoring word
          -- y_i = max scoring word index within beam
          local prev_k, y_i = flat_to_rc(out_float, index[1])
          states[i][k] = State.advance(states[i-1][prev_k], y_i)

          -- Check if the newly selected word causes the current
          -- beam to be equal to any other beam. If it is the same
          -- we dont want to select the current word, since we want
          -- the beams to be different.
          local diff = true
          for k2 = 1, k-1 do
            if State.same(states[i][k2], states[i][k]) then
              diff = false
            end
          end

          -- If the newly selected word DOES NOT cause the current
          -- beam to be the same as any other beam (or we are at the
          -- first target word, go ahead and mark this beam as done)
          if i < 2 or diff then
            if model_opt.attn == 1 then
              max_attn, max_index = decoder_softmax.output[prev_k]:max(1)
              attn_argmax[i][k] = State.advance(attn_argmax[i-1][prev_k],max_index[1])
            end
            prev_ks[i][k] = prev_k
            next_ys[i][k] = y_i
            scores[i][k] = score
            flat_out[index[1]] = -1e9
            break -- move on to next k
          end

          -- We come here if we have not finalized this beam, i.e.
          -- it was the same as some other beam. This failure happened
          -- because of currently selected max_word, so lets set its
          -- score to a very small value.
          flat_out[index[1]] = -1e9
        end
      end

      -- New RNN states will depend on which beams we are "moving forward"
      -- with - e.g. in the above example we will discard RNN states for
      -- beam 3 (and RNN state for beam 1 will be replicated twice)
      for j = 1, #rnn_state_dec do
        rnn_state_dec[j]:copy(rnn_state_dec[j]:index(1, prev_ks[i]))
      end

      if (DEBUG) then
        local decsum = 0
        for idx = 1,#rnn_state_dec do
          decsum = decsum + rnn_state_dec[idx]:sum()
        end
        print("Decoder output state:", decsum)
      end

      -- Save RNN state decoder
      table.insert(current_rnn_state_decs, deepcopy(rnn_state_dec))

      -- Beam 1 will ALWAYS be the best beam, lets assume its
      -- our final answer
      end_hyp = states[i][1]
      end_score = scores[i][1]
      if model_opt.attn == 1 then
        end_attn_argmax = attn_argmax[i][1]
      end
      
      -- If Beam 1 actually ends i.e. has EOS symbol, our assumption
      -- is correct and we finish decoding
      if end_hyp[#end_hyp] == END then
        done = true
        found_eos = true
      else
        -- If Beam 1 does not have an EOS symbol, its possible that
        --   some other beam does have and EOS symbol.
        -- We will consider all beams with EOS symbol, and pick the
        --   one with max_score
        -- BUT we still continue decoding in the hope that Beam 1
        --   will produce EOS at some point. If that never happens
        --   max_{hyp,score} will come into account
        for k = 1, K do
          local possible_hyp = states[i][k]
          if possible_hyp[#possible_hyp] == END then
            found_eos = true
            if scores[i][k] > max_score then
              max_hyp = possible_hyp
              max_score = scores[i][k]
              best_beam = k
              if model_opt.attn == 1 then
                max_attn_argmax = attn_argmax[i][k]
              end
            end
          end
        end
      end
    end

    -- Here, we have the following variables now:
    -- end_{hyp,score,attn_argmax} == Beam 1
    -- max_{hyp,score,attn_argmax} == undefined if Beam 1 had EOS
    --                             == some BEAM if Beam 1 never had EOS

    local best_mscore = -1e9
    local mscore_hyp
    local mscore_scores
    local mscore_attn_argmax
    local gold_table
    
    -- opt.simple is usually 0
    -- If EOS was found (by any beam) AND that beam had a good score,
    -- we keep that beam (good score means more than Beam 1)
    -- If EOS was not found at all, Beam 1 is our best shot, so set
    -- max_{*} = end_{*}
    if opt.simple == 1 or end_score > max_score or not found_eos then
      best_beam = 1
      -- TODO: Sometimes, max_hyp is BOS word word word EOS word
      --   in this case, the following check will fail. Investigate
      --   further.
      max_hyp = end_hyp
      max_score = end_score
      max_attn_argmax = end_attn_argmax
    end

    local best_hyp=states[i]
    local best_scores=scores[i]
    local best_attn_argmax=attn_argmax[i]

    if (DEBUG) then print("Lengths",#current_rnn_state_decs, target_pointer, #max_hyp) end

    -- max_{*} now has the best hypothesis (could be any beam)

    -- Call Policy
    local to_write = 0
    local num_new_words = #max_hyp - target_pointer

    -- If we have READ all words in the source sentence, to_write is all words
    if source_pointer == source_l then
      to_write = num_new_words
      fully_done = true
    else
      -- If we are here, it means we have NOT read all
      -- source words yet, so we should never commit EOS
      best_beam_policy, to_write = policy(source[{{1,source_pointer}}], states, attn_argmax, scores, prev_ks, next_ys, K, target_pointer, scores_history, i)

      -- policy will return best_beam and how many words it thinks we should
      -- write
      -- best_beam can be -1, in which consider it to be max_hyp
      -- If it is not, lets change max_hyp to one that beam says
      if best_beam_policy ~= -1 then
        best_beam = best_beam_policy
        idx = target_pointer+to_write
        if idx > #states then
          -- This happens when the policy is dumb, i.e. it doesn't
          -- check if to_write number of writes is even possible
          idx = #states
        end
        max_hyp = states[idx][best_beam]
        -- TODO: max_score and max_attn_argmax should be set!
      end

      -- If to_write < 0, it means write everything
      if to_write < 0 then
        -- If the last state is EOS, we do not write the EOS
        if max_hyp[#max_hyp] == END then
          to_write = num_new_words - 1
        else
          -- This will only happen when we go into infinite generation mode
          to_write = num_new_words
        end
      end

      -- This will happen when the policy is dumb, i.e. it doesn't
      -- check if to_write number of writes is even possible
      if to_write >= num_new_words then
        -- TODO: Sometimes, max_hyp is BOS word word word EOS word
        --   in this case, the following check will fail. Investigate
        --   further.
        if max_hyp[#max_hyp] == END then
          to_write = num_new_words - 1
        else
          -- This will only happen when we go into infinite generation mode
          to_write = num_new_words
        end
      end
    end

    if (DEBUG) then
      print("Number of new words: ",num_new_words)
      print("Number of words to write: ",to_write)
      print("Number of saved states:",#current_rnn_state_decs)
    end

    -- Commit if required
    if (to_write > 0) then
      if (DEBUG) then print(to_write) end
      if (DEBUG) then print(unpack(max_hyp, target_pointer+1, target_pointer+to_write)) end

      -- Write to_write words from target_pointer+1
      local committed_pointer = target_pointer + to_write

      for target_idx = target_pointer+1, committed_pointer do
        table.insert(stream_hyp, max_hyp[target_idx])
        if (DEBUG) then print("WRITE",idx2word_targ[max_hyp[target_idx]]) end
        if opt.delay_file ~= '' then
          table.insert(delay_hyp, source_pointer)
        end
      end

      -- Consider the to_write'th time step for the next "committed" RNN
      -- decoder state
      commited_rnn_state_dec = deepcopy(current_rnn_state_decs[to_write])

      -- Collapse beams to best beam
      -- For each layer, copy best beam onto every beam
      for l = 1,#commited_rnn_state_dec do
        for k = 1, K do
          commited_rnn_state_dec[l][{{k},{}}]:copy(commited_rnn_state_dec[l][{{best_beam},{}}])
        end
      end

      -- For each beam, set the last prev_ks, next_ys and scores to the one of the
      -- best beam
      for k = 1, K do
        prev_ks[{{committed_pointer}, {k}}]:copy(prev_ks[{{committed_pointer}, {best_beam}}])
        next_ys[{{committed_pointer}, {k}}]:copy(next_ys[{{committed_pointer}, {best_beam}}])
        scores[{{committed_pointer}, {k}}]:copy(scores[{{committed_pointer}, {best_beam}}])
        
        for i = 1,#states[committed_pointer][best_beam] do
          states[committed_pointer][k][i] = states[committed_pointer][best_beam][i]
        end

        for i = 1,#attn_argmax[committed_pointer][best_beam] do
          attn_argmax[committed_pointer][k][i] = attn_argmax[committed_pointer][best_beam][i]
        end
      end
      
      if (DEBUG) then
        local decsum = 0
        for idx = 1,#commited_rnn_state_dec do
          decsum = decsum + commited_rnn_state_dec[idx]:sum()
        end
        print("Committing dec state:", decsum)
        print("Committing dec index:", to_write)
      end
      
      -- Set target pointer to correct location
      target_pointer = committed_pointer
    end

    -- keeping scores of previous decoder run
    scores_history[{{},{1},{}}] = scores_history[{{},{2},{}}]

    -- Increment source pointer
    source_pointer = source_pointer + 1
  end
  if (DEBUG) then print(stream_hyp) end
  return stream_hyp, max_score, max_attn_argmax, gold_score, best_hyp, best_scores, best_attn_argmax, delay_hyp
end

function idx2key(file)
  local f = io.open(file,'r')
  local t = {}
  for line in f:lines() do
    local c = {}
    for w in line:gmatch'([^%s]+)' do
      table.insert(c, w)
    end
    t[tonumber(c[2])] = c[1]
  end
  return t
end

function flip_table(u)
  local t = {}
  for key, value in pairs(u) do
    t[value] = key
  end
  return t
end

function get_layer(layer)
  if layer.name ~= nil then
    if layer.name == 'decoder_attn' then
      decoder_attn = layer
    elseif layer.name:sub(1,3) == 'hop' then
      hop_attn = layer
    elseif layer.name:sub(1,7) == 'softmax' then
      table.insert(softmax_layers, layer)
    elseif layer.name == 'word_vecs_enc' then
      word_vecs_enc = layer
    elseif layer.name == 'word_vecs_dec' then
      word_vecs_dec = layer
    end
  end
end

local function features2featureidx(features, feature2idx, start_symbol)
  local out = {}

  if start_symbol == 1 then
    table.insert(out, {})
    for j = 1, #feature2idx do
      table.insert(out[#out], torch.Tensor(1):fill(START))
    end
  end

  for i = 1, #features do
    table.insert(out, {})
    for j = 1, #feature2idx do
      local value = feature2idx[j][features[i][j]]
      if value == nil then
        value = UNK
      end
      table.insert(out[#out], torch.Tensor(1):fill(value))
    end
  end

  if start_symbol == 1 then
    table.insert(out, {})
    for j = 1, #feature2idx do
      table.insert(out[#out], torch.Tensor(1):fill(END))
    end
  end

  return out
end

function sent2wordidx(sent, word2idx, start_symbol)
  local t = {}
  local u = {}
  if start_symbol == 1 then
    table.insert(t, START)
    table.insert(u, START_WORD)
  end

  for word in sent:gmatch'([^%s]+)' do
    local idx = word2idx[word] or UNK
    table.insert(t, idx)
    table.insert(u, word)
  end
  if start_symbol == 1 then
    table.insert(t, END)
    table.insert(u, END_WORD)
  end
  return torch.LongTensor(t), u
end

function sent2charidx(sent, char2idx, max_word_l, start_symbol)
  local words = {}
  if start_symbol == 1 then
    table.insert(words, START_WORD)
  end
  for word in sent:gmatch'([^%s]+)' do
    table.insert(words, word)
  end
  if start_symbol == 1 then
    table.insert(words, END_WORD)
  end
  local chars = torch.ones(#words, max_word_l)
  for i = 1, #words do
    chars[i] = word2charidx(words[i], char2idx, max_word_l, chars[i])
  end
  return chars, words
end

function word2charidx(word, char2idx, max_word_l, t)
  t[1] = START
  local i = 2
  for _, char in utf8.next, word do
    char = utf8.char(char)
    local char_idx = char2idx[char] or UNK
    t[i] = char_idx
    i = i+1
    if i >= max_word_l then
      t[i] = END
      break
    end
  end
  if i < max_word_l then
    t[i] = END
  end
  return t
end

function wordidx2sent(sent, idx2word, source_str, attn, skip_end)
  local t = {}
  local start_i, end_i
  skip_end = skip_start_end or true
  if skip_end then
    end_i = #sent-1
  else
    end_i = #sent
  end
  for i = 2, end_i do -- skip START and END
    if sent[i] == UNK then
      if opt.replace_unk == 1 then
        local s = source_str[attn[i]]
        if phrase_table[s] ~= nil then
          print(s .. ':' ..phrase_table[s])
        end
        local r = phrase_table[s] or s
        table.insert(t, r)
      else
        table.insert(t, idx2word[sent[i]])
      end
    else
      table.insert(t, idx2word[sent[i]])
    end
  end
  return table.concat(t, ' ')
end

function clean_sent(sent)
  local s = stringx.replace(sent, UNK_WORD, '')
  s = stringx.replace(s, START_WORD, '')
  s = stringx.replace(s, END_WORD, '')
  s = stringx.replace(s, START_CHAR, '')
  s = stringx.replace(s, END_CHAR, '')
  return s
end

function strip(s)
  return s:gsub("^%s+",""):gsub("%s+$","")
end

local function extract_features(line)
  local cleaned_tokens = {}
  local features = {}

  for entry in line:gmatch'([^%s]+)' do
    local field = entry:split('%-|%-')
    local word = clean_sent(field[1])
    if string.len(word) > 0 then
      table.insert(cleaned_tokens, word)

      if #field > 1 then
        table.insert(features, {})
      end

      for i= 2, #field do
        table.insert(features[#features], field[i])
      end
    end
  end

  return cleaned_tokens, features
end

function init(arg)
  -- parse input params
  opt = cmd:parse(arg)

  -- some globals
  PAD = 1; UNK = 2; START = 3; END = 4
  PAD_WORD = '<blank>'; UNK_WORD = '<unk>'; START_WORD = '<s>'; END_WORD = '</s>'
  START_CHAR = '{'; END_CHAR = '}'
  MAX_SENT_L = opt.max_sent_l

  assert(path.exists(opt.model), 'model does not exist')

  if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    if opt.cudnn == 1 then
      require 'cudnn'
    end
  end
  print('loading ' .. opt.model .. '...')
  checkpoint = torch.load(opt.model)
  print('done!')

  if opt.replace_unk == 1 then
    phrase_table = {}
    if path.exists(opt.srctarg_dict) then
      local f = io.open(opt.srctarg_dict,'r')
      for line in f:lines() do
        local c = line:split("|||")
        phrase_table[strip(c[1])] = c[2]
      end
    end
  end

  if opt.rescore ~= '' then
    require 's2sa.scorer'
    if not scorers[opt.rescore] then
      error("metric "..opt.rescore.." not defined")
    end
  end

  if opt.policy == 'agent' then
    -- Load agent model/vocab if we are in first run
    assert(path.exists(opt.agent_model), 'agent model does not exist')
    assert(path.exists(opt.agent_vocab), 'agent vocab does not exist')

    print("Loading agent dictionary...")
    local vocab = io.open(opt.agent_vocab)
    agent_vocab_size = 0
    agent_vocab = {}
    for line in vocab:lines() do
      space_idx = line:find(" ")
      w, w_idx = line:sub(0, space_idx-1), line:sub(space_idx+1)
      agent_vocab[w] = w_idx
      agent_vocab_size = agent_vocab_size + 1
    end

    print("Loading agent model...")
    agent_model = torch.load(opt.agent_model)
  end

  -- load model and word2idx/idx2word dictionaries
  model, model_opt = checkpoint[1], checkpoint[2]
  for i = 1, #model do
    model[i]:evaluate()
  end
  -- for backward compatibility
  model_opt.brnn = model_opt.brnn or 0
  model_opt.input_feed = model_opt.input_feed or 1
  model_opt.attn = model_opt.attn or 1
  model_opt.num_source_features = model_opt.num_source_features or 0

  idx2word_src = idx2key(opt.src_dict)
  word2idx_src = flip_table(idx2word_src)
  idx2word_targ = idx2key(opt.targ_dict)
  word2idx_targ = flip_table(idx2word_targ)

  idx2feature_src = {}
  feature2idx_src = {}

  for i = 1, model_opt.num_source_features do
    table.insert(idx2feature_src, idx2key(opt.feature_dict_prefix .. '.source_feature_' .. i .. '.dict'))
    table.insert(feature2idx_src, flip_table(idx2feature_src[i]))
  end

  -- load character dictionaries if needed
  if model_opt.use_chars_enc == 1 or model_opt.use_chars_dec == 1 then
    utf8 = require 'lua-utf8'
    char2idx = flip_table(idx2key(opt.char_dict))
    model[1]:apply(get_layer)
  end
  if model_opt.use_chars_dec == 1 then
    word2charidx_targ = torch.LongTensor(#idx2word_targ, model_opt.max_word_l):fill(PAD)
    for i = 1, #idx2word_targ do
      word2charidx_targ[i] = word2charidx(idx2word_targ[i], char2idx,
        model_opt.max_word_l, word2charidx_targ[i])
    end
  end
  -- load gold labels if it exists
  if path.exists(opt.targ_file) then
    print('loading GOLD labels at ' .. opt.targ_file)
    gold = {}
    local file = io.open(opt.targ_file, 'r')
    for line in file:lines() do
      table.insert(gold, line)
    end
  else
    if opt.score_gold == 1 then
      error('missing targ_file option to calculate gold scores')
    end
  end

  if opt.gpuid >= 0 then
    cutorch.setDevice(opt.gpuid)
    for i = 1, #model do
      if opt.gpuid2 >= 0 then
        if i == 1 or i == 4 then
          cutorch.setDevice(opt.gpuid)
        else
          cutorch.setDevice(opt.gpuid2)
        end
      end
      model[i]:double():cuda()
      model[i]:evaluate()
    end
  end

  softmax_layers = {}
  model[2]:apply(get_layer)
  if model_opt.attn == 1 then
    decoder_attn:apply(get_layer)
    decoder_softmax = softmax_layers[1]
    attn_layer = torch.zeros(opt.beam, MAX_SENT_L)
  end

  context_proto = torch.zeros(1, MAX_SENT_L, model_opt.rnn_size)
  local h_init_dec = torch.zeros(opt.beam, model_opt.rnn_size)
  local h_init_enc = torch.zeros(1, model_opt.rnn_size)
  if opt.gpuid >= 0 then
    h_init_enc = h_init_enc:cuda()
    h_init_dec = h_init_dec:cuda()
    cutorch.setDevice(opt.gpuid)
    if opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid)
      context_proto = context_proto:cuda()
      cutorch.setDevice(opt.gpuid2)
      context_proto2 = torch.zeros(opt.beam, MAX_SENT_L, model_opt.rnn_size):cuda()
    else
      context_proto = context_proto:cuda()
    end
    if model_opt.attn == 1 then
      attn_layer = attn_layer:cuda()
    end
  end
  init_fwd_enc = {}
  init_fwd_dec = {} -- initial context
  if model_opt.input_feed == 1 then
    table.insert(init_fwd_dec, h_init_dec:clone())
  end

  for L = 1, model_opt.num_layers do
    table.insert(init_fwd_enc, h_init_enc:clone())
    table.insert(init_fwd_enc, h_init_enc:clone())
    table.insert(init_fwd_dec, h_init_dec:clone()) -- memory cell
    table.insert(init_fwd_dec, h_init_dec:clone()) -- hidden state
  end

  pred_score_total = 0
  gold_score_total = 0
  pred_words_total = 0
  gold_words_total = 0

  State = StateAll
  sent_id = 0
end

function search(line)
  sent_id = sent_id + 1
  local cleaned_tokens, source_features_str = extract_features(line)
  local cleaned_line = table.concat(cleaned_tokens, ' ')
  print('SENT ' .. sent_id .. ': ' ..line)
  local source, source_str
  local source_features = features2featureidx(source_features_str, feature2idx_src, model_opt.start_symbol)
  if model_opt.use_chars_enc == 0 then
    source, source_str = sent2wordidx(cleaned_line, word2idx_src, model_opt.start_symbol)
  else
    source, source_str = sent2charidx(cleaned_line, char2idx, model_opt.max_word_l, model_opt.start_symbol)
  end
  if gold then
    target, target_str = sent2wordidx(gold[sent_id], word2idx_targ, 1)
  end
  state = State.initial(START)
  if opt.stream == 1 then
    pred, pred_score, attn, gold_score, all_sents, all_scores, all_attn, pred_delay = generate_beam_stream(model,
      state, opt.beam, MAX_SENT_L, source, source_features, target)
  else
    pred, pred_score, attn, gold_score, all_sents, all_scores, all_attn = generate_beam(model,
      state, opt.beam, MAX_SENT_L, source, source_features, target)
  end
  pred_score_total = pred_score_total + pred_score
  pred_words_total = pred_words_total + #pred - 1
  pred_sent = wordidx2sent(pred, idx2word_targ, source_str, attn, true)

  print('PRED ' .. sent_id .. ': ' .. pred_sent)
  if gold ~= nil then
    print('GOLD ' .. sent_id .. ': ' .. gold[sent_id])
    if opt.score_gold == 1 then
      print(string.format("PRED SCORE: %.4f, GOLD SCORE: %.4f", pred_score, gold_score))
      gold_score_total = gold_score_total + gold_score
      gold_words_total = gold_words_total + target:size(1) - 1
    end
  end

  nbests = {}

  if opt.n_best > 1 then
    for n = 1, opt.n_best do
      pred_sent_n = wordidx2sent(all_sents[n], idx2word_targ, source_str, all_attn[n], false)
      local out_n = string.format("%d ||| %s ||| %.4f", n, pred_sent_n, all_scores[n])
      print(out_n)
      nbests[n] = out_n
    end
  end

  print('')

  if opt.delay_file ~= '' then
    -- Remove EOS token delay
    table.remove(pred_delay, #pred_delay)
    return pred_sent, nbests, table.concat(pred_delay, ' ')
  else
    return pred_sent, nbests
  end
end

function getOptions()
  return opt
end

return {
  init = init,
  search = search,
  getOptions = getOptions
}
