-- load libraries
require 'nn'
require 'torch'
require 'xlua'
require 'optim'

-- Function to tokenize line into individual elements
function string:splitAtTabs()
	local sep, values = "\t", {}
	local pattern = string.format("([^%s]+)", sep)
	self:gsub(pattern, function(c) values[#values+1] = c end)
	return values
end

function main()
	-- Parse arguments
	local cmd = torch.CmdLine()
	cmd:option('-input', '', [[Path to Input data]])
	cmd:option('-output', './', [[Path to Output processed data]])
	cmd:option('-num_feats', 5, [[Number of Input features]])
	cmd:option('-shuffle', true, [[Shuffle datasets before creating splits]])
	local opt = cmd:parse(arg)

	-- Check input/output directories
	if opt.input == '' or not path.exists(opt.input) then
		print("Input file missing")
		os.exit()
	end
	data_file = opt.input

	if not path.exists(opt.output) then
		print("Output directory does not exist")
	end

	-- Build dictionary
	print("Building dictionary...")
	num_lines = 0
	dict_to_idx = {}
	idx_to_dict = {}
	curr_idx = 1
	dict_to_idx["<unk>"] = curr_idx
	idx_to_dict[curr_idx] = "<unk>"
	curr_idx = curr_idx + 1
	dict_to_idx["<bos>"] = curr_idx
	idx_to_dict[curr_idx] = "<bos>"
	curr_idx = curr_idx + 1
	dict_to_idx["<eos>"] = curr_idx
	idx_to_dict[curr_idx] = "<eos>"
	curr_idx = curr_idx + 1

	for line in io.lines(data_file) do
		for idx,word in pairs(line:splitAtTabs()) do
			if idx > opt.num_feats then
				break
			end
			if dict_to_idx[word] == nil then
				dict_to_idx[word] = curr_idx
				idx_to_dict[curr_idx] = word
				curr_idx = curr_idx + 1
			end
		end
		num_lines = num_lines + 1
	end

	vocabSize = curr_idx-1

	print("Vocab size:", vocabSize)
	print("Number of Lines:", num_lines)

	print("Building dataset...");
	X = torch.Tensor(num_lines, opt.num_feats):zero()
	y = torch.Tensor(num_lines, 1)

	local sample_idx = 1
	for line in io.lines(data_file) do
		if sample_idx % 1000 == 0 then
			xlua.progress(sample_idx, num_lines)
		end
		local label
		for idx, word in pairs(line:splitAtTabs()) do
			if idx > opt.num_feats then
				label = word
				break
			end
			X[{{sample_idx}, {idx}}] = dict_to_idx[word]
		end

		if label == 'S' then
			y[{{sample_idx}, {1}}] = 1
		elseif label == 'DS' then
			y[{{sample_idx}, {1}}] = 2
		else
			error("[ERROR] Input contains erroneous label!")
		end
		sample_idx = sample_idx + 1
	end
	xlua.progress(num_lines, num_lines)
	print("")

	function dict_to_idx:size() return vocabSize end
	function idx_to_dict:size() return vocabSize end

	print("Preparing train/test splits...")
	num_examples = num_lines

	local shuffleIdx = nil
	if opt.shuffle then
		shuffleIdx = torch.randperm(num_examples)
	else
		shuffleIdx = torch.round(1, num_examples)
	end
	
	num_train = torch.round(0.7*num_examples)
	num_dev = torch.round(0.15*num_examples)
	num_test = num_examples - num_train - num_dev

	X_train = torch.Tensor(num_train, opt.num_feats)
	y_train = torch.Tensor(num_train, 1)

	X_dev = torch.Tensor(num_dev, opt.num_feats)
	y_dev = torch.Tensor(num_dev, 1)

	X_test = torch.Tensor(num_test, opt.num_feats)
	y_test = torch.Tensor(num_test, 1)

	for idx=1, num_train do
		if idx % 1000 == 0 then
			xlua.progress(idx, num_examples)
		end
		X_train[{{idx},{}}] = X[{{shuffleIdx[idx]},{}}]
		y_train[{{idx},{}}] = y[{{shuffleIdx[idx]},{}}]
	end

	for idx=num_train+1, num_train+num_dev do
		if idx % 1000 == 0 then
			xlua.progress(idx, num_examples)
		end
		X_dev[{{idx - num_train},{}}] = X[{{shuffleIdx[idx]},{}}]
		y_dev[{{idx - num_train},{}}] = y[{{shuffleIdx[idx]},{}}]
	end

	for idx=num_train+num_dev+1, num_examples do
		if idx % 1000 == 0 then
			xlua.progress(idx, num_examples)
		end
		X_test[{{idx - num_train - num_dev},{}}] = X[{{shuffleIdx[idx]},{}}]
		y_test[{{idx - num_train - num_dev},{}}] = y[{{shuffleIdx[idx]},{}}]
	end
	xlua.progress(num_examples, num_examples)

	print("Number of training examples: " .. X_train:size()[1])
	print("Number of development examples: " .. X_dev:size()[1])
	print("Number of test examples: " .. X_test:size()[1])
	print("")

	print("Saving vocabulary")
	vocab_file = io.open(paths.concat(opt.output, "agent.vocab"), "w")
	for idx = 1, idx_to_dict:size() do
		vocab_file:write(idx_to_dict[idx] .. " " .. idx .. "\n")
	end
	vocab_file:close()

	print("Saving splits")
	torch.save(paths.concat(opt.output, "agent.data.train"), X_train)
	torch.save(paths.concat(opt.output, "agent.labels.train"), y_train)
	torch.save(paths.concat(opt.output, "agent.data.dev"), X_dev)
	torch.save(paths.concat(opt.output, "agent.labels.dev"), y_dev)
	torch.save(paths.concat(opt.output, "agent.data.test"), X_test)
	torch.save(paths.concat(opt.output, "agent.labels.test"), y_test)
end

main()