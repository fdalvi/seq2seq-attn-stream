-- load libraries
require 'nn'
require 'torch'
require 'xlua'
require 'optim'

agent = require('classifier')

function main()
	-- Parse arguments
	local cmd = torch.CmdLine()
	cmd:option('-input', '', [[Path to data directory with test sets]])
	cmd:option('-model', '', [[Path to model]])
	cmd:option('-use_gpu', true, [[Whether the computation should be done on a GPU]])
	cmd:option('-many_hot', true, [[Whether to use Many hot representation]])
	local opt = cmd:parse(arg)

	if opt.input == '' or not path.exists(opt.input) then
		error("[ERROR] Input Test directory missing")
	end
	if not path.exists(paths.concat(opt.input, "agent.vocab")) then
		error("[ERROR] Input vocabulary missing")
	end
	if not path.exists(paths.concat(opt.input, "agent.data.test")) or
		not path.exists(paths.concat(opt.input, "agent.labels.test")) then
		error("[ERROR] Test set missing")
	end
	if not path.exists(opt.model) then
		error("Trained model missing!")
	end

	if opt.use_gpu then
		require 'cunn'
		require 'cutorch'
	end

	print("Loading dictionary...")
	local vocab = io.open(paths.concat(opt.input, "agent.vocab"))
	vocab_size = 0
	for line in vocab:lines() do
		vocab_size = vocab_size + 1
	end

	print("Loading test data...")
	X_test = torch.load(paths.concat(opt.input, "agent.data.test"))
	y_test = torch.load(paths.concat(opt.input, "agent.labels.test"))

	-- X_test = X_test:cuda()
	-- y_test = y_test:cuda()

	local num_feats = X_test:size(2)

	local n_input
	if opt.many_hot then
		n_input = vocab_size -- many hot
	else
		n_input = num_feats * vocab_size
	end
	batch_size = 1024

	print("Loading model...")
	model = torch.load(opt.model)

	print("Loading complete:")
	print(model)
	print("Vocab Size:", vocab_size)
	print("Number of features:", num_feats)

	num_correct, num_correct_S, num_S, num_correct_DS, num_DS = agent.test(X_test, y_test, model, n_input, num_feats, opt.use_gpu, opt.many_hot, vocab_size, batch_size, true)

	print("Accuracy: " .. num_correct/X_test:size(1))
	print("S Accuracy: " .. num_correct_S/num_S)
	print("DS Accuracy: " .. num_correct_DS/num_DS)
end

main()