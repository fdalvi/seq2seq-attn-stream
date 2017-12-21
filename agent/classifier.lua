-- load libraries
require 'nn'
require 'torch'
require 'xlua'
require 'optim'

local exports = {}

function train(opt)
	-- Check input paths
	if opt.input == '' or not path.exists(opt.input) then
		error("[ERROR] Input directory missing")
	end
	if not path.exists(paths.concat(opt.input, "agent.vocab")) then
		error("[ERROR] Input vocabulary missing")
	end
	if not path.exists(paths.concat(opt.input, "agent.data.train")) or
		not path.exists(paths.concat(opt.input, "agent.labels.train")) then
		error("[ERROR] Training set missing")
	end
	if not path.exists(paths.concat(opt.input, "agent.data.dev")) or
		not path.exists(paths.concat(opt.input, "agent.labels.dev")) then
		error("[ERROR] Development set missing")
	end
	if not path.exists(paths.concat(opt.input, "agent.data.test")) or
		not path.exists(paths.concat(opt.input, "agent.labels.test")) then
		error("[ERROR] Test set missing")
	end
	if not path.exists(opt.output) then
		error("Output directory does not exist")
	end

	if opt.use_gpu then
		require 'cunn'
		require 'cutorch'
	end

	-- Load data
	print("Loading dictionary...")
	local vocab = io.open(paths.concat(opt.input, "agent.vocab"))
	vocab_size = 0
	for line in vocab:lines() do
		vocab_size = vocab_size + 1
	end

	print("Loading splits...")
	X_train = torch.load(paths.concat(opt.input, "agent.data.train"))
	y_train = torch.load(paths.concat(opt.input, "agent.labels.train"))

	X_dev = torch.load(paths.concat(opt.input, "agent.data.dev"))
	y_dev = torch.load(paths.concat(opt.input, "agent.labels.dev"))

	X_test = torch.load(paths.concat(opt.input, "agent.data.test"))
	y_test = torch.load(paths.concat(opt.input, "agent.labels.test"))

	print("Number of training examples: " .. X_train:size()[1])
	print("Number of dev examples: " .. X_dev:size()[1])
	print("Number of test examples: " .. X_test:size()[1])
	print("")

	-- Create model
	local num_feats = X_train:size(2)

	print("Building model and optimizer...")
	if opt.many_hot then
		n_input = vocab_size -- many hot
	else
		n_input = num_feats * vocab_size
	end
	
	n_hidden_1 = 1000
	n_hidden_2 = 1000
	n_output = 2

	n_epochs = 10
	n_examples = 2
	batch_size = 1024

	local optimState = {
		learningRate=0.001,
		beta1=0.9,
		beta2=0.999,
		epsilon=1e-08,
		learningRateDecay=0.0
	}

	model = nn.Sequential()
	model:add(nn.Linear(n_input, n_hidden_1))
	model:add(nn.ReLU())
	model:add(nn.Linear(n_hidden_1, n_hidden_2))
	model:add(nn.ReLU())
	model:add(nn.Linear(n_hidden_2, n_output))

	local classWeights = torch.Tensor({1 - y_train:eq(1):sum()/y_train:size(1), y_train:eq(1):sum()/y_train:size(1)})
	criterion = nn.CrossEntropyCriterion()

	if opt.use_gpu then
		model = model:cuda()
		criterion = criterion:cuda()

		-- X_train = X_train:cuda()
		-- y_train = y_train:cuda()

		-- X_dev = X_dev:cuda()
		-- y_dev = y_dev:cuda()

		-- X_test = X_test:cuda()
		-- y_test = y_test:cuda()
	end

	print("Training model for " .. n_epochs .. " epochs...")
	
	-- Create logging files
	train_log = io.open(paths.concat(opt.output, "train.log"), "w")
	dev_log = io.open(paths.concat(opt.output, "dev.log"), "w")
	test_log = io.open(paths.concat(opt.output, "test.log"), "w")

	params, gradParams = model:getParameters()

	local prepareBatch = toOneHot
	if opt.many_hot then prepareBatch = toManyHot end

	for i = 1, n_epochs do
		print("Epoch " .. i)
		local total_loss = 0
		for t = 1, X_train:size(1), batch_size do
			xlua.progress(t, X_train:size(1))

			local X_batch = X_train[{{t, math.min(t+batch_size-1, X_train:size(1))},{}}]
			local y_batch = y_train[{{t, math.min(t+batch_size-1, y_train:size(1))},{}}]

			local x = prepareBatch(X_batch, n_input, num_feats, vocab_size)
			local y = y_batch

			if opt.use_gpu then 
				x = x:cuda()
				y = y:cuda()
			end

			local feval = function(params_new)
				if params ~= params_new then
					params:copy(params_new)
				end
				gradParams:zero()

				local outputs = model:forward(x)
				local loss = criterion:forward(outputs, y)
				local dloss_doutputs = criterion:backward(outputs, y)
				model:backward(x, dloss_doutputs)

				total_loss = total_loss + loss

				loss = loss / batch_size
				gradParams = gradParams:div(batch_size)

				return loss, gradParams
			end

			optim.adam(feval, params, optimState)
		end
		xlua.progress(X_train:size(1), X_train:size(1))

		print("Loss: " .. total_loss / X_train:size(1))
		num_correct, num_correct_S, num_S, num_correct_DS, num_DS = test(X_train, y_train, model, n_input, num_feats, opt.use_gpu, opt.many_hot, vocab_size, batch_size)

		print("Train Accuracy: " .. num_correct/X_train:size(1) * 100)
		print("S Accuracy: " .. num_correct_S/num_S)
		print("DS Accuracy: " .. num_correct_DS/num_DS)
		train_log:write("Epoch " .. i .. "\n")
		train_log:write("Accuracy: " .. num_correct/X_train:size(1) .. "\n")
		train_log:write("S Accuracy: " .. num_correct_S/num_S .. "\n")
		train_log:write("DS Accuracy: " .. num_correct_DS/num_DS .. "\n")
		train_log:write("=================================\n")

		num_correct, num_correct_S, num_S, num_correct_DS, num_DS = test(X_dev, y_dev, model, n_input, num_feats, opt.use_gpu, opt.many_hot, vocab_size, batch_size)

		print("Dev Accuracy: " .. num_correct/X_dev:size(1) * 100)
		print("S Accuracy: " .. num_correct_S/num_S)
		print("DS Accuracy: " .. num_correct_DS/num_DS)
		dev_log:write("Epoch " .. i .. "\n")
		dev_log:write("Accuracy: " .. num_correct/X_dev:size(1) .. "\n")
		dev_log:write("S Accuracy: " .. num_correct_S/num_S .. "\n")
		dev_log:write("DS Accuracy: " .. num_correct_DS/num_DS .. "\n")
		dev_log:write("=================================\n")

		torch.save(paths.concat(opt.output, "epoch_".. i ..".mdl"), model)
	end
	xlua.progress(X_train:size(1), X_train:size(1))

	print("Testing...")
	num_correct, num_correct_S, num_S, num_correct_DS, num_DS = test(X_test, y_test, model, n_input, num_feats, opt.use_gpu, opt.many_hot, vocab_size, batch_size)

	print("Accuracy: " .. num_correct/X_test:size(1))
	print("S Accuracy: " .. num_correct_S/num_S)
	print("DS Accuracy: " .. num_correct_DS/num_DS)

	test_log:write("Accuracy: " .. num_correct/X_test:size(1) .. "\n")
	test_log:write("S Accuracy: " .. num_correct_S/num_S .. "\n")
	test_log:write("DS Accuracy: " .. num_correct_DS/num_DS .. "\n")

	torch.save(paths.concat(opt.output, "final.mdl"), model)

	train_log:close()
	dev_log:close()
	test_log:close()
end

function toOneHot(x, n_input, num_feats, vocab_size)
	x_onehot = torch.CudaTensor(x:size(1), n_input):zero()
	for batch_idx = 1, x:size(1) do
		for word_idx = 1, num_feats do
			x_onehot[{{batch_idx}, {(word_idx-1) * vocab_size + x[{batch_idx, word_idx}]}}] = 1
		end
	end
	return x_onehot
end

function toManyHot(x, n_input, num_feats)
	x_manyhot = torch.Tensor(x:size(1), n_input):zero()
	for batch_idx = 1, x:size(1) do
		for word_idx = 1, num_feats do
			x_manyhot[{{batch_idx}, {x[{batch_idx, word_idx}]}}] = 1
		end
	end
	return x_manyhot
end

function toOneHot_nobatch(x, num_feats)
	x_onehot = torch.Tensor(n_input):zero()
	for idx = 1, num_feats do
		x_onehot[(idx-1) * vocab_size + x[idx]] = 1
	end
	return x_onehot
end


function test(samples, labels, model, n_input, num_feats, use_gpu, many_hot, vocab_size, batch_size, verbose)
	local num_correct = 0
	local num_correct_S = 0
	local num_S = 0
	local num_correct_DS = 0
	local num_DS = 0

	local prepareBatch = toOneHot
	if many_hot then prepareBatch = toManyHot end

	print(use_gpu)
	print(batch_size)

	for t = 1, samples:size(1), batch_size do
		if verbose == true then	xlua.progress(t, samples:size(1)) end
		local x = prepareBatch(samples[{{t, math.min(t+batch_size, samples:size(1))},{}}], n_input, num_feats, vocab_size)
		local y = labels[{{t, math.min(t+batch_size, samples:size(1))},{}}]

		if use_gpu then 
			x = x:cuda()
			y = y:cuda()
		end

		local prediction = model:forward(x)
		_, idx = torch.max(prediction, 2)
		if use_gpu then idx = idx:cuda() else idx = idx:double() end

		num_correct = num_correct + y:eq(idx):sum()
		num_S = num_S + y:eq(1):sum()
		num_DS = num_DS + y:eq(2):sum()
		num_correct_S = num_correct_S + y:eq(idx):cmul(y:eq(1)):sum()
		num_correct_DS = num_correct_DS + y:eq(idx):cmul(y:eq(2)):sum()
	end
	if verbose == true then	xlua.progress(samples:size(1), samples:size(1)) end
	return num_correct, num_correct_S, num_S, num_correct_DS, num_DS
end

function test_nobatch(samples, labels, model, num_feats, use_gpu)
	local num_correct = 0
	local num_correct_S = 0
	local num_S = 0
	local num_correct_DS = 0
	local num_DS = 0
	for t = 1, samples:size(1) do
		local x = toOneHot_nobatch(samples[t], num_feats)
		local y = labels[t]

		if use_gpu then x = x:cuda() end

		local prediction = model:forward(x)
		_, idx = torch.max(prediction, 1)
		
		if y[1] == 1 then num_S = num_S + 1 else num_DS = num_DS + 1 end
		if idx[1] == y[1] then
			num_correct = num_correct + 1
			if y[1] == 1 then num_correct_S = num_correct_S + 1 else num_correct_DS = num_correct_DS + 1 end
		end
	end
	return num_correct, num_correct_S, num_S, num_correct_DS, num_DS
end

exports.train = train
exports.test = test

return exports