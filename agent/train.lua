-- load libraries
require 'nn'
require 'torch'
require 'xlua'
require 'optim'

local agent = require 'classifier'

function main()
	-- Parse arguments
	local cmd = torch.CmdLine()
	cmd:option('-input', '', [[Path to processed data directory]])
	cmd:option('-output', './', [[Path to Output folder to save models]])
	cmd:option('-use_gpu', true, [[Whether the computation should be done on a GPU]])
	cmd:option('-many_hot', true, [[Whether to use Many hot representation]])
	local opt = cmd:parse(arg)

	agent.train(opt)
end

main()
