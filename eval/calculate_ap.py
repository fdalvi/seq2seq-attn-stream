import argparse
import math

def main():
	parser = argparse.ArgumentParser(description='Calculate AP.')
	parser.add_argument('-s', '--source', required=True, help='source file')
	parser.add_argument('-t', '--target', required=True, help='target file')
	parser.add_argument('-p', '--policy', required=True, help='policy')

	args = parser.parse_args()

	if len(args.policy.split(",")) != 3 and args.policy != 'direct' and 'chunk' not in args.policy:
		raise RuntimeError("Policy should be of the form 6,1,1 or 5,2,2 or 'direct' or 'chunk6' etc.")

	source_tokens = []
	target_tokens = []
	with open(args.source) as fp:
		for line in fp:
			source_tokens.append(line.strip().split(' '))

	with open(args.target) as fp:
		for line in fp:
			target_tokens.append(line.strip().split(' '))

	if len(args.policy.split(",")) == 3:
		start = int(args.policy.split(",")[0])
		read = int(args.policy.split(",")[1])
		write = int(args.policy.split(",")[2])

	if 'chunk' in args.policy:
		input_chunk = int(args.policy[5:])

		# modify target_tokens
		combined_target_tokens = [[] for _ in range(len(source_tokens))]
		total_processed_chunks = 0
		source_idx = 0
		X = source_tokens[source_idx]
		curr_chunks = math.ceil(1.0*len(X)/input_chunk)
		for idx, Y in enumerate(target_tokens):
			if idx < total_processed_chunks + curr_chunks:
				combined_target_tokens[source_idx].append(target_tokens[idx])
			
			if idx+1 == total_processed_chunks + curr_chunks:
				source_idx += 1
				total_processed_chunks += curr_chunks
				if total_processed_chunks != len(target_tokens):
					X = source_tokens[source_idx]
					curr_chunks = math.ceil(1.0*len(X)/input_chunk)
		target_tokens = combined_target_tokens

	assert(len(source_tokens) == len(target_tokens))

	aps = 0.0
	for idx, Y in enumerate(target_tokens):
		X = source_tokens[idx]

		if args.policy == 'direct':
			ap = 1.0 * sum([int(y) for y in Y]) / (len(X) * len(Y))
			aps += ap
		elif 'chunk' in args.policy:
			ap = 0.0
			true_target_length = 0
			for idx,chunks in enumerate(Y):
				ap += (idx+1) * len(chunks) * input_chunk
				true_target_length += len(chunks)
			ap = ap / (len(X) * true_target_length)
			aps += ap
		else:
			ap = 0.0
			for w in range(len(Y)):
				additional_delay = int(w/read)*write
				ap += min(start + additional_delay, len(X))
			ap = ap / (len(X) * len(Y))
			aps += ap
	print (aps / len(target_tokens))

if __name__ == '__main__':
	main()