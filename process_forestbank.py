import lal
from sys import argv

def has_intolerable_errors(hv):
	err_list = lal.io.check_correctness_head_vector(hv)
	
	err1 = "self-loop"
	err2 = "out of bounds"
	err3 = "cycles"
	err4 = "not a valid non-negative"
	errs = [err1, err2, err3, err4]
	has_error = lambda s: any(map(lambda err: s.find(err) != -1, errs))
	
	return any(map(has_error, err_list))

def process_head_vector(hv):
	print("===========")
	print(hv)
	
	if has_intolerable_errors(hv):
		err_list = lal.io.check_correctness_head_vector(hv)
		for err in err_list:
			print(err)
		return True
	
	# the full directed *graph*
	dg = lal.graphs.from_head_vector_to_directed_graph(hv)
	
	# each individual tree
	connected_components = dg.get_connected_components()
	for tree in connected_components:
		rt = lal.graphs.rooted_tree(tree)
		assert(rt.is_rooted_tree())
		
	return False

filename = argv[1]
with open(filename, 'r') as f:
	
	total_sentences = 0
	num_errors = 0
	
	data = []
	for line in f:
		
		if line[0] == '#':
			# comment
			pass
			
		elif line == "\n":
			# new 'tree'
			
			hv = list(map(lambda l: int(l.split('\t')[6]), data))
			total_sentences += 1
			num_errors += process_head_vector(hv)
			
			data = []
		else:
			data.append(line[:-1])
	
	if len(data) > 0:
		hv = list(map(lambda l: int(l.split('\t')[6]), data))
		total_sentences += 1
		num_errors += process_head_vector(hv)
	
	print("Number of sentences:", total_sentences)
	print("Number of errors:", num_errors)
