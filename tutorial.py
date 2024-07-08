import lal

hv = [3, 3, 0, 5, 3, 9, 9, 9, 5, 9, 3, 13, 0, 0, 0, 15, 18, 15, 18, 18, 20, 0, 0, 23]

dg = lal.graphs.from_head_vector_to_directed_graph(hv)
n = dg.get_num_nodes()
for u in range(0, n):
	head = dg.get_in_neighbors(u)
	if len(head) == 1:
		gov = head[0]
		edge_length = abs(gov - u)

#Dr_1 = the expected value of D we would find in a uniformly random permutation of the words in the sentence
Dr_1 = lal.properties.exp_sum_edge_lengths(dg.to_undirected()) 
Dr_2 = 0
D = lal.linarr.sum_edge_lengths(dg)
Dmin = 0

# ccs[0], ccs[1]
ccs = dg.get_connected_components()
print(len(ccs))
for cc in ccs:
	print(type(cc))
	rt = lal.graphs.rooted_tree(cc)
	
	Dmin_cc = lal.linarr.min_sum_edge_lengths(rt) # “If we could arrange the words of a sentence in any way possible, what would be a way to do sosuch that it produces the minimum value of D for said sentence?”
	Dmin += Dmin_cc[0]
	
	Dr_cc = lal.properties.exp_sum_edge_lengths(rt)
	Dr_2 += Dr_cc

# Omega_1 = (Dr_1 - D)/(Dr_1 - Dmin)
# print(Omega_1)

# Omega_2 = (Dr_2 - D)/(Dr_2 - Dmin)
# print(Omega_2)

# print("Dr_1=", Dr_1)
# print("Dr_2=", Dr_2)

