import lal

hv = [2, 0, 2, 5, 9, 5, 9, 9, 3, 9, 0] 
dg = lal.graphs.from_head_vector_to_directed_graph(hv)
# print(dg)
# when you work with the entire sentence  you work with the dg

# to get the distance between each token and the root 
n = dg.get_num_nodes() # gets the indices of each node
for u in range(0, n):
    head = dg.get_in_neighbours(u)
    print(u, ":", head, "-->", len(head))
    if len(head) == 1:
        gov = head[0]
        edge_length = abs(gov - u)
        print(u, "->", edge_length)

# D = lal.linarr.sum_edge_lengths(dg)
# print(D)

# ccs = dg.get_connected_components()

# for cc in ccs:
#     rt = lal.graphs.rooted_tree(cc)
#     print(rt)

# to get the mean hirearchical distance

ccs = dg.get_connected_components()

for cc in ccs:
    rt = lal.graphs.rooted_tree(cc)
    if rt.get_num_nodes() > 1:
        # print(lal.properties.mean_hierarchical_distance(rt))
        pass



##### to calculate omega (omega = (D_Random - D) / (D_random - D_min))#####
# in our case we have to add up the minimums of each components
# the minimum in a forest is the sum of the mins of each components 
# for the d_random there is a function in the library to do it 

# for the whole graph

D_1 = lal.linarr.sum_edge_lengths(dg)
Dr_1 = lal.properties.exp_sum_edge_lengths(dg.to_undirected()) # for the whole graph, the suffle involves all the vertices
Dmin_1 = 0

ccs = dg.get_connected_components()

for cc in ccs:
    rt = lal.graphs.rooted_tree(cc)
    Dmin_cc = lal.linarr.min_sum_edge_lengths(rt)
    # how to make minimums in projective or planar cases
    # Dmin_cc_proj = lal.linarr.min_sum_edge_lengths_projective(rt)
    # Dmin_cc_plan = lal.linarr.min_sum_edge_lengths_planar(rt)
    print(Dmin_cc)
    Dmin += Dmin_cc[0]

omega_1 = (Dr-D) / (Dr-Dmin)
print(Dmin)
print(omega_1)
print("n=", dg.get_num_nodes())
print("e=", dg.get_num_edges())

# another way of computing omega for us would be to do it per component (in the for cc loop)

# you can compute number of fluxes as well 
res = lal.linarr
