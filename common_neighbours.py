def compute_common_neighbors(G, src, dest):

    src_successors = set(G.successors(src))
    dest_successors = set(G.successors(dest))
    
    common_successors = src_successors.intersection(dest_successors)
    
    src_predecessors = set(G.predecessors(src))
    dest_predecessors = set(G.predecessors(dest))
    
    common_predecessors = src_predecessors.intersection(dest_predecessors)
    
    return len(common_successors), len(common_predecessors)