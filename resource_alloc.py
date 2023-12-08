def resource_allocation_index(G, src, dest):

    common_predecessors = set(G.predecessors(src)).intersection(set(G.predecessors(dest)))
    common_successors = set(G.successors(src)).intersection(set(G.successors(dest)))
    
    ra_predecessors = sum(1 / G.degree(v) for v in common_predecessors if G.degree(v) > 0)
    ra_successors = sum(1 / G.degree(v) for v in common_successors if G.degree(v) > 0)
    
    return ra_predecessors, ra_successors