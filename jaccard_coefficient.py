def jaccard_coefficient(G, src, dest):

    src_successors = set(G.successors(src))
    dest_successors = set(G.successors(dest))
    
    union_successors = src_successors.union(dest_successors)
    intersection_successors = src_successors.intersection(dest_successors)
    jaccard_successors = len(intersection_successors) / len(union_successors) if len(union_successors) > 0 else 0
    
    src_predecessors = set(G.predecessors(src))
    dest_predecessors = set(G.predecessors(dest))
    
    union_predecessors = src_predecessors.union(dest_predecessors)
    intersection_predecessors = src_predecessors.intersection(dest_predecessors)
    jaccard_predecessors = len(intersection_predecessors) / len(union_predecessors) if len(union_predecessors) > 0 else 0
    
    return jaccard_successors, jaccard_predecessors