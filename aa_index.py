import math

def adamic_adar_index(G, src, dest):

    common_predecessors = set(G.predecessors(src)).intersection(set(G.predecessors(dest)))
    common_successors = set(G.successors(src)).intersection(set(G.successors(dest)))
    
    aa_predecessors = sum(1 / math.log(G.degree(v)) for v in common_predecessors if G.degree(v) > 1)
    aa_successors = sum(1 / math.log(G.degree(v)) for v in common_successors if G.degree(v) > 1)
    
    return aa_predecessors, aa_successors