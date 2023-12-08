import random
import networkx as nx
import sys
# from tqdm import tqdm
from tqdm.notebook import tqdm
from common_neighbours import compute_common_neighbors
from jaccard_coefficient import jaccard_coefficient
from aa_index import adamic_adar_index
from resource_alloc import resource_allocation_index

def feature_extraction_pos(G, sampled_edges):
    positive_examples=[]
    for edge in tqdm(sampled_edges, desc='Processing edges', unit='edge', dynamic_ncols=True):

        src, dest = edge

        common_successors, common_predecessors = compute_common_neighbors(G, src, dest)
        jaccard_successors, jaccard_predecessors = jaccard_coefficient(G, src, dest)
        preferential_attachment = G.degree(src) * G.degree(dest)

        positive_examples.append((common_successors, common_predecessors, jaccard_successors, jaccard_predecessors, preferential_attachment, 1))

    return positive_examples

def feature_extraction_neg(G, num_edges_to_sample, all_nodes):
    negative_examples = []
    progress_bar = tqdm(total=num_edges_to_sample, desc='Generating negative examples', unit='example', dynamic_ncols=True)

    while len(negative_examples) < num_edges_to_sample:

        src, dest = random.sample(all_nodes, 2)
        
        # Check if there is no edge between the selected nodes
        if not G.has_edge(src, dest):

            common_successors, common_predecessors = compute_common_neighbors(G, src, dest)
            jaccard_successors, jaccard_predecessors = jaccard_coefficient(G, src, dest)
            preferential_attachment = G.degree(src) * G.degree(dest)
            
            negative_examples.append((common_successors, common_predecessors, jaccard_successors, jaccard_predecessors, preferential_attachment, 0))
            progress_bar.update(1)

    progress_bar.close()
    return negative_examples

def feature_extraction_neg2(G, num_edges_to_sample, all_nodes):
    negative_examples = []
    progress_bar = tqdm(total=num_edges_to_sample, desc='Generating negative examples', unit='example')
    # Precompute global features
    katz_centrality = nx.katz_centrality(G, alpha=0.005, beta=1)
    pagerank = nx.pagerank(G, alpha=0.85)
    clustering_coefficient = nx.clustering(G.to_undirected())
    shortest_path = (nx.shortest_path_length(G, source=src, target=dest, method='dijkstra') if nx.has_path(G, src, dest) else -1)
    print("Precomputed global features neg2")

    while len(negative_examples) < num_edges_to_sample:

        src, dest = random.sample(all_nodes, 2)
        
        if not G.has_edge(src, dest):
        
            # Existing Features
            common_successors, common_predecessors = compute_common_neighbors(G, src, dest)
            jaccard_successors, jaccard_predecessors = jaccard_coefficient(G, src, dest)
            preferential_attachment = G.degree(src) * G.degree(dest)
            
            # New Features
            aa_predecessors, aa_successors = adamic_adar_index(G, src, dest)
            ra_predecessors, ra_successors = resource_allocation_index(G, src, dest)
            katz_src, katz_dest = katz_centrality[src], katz_centrality[dest]
            pr_src, pr_dest = pagerank[src], pagerank[dest]
            cc_src, cc_dest = clustering_coefficient[src], clustering_coefficient[dest]
            
            negative_examples.append((
                common_successors, common_predecessors, jaccard_successors, jaccard_predecessors,
                preferential_attachment, aa_predecessors, aa_successors, ra_predecessors, ra_successors,
                katz_src, katz_dest, pr_src, pr_dest, cc_src, cc_dest, shortest_path, 1
            ))
            progress_bar.update(1)

    progress_bar.close()
    return negative_examples

def feature_extraction_pos2(G, sampled_edges):
    positive_examples = []
    # Precompute global features
    katz_centrality = nx.katz_centrality(G, alpha=0.005, beta=1)
    pagerank = nx.pagerank(G, alpha=0.85)
    clustering_coefficient = nx.clustering(G.to_undirected())
    print("Precomputed global features pos2")
    
    for edge in tqdm(sampled_edges, desc='Processing edges', unit='edge', dynamic_ncols=True):
        src, dest = edge
        
        shortest_path = (nx.shortest_path_length(G, source=src, target=dest, method='dijkstra') if nx.has_path(G, src, dest) else -1)
        
        # Existing Features
        common_successors, common_predecessors = compute_common_neighbors(G, src, dest)
        jaccard_successors, jaccard_predecessors = jaccard_coefficient(G, src, dest)
        preferential_attachment = G.degree(src) * G.degree(dest)
        
        # New Features
        aa_predecessors, aa_successors = adamic_adar_index(G, src, dest)
        ra_predecessors, ra_successors = resource_allocation_index(G, src, dest)
        katz_src, katz_dest = katz_centrality[src], katz_centrality[dest]
        pr_src, pr_dest = pagerank[src], pagerank[dest]
        cc_src, cc_dest = clustering_coefficient[src], clustering_coefficient[dest]
        
        positive_examples.append((
            common_successors, common_predecessors, jaccard_successors, jaccard_predecessors,
            preferential_attachment, aa_predecessors, aa_successors, ra_predecessors, ra_successors,
            katz_src, katz_dest, pr_src, pr_dest, cc_src, cc_dest, shortest_path, 1
        ))
    
    return positive_examples

def feature_extraction_pos3(G, sampled_edges):
    positive_examples = []
    
    # Precompute existing global features
    katz_centrality = nx.katz_centrality(G, alpha=0.005, beta=1)
    pagerank = nx.pagerank(G, alpha=0.85)
    clustering_coefficient = nx.clustering(G.to_undirected())
    
    # Precompute new global features
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, k=10)
    average_neighbor_degree = nx.average_neighbor_degree(G)
    harmonic_centrality = nx.harmonic_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=100)
    
    print("Precomputed global features pos3")
    
    for edge in tqdm(sampled_edges, desc='Processing edges', unit='edge', dynamic_ncols=True):
        src, dest = edge
        
        # Existing Features
        common_successors, common_predecessors = compute_common_neighbors(G, src, dest)
        jaccard_successors, jaccard_predecessors = jaccard_coefficient(G, src, dest)
        preferential_attachment = G.degree(src) * G.degree(dest)
        
        # New Features
        aa_predecessors, aa_successors = adamic_adar_index(G, src, dest)
        ra_predecessors, ra_successors = resource_allocation_index(G, src, dest)
        katz_src, katz_dest = katz_centrality[src], katz_centrality[dest]
        pr_src, pr_dest = pagerank[src], pagerank[dest]
        cc_src, cc_dest = clustering_coefficient[src], clustering_coefficient[dest]
        
        # Additional New Features
        closeness_src, closeness_dest = closeness_centrality[src], closeness_centrality[dest]
        betweenness_src, betweenness_dest = betweenness_centrality[src], betweenness_centrality[dest]
        avg_neighbor_degree_src, avg_neighbor_degree_dest = average_neighbor_degree[src], average_neighbor_degree[dest]
        harmonic_src, harmonic_dest = harmonic_centrality[src], harmonic_centrality[dest]
        eigenvector_src, eigenvector_dest = eigenvector_centrality[src], eigenvector_centrality[dest]
        
        positive_examples.append((
            common_successors, common_predecessors, jaccard_successors, jaccard_predecessors,
            preferential_attachment, aa_predecessors, aa_successors, ra_predecessors, ra_successors,
            katz_src, katz_dest, pr_src, pr_dest, cc_src, cc_dest, closeness_src, closeness_dest,
            betweenness_src, betweenness_dest, avg_neighbor_degree_src, avg_neighbor_degree_dest,
            harmonic_src, harmonic_dest, eigenvector_src, eigenvector_dest, 1
        ))
    
    return positive_examples

def feature_extraction_neg3(G, num_edges_to_sample, all_nodes):
    negative_examples = []
    
    # Precompute existing global features
    katz_centrality = nx.katz_centrality(G, alpha=0.005, beta=1)
    pagerank = nx.pagerank(G, alpha=0.85)
    clustering_coefficient = nx.clustering(G.to_undirected())
    
    # Precompute new global features
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, k=10)
    average_neighbor_degree = nx.average_neighbor_degree(G)
    harmonic_centrality = nx.harmonic_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=100)
    
    print("Precomputed global features neg3")
    progress_bar = tqdm(total=num_edges_to_sample, desc='Generating negative examples', unit='example')

    while len(negative_examples) < num_edges_to_sample:

        src, dest = random.sample(all_nodes, 2)
        
        if not G.has_edge(src, dest):
        
            # Existing Features
            common_successors, common_predecessors = compute_common_neighbors(G, src, dest)
            jaccard_successors, jaccard_predecessors = jaccard_coefficient(G, src, dest)
            preferential_attachment = G.degree(src) * G.degree(dest)
            
            # New Features
            aa_predecessors, aa_successors = adamic_adar_index(G, src, dest)
            ra_predecessors, ra_successors = resource_allocation_index(G, src, dest)
            katz_src, katz_dest = katz_centrality[src], katz_centrality[dest]
            pr_src, pr_dest = pagerank[src], pagerank[dest]
            cc_src, cc_dest = clustering_coefficient[src], clustering_coefficient[dest]
            
            # Additional New Features
            closeness_src, closeness_dest = closeness_centrality[src], closeness_centrality[dest]
            betweenness_src, betweenness_dest = betweenness_centrality[src], betweenness_centrality[dest]
            avg_neighbor_degree_src, avg_neighbor_degree_dest = average_neighbor_degree[src], average_neighbor_degree[dest]
            harmonic_src, harmonic_dest = harmonic_centrality[src], harmonic_centrality[dest]
            eigenvector_src, eigenvector_dest = eigenvector_centrality[src], eigenvector_centrality[dest]
            
            negative_examples.append((
                common_successors, common_predecessors, jaccard_successors, jaccard_predecessors,
                preferential_attachment, aa_predecessors, aa_successors, ra_predecessors, ra_successors,
                katz_src, katz_dest, pr_src, pr_dest, cc_src, cc_dest, closeness_src, closeness_dest,
                betweenness_src, betweenness_dest, avg_neighbor_degree_src, avg_neighbor_degree_dest,
                harmonic_src, harmonic_dest, eigenvector_src, eigenvector_dest, 0
            ))
            progress_bar.update(1)

    progress_bar.close()
    return negative_examples

def feature_extraction_neg4(G, num_edges_to_sample, all_nodes):
    negative_examples = []
    progress_bar = tqdm(total=num_edges_to_sample, desc='Generating negative examples', unit='example', dynamic_ncols=True)

    while len(negative_examples) < num_edges_to_sample:

        src, dest = random.sample(all_nodes, 2)
        
        if not G.has_edge(src, dest):
        
            # Existing Features
            common_successors, common_predecessors = compute_common_neighbors(G, src, dest)
            jaccard_successors, jaccard_predecessors = jaccard_coefficient(G, src, dest)
            preferential_attachment = G.degree(src) * G.degree(dest)
            
            negative_examples.append((
                common_successors, common_predecessors, jaccard_successors, jaccard_predecessors,
                preferential_attachment, 0
            ))
            progress_bar.update(1)

    progress_bar.close()
    return negative_examples

def feature_extraction_pos4(G, sampled_edges):
    positive_examples = []
    
    for edge in tqdm(sampled_edges, desc='Processing edges', unit='edge', dynamic_ncols=True):
        src, dest = edge
        
        # Existing Features
        common_successors, common_predecessors = compute_common_neighbors(G, src, dest)
        jaccard_successors, jaccard_predecessors = jaccard_coefficient(G, src, dest)
        preferential_attachment = G.degree(src) * G.degree(dest)
        
        # Append features and label to the list
        positive_examples.append((
            common_successors, common_predecessors, jaccard_successors, jaccard_predecessors,
            preferential_attachment, 1
        ))
    
    return positive_examples
