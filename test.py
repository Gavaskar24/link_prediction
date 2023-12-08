from common_neighbours import compute_common_neighbors
from jaccard_coefficient import jaccard_coefficient
from aa_index import adamic_adar_index
from resource_alloc import resource_allocation_index
import pandas as pd
import networkx as nx

def test_model(G, test_df, scaler, columns, ensemble_model):
    test_data = []

    for index, row in test_df.iterrows():

        src, dest = row['From'], row['To']

        common_successors, common_predecessors = compute_common_neighbors(G, src, dest)
        jaccard_successors, jaccard_predecessors = jaccard_coefficient(G, src, dest)
        preferential_attachment = G.degree(src) * G.degree(dest)

        test_data.append((common_successors, common_predecessors, jaccard_successors, jaccard_predecessors, preferential_attachment))

    test_features_df = pd.DataFrame(test_data, columns=columns)

    test_features_scaled = scaler.transform(test_features_df)
    
    test_probs = ensemble_model.predict_proba(test_features_scaled)[:, 1]
    
    return test_probs, test_features_scaled

# def test_model(G, test_df, scaler, columns, ensemble_model):
#     test_data = []

#     # Precompute existing global features
#     katz_centrality = nx.katz_centrality(G, alpha=0.005, beta=1)
#     pagerank = nx.pagerank(G, alpha=0.85)
#     clustering_coefficient = nx.clustering(G.to_undirected())
    
#     # Precompute new global features
#     closeness_centrality = nx.closeness_centrality(G)
#     betweenness_centrality = nx.betweenness_centrality(G, k=10)  # using approximation with k samples
#     average_neighbor_degree = nx.average_neighbor_degree(G)
#     harmonic_centrality = nx.harmonic_centrality(G)
#     eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=100)

#     print("Precomputed global features test model")

#     for index, row in test_df.iterrows():

#         src, dest = row['From'], row['To']

#         # Existing Features
#         common_successors, common_predecessors = compute_common_neighbors(G, src, dest)
#         jaccard_successors, jaccard_predecessors = jaccard_coefficient(G, src, dest)
#         preferential_attachment = G.degree(src) * G.degree(dest)
        
#         # New Features
#         aa_predecessors, aa_successors = adamic_adar_index(G, src, dest)
#         ra_predecessors, ra_successors = resource_allocation_index(G, src, dest)
#         katz_src, katz_dest = katz_centrality[src], katz_centrality[dest]
#         pr_src, pr_dest = pagerank[src], pagerank[dest]
#         cc_src, cc_dest = clustering_coefficient[src], clustering_coefficient[dest]
        
#         # Additional New Features
#         closeness_src, closeness_dest = closeness_centrality[src], closeness_centrality[dest]
#         betweenness_src, betweenness_dest = betweenness_centrality[src], betweenness_centrality[dest]
#         avg_neighbor_degree_src, avg_neighbor_degree_dest = average_neighbor_degree[src], average_neighbor_degree[dest]
#         harmonic_src, harmonic_dest = harmonic_centrality[src], harmonic_centrality[dest]
#         eigenvector_src, eigenvector_dest = eigenvector_centrality[src], eigenvector_centrality[dest]

#         test_data.append((
#             common_successors, common_predecessors, jaccard_successors, jaccard_predecessors,
#             preferential_attachment, aa_predecessors, aa_successors, ra_predecessors, ra_successors,
#             katz_src, katz_dest, pr_src, pr_dest, cc_src, cc_dest, closeness_src, closeness_dest,
#             betweenness_src, betweenness_dest, avg_neighbor_degree_src, avg_neighbor_degree_dest,
#             harmonic_src, harmonic_dest, eigenvector_src, eigenvector_dest
#         ))

#     test_features_df = pd.DataFrame(test_data, columns=columns)

#     test_features_scaled = scaler.transform(test_features_df)
    
#     test_probs = ensemble_model.predict_proba(test_features_scaled)[:, 1]
    
#     return test_probs, test_features_scaled

import xgboost as xgb

def test_xgboost_model(G, test_df, scaler, columns, bst_model):
    test_data = []

    for index, row in test_df.iterrows():

        src, dest = row['From'], row['To']

        common_successors, common_predecessors = compute_common_neighbors(G, src, dest)
        jaccard_successors, jaccard_predecessors = jaccard_coefficient(G, src, dest)
        preferential_attachment = G.degree(src) * G.degree(dest)
        
        test_data.append((
            common_successors, common_predecessors, jaccard_successors, jaccard_predecessors,
            preferential_attachment
        ))

    test_features_df = pd.DataFrame(test_data, columns=columns)
    test_features_scaled = scaler.transform(test_features_df)
    
    dtest = xgb.DMatrix(test_features_scaled)
    
    test_probs = bst_model.predict(dtest)
    
    return test_probs, test_features_scaled