import random
import networkx as nx

def random_walk_sampling(G, start_node=None, num_edges=25000):

    max_steps = num_edges * 10

    if start_node is None:
        start_node = random.choice(list(G.nodes()))
    
    sampled_edges = set()
    current_node = start_node
    steps_taken = 0
    
    while len(sampled_edges) < num_edges:
        neighbors = list(G.neighbors(current_node))
        if not neighbors:  # If the current node has no neighbors, choose a new start node
            current_node = random.choice(list(G.nodes()))
            continue
        
        next_node = random.choice(neighbors)
        sampled_edges.add((current_node, next_node))
        
        if len(sampled_edges) >= num_edges:
            break
        
        current_node = next_node
        steps_taken += 1
        
        if steps_taken >= max_steps and len(sampled_edges) < num_edges:
            # If maximum steps are reached but the required sample size is not met,
            # choose a new start node and reset the steps_taken
            current_node = random.choice(list(G.nodes()))
            steps_taken = 0
    
    sampled_subgraph = G.edge_subgraph(sampled_edges)
    return sampled_subgraph

def uniform_random_node_sampling(G, max_edges):
    sampled_edges = set()
    all_nodes = list(G.nodes())
    
    while len(sampled_edges) < max_edges:

        node = random.choice(all_nodes)
        
        # Add all edges of the selected node to the subgraph
        neighbors = list(G.successors(node)) + list(G.predecessors(node))
        for neighbor in neighbors:

            if len(sampled_edges) >= max_edges:
                break
            edge = (node, neighbor) if G.has_edge(node, neighbor) else (neighbor, node)
            sampled_edges.add(edge)
    
    sampled_graph = G.edge_subgraph(sampled_edges)
    return sampled_graph

import pandas as pd
import collections

# def preferential_uniform_random_node_sampling(G, max_edges, test_file_path):

#     test_df = pd.read_csv(test_file_path)

#     test_df.columns = test_df.columns.str.replace(' ', '')

#     test_df['From'] = test_df['From'].astype(int)
#     test_df['To'] = test_df['To'].astype(int)

#     test_nodes = set(test_df['From']).union(set(test_df['To']))
    
#     sampled_edges = set()
#     bfs_deque = collections.deque()
    
#     # First, add all edges connected to the nodes in test_nodes to the sampled_edges set
#     for node in test_nodes:
#         if node in G:
#             neighbors = list(G.successors(node)) + list(G.predecessors(node))
#             for neighbor in neighbors:
#                 edge = (node, neighbor) if G.has_edge(node, neighbor) else (neighbor, node)
#                 sampled_edges.add(edge)
#                 bfs_deque.append(neighbor)  # Add the neighbor to the BFS deque
#                 if len(sampled_edges) >= max_edges:
#                     return G.edge_subgraph(sampled_edges)
    
#     while len(sampled_edges) < max_edges and bfs_deque:

#         node = bfs_deque.popleft()
        
#         # Add all edges of the selected node to the subgraph
#         neighbors = list(G.successors(node)) + list(G.predecessors(node))
#         for neighbor in neighbors:

#             if len(sampled_edges) >= max_edges:
#                 break
#             edge = (node, neighbor) if G.has_edge(node, neighbor) else (neighbor, node)
#             if edge not in sampled_edges:
#                 sampled_edges.add(edge)
#                 bfs_deque.append(neighbor)  # Add the neighbor to the BFS deque
    
#     # If max_edges is not reached, perform uniform random sampling for the remaining edges
#     all_nodes = list(G.nodes())
#     while len(sampled_edges) < max_edges:
#         node = random.choice(all_nodes)
#         neighbors = list(G.successors(node)) + list(G.predecessors(node))
#         for neighbor in neighbors:
#             if len(sampled_edges) >= max_edges:
#                 break
#             edge = (node, neighbor) if G.has_edge(node, neighbor) else (neighbor, node)
#             sampled_edges.add(edge)
    
#     sampled_graph = G.edge_subgraph(sampled_edges)
    
#     return sampled_graph


# def preferential_uniform_random_node_sampling(G, max_edges, test_file_path):

#     test_df = pd.read_csv(test_file_path)
#     test_df['From'] = test_df['From'].astype(int)
#     test_df['To'] = test_df['To'].astype(int)
    
#     test_nodes = set(test_df['From']).union(set(test_df['To']))
    
#     sampled_edges = set()
#     bfs_deque = collections.deque()
    
#     # First, add all edges connected to the nodes in test_nodes to the sampled_edges set
#     for node in test_nodes:
#         if node in G:  # Check if the node exists in the graph
#             neighbors = list(G.successors(node)) + list(G.predecessors(node))
#             for neighbor in neighbors:
#                 edge = (node, neighbor) if G.has_edge(node, neighbor) else (neighbor, node)
#                 sampled_edges.add(edge)
#                 bfs_deque.append(neighbor)  # Add the neighbor to the BFS deque
    
#     while len(sampled_edges) < max_edges and bfs_deque:

#         node = bfs_deque.popleft()
        
#         # Add all edges of the selected node to the subgraph
#         neighbors = list(G.successors(node)) + list(G.predecessors(node))
#         for neighbor in neighbors:

#             if len(sampled_edges) >= max_edges:
#                 break
#             edge = (node, neighbor) if G.has_edge(node, neighbor) else (neighbor, node)
#             if edge not in sampled_edges:
#                 sampled_edges.add(edge)
#                 bfs_deque.append(neighbor)  # Add the neighbor to the BFS deque
    
#     # If max_edges is not reached, perform uniform random sampling for the remaining edges
#     all_nodes = list(G.nodes())
#     while len(sampled_edges) < max_edges:
#         node = random.choice(all_nodes)
#         neighbors = list(G.successors(node)) + list(G.predecessors(node))
#         for neighbor in neighbors:
#             if len(sampled_edges) >= max_edges:
#                 break
#             edge = (node, neighbor) if G.has_edge(node, neighbor) else (neighbor, node)
#             sampled_edges.add(edge)
    
#     sampled_graph = G.edge_subgraph(sampled_edges)
    
#     return sampled_graph


def preferential_uniform_random_node_sampling(G, max_edges, test_file_path):

    test_df = pd.read_csv(test_file_path)
    test_df['From'] = test_df['From'].astype(int)
    test_df['To'] = test_df['To'].astype(int)
    
    test_nodes = set(test_df['From']).union(set(test_df['To']))
    
    sampled_edges = set()
    bfs_deque = collections.deque()
    
    # First, add all edges connected to the nodes in test_nodes to the sampled_edges set
    for node in test_nodes:
        if node in G:
            neighbors = list(G.successors(node)) + list(G.predecessors(node))
            for neighbor in neighbors:
                if len(sampled_edges) >= max_edges:
                    break
                edge = (node, neighbor) if G.has_edge(node, neighbor) else (neighbor, node)
                sampled_edges.add(edge)
                bfs_deque.append(neighbor)  # Add the neighbor to the BFS deque
    
    while len(sampled_edges) < max_edges and bfs_deque:

        node = bfs_deque.popleft()
        
        # Add all edges of the selected node to the subgraph
        neighbors = list(G.successors(node)) + list(G.predecessors(node))
        for neighbor in neighbors:

            if len(sampled_edges) >= max_edges:
                break
            edge = (node, neighbor) if G.has_edge(node, neighbor) else (neighbor, node)
            if edge not in sampled_edges:
                sampled_edges.add(edge)
                bfs_deque.append(neighbor)  # Add the neighbor to the BFS deque
    
    sampled_graph = G.edge_subgraph(sampled_edges)
    
    return sampled_graph


# def uniform_random_node_sampling(G, max_edges, test_file_path):

#     test_df = pd.read_csv(test_file_path)
#     test_df['From'] = test_df['From'].astype(int)
#     test_df['To'] = test_df['To'].astype(int)
    
#     test_nodes = set(test_df['From']).union(set(test_df['To']))
    
#     sampled_edges = set()
    
#     # For each node in test_nodes, add one edge preferably connecting another node in the test set
#     for node in test_nodes:
#         if node in G:
#             neighbors = set(G.neighbors(node)).union(set(G.predecessors(node)))
#             test_neighbors = neighbors.intersection(test_nodes)
            
#             if test_neighbors:
#                 selected_neighbor = random.choice(list(test_neighbors))
#             elif neighbors:
#                 selected_neighbor = random.choice(list(neighbors))
#             else:
#                 continue
            
#             edge = (node, selected_neighbor) if G.has_edge(node, selected_neighbor) else (selected_neighbor, node)
#             sampled_edges.add(edge)
            
#             if len(sampled_edges) >= max_edges:
#                 break
    
#     # If the total number of edges is still below max_edges, perform uniform random sampling for the remaining edges amongst the test set nodes
#     test_edges = [(u, v) for u in test_nodes for v in (set(G.neighbors(u)).union(set(G.predecessors(u))) & test_nodes) if u != v]
#     while len(sampled_edges) < max_edges and test_edges:
#         edge = random.choice(test_edges)
#         sampled_edges.add(edge)
#         test_edges.remove(edge)
    
#     sampled_graph = G.edge_subgraph(sampled_edges)
    
#     return sampled_graph

# def uniform_random_node_sampling(G, max_edges, test_file_path):

#     test_df = pd.read_csv(test_file_path)
#     test_df['From'] = test_df['From'].astype(int)
#     test_df['To'] = test_df['To'].astype(int)
    
#     test_nodes = set(test_df['From']).union(set(test_df['To']))
    
#     sampled_edges = set()
    
#     # For each node in test_nodes, add one edge preferably connecting another node in the test set
#     for node in test_nodes:
#         if node in G:
#             neighbors = set(G.neighbors(node)).union(set(G.predecessors(node)))
#             test_neighbors = neighbors.intersection(test_nodes)
            
#             if test_neighbors:
#                 selected_neighbor = random.choice(list(test_neighbors))
#             elif neighbors:
#                 selected_neighbor = random.choice(list(neighbors))
#             else:
#                 continue
            
#             edge = (node, selected_neighbor) if G.has_edge(node, selected_neighbor) else (selected_neighbor, node)
#             sampled_edges.add(edge)
            
#             if len(sampled_edges) >= max_edges:
#                 break
    
#     # Perform uniform random sampling for the remaining edges amongst the test set nodes
#     test_edges = [(u, v) for u in test_nodes for v in (set(G.neighbors(u)).union(set(G.predecessors(u))) & test_nodes) if u != v]
#     while len(sampled_edges) < max_edges and test_edges:
#         edge = random.choice(test_edges)
#         sampled_edges.add(edge)
#         test_edges.remove(edge)
    
#     # If the total number of edges is still below max_edges, perform uniform random sampling for the remaining edges amongst all nodes
#     all_edges = [(u, v) for u in G.nodes() for v in (set(G.neighbors(u)).union(set(G.predecessors(u)))) if u != v and (u, v) not in sampled_edges and (v, u) not in sampled_edges]
#     while len(sampled_edges) < max_edges and all_edges:
#         edge = random.choice(all_edges)
#         sampled_edges.add(edge)
#         all_edges.remove(edge)
    
#     sampled_graph = G.edge_subgraph(sampled_edges)
    
#     return sampled_graph


# def uniform_random_node_sampling(G, max_edges, test_file_path):

#     test_df = pd.read_csv(test_file_path)
#     test_df['From'] = test_df['From'].astype(int)
#     test_df['To'] = test_df['To'].astype(int)
    
#     test_nodes = set(test_df['From']).union(set(test_df['To']))
    
#     sampled_edges = set()
    
#     # For each node in test_nodes, add one edge preferably connecting another node in the test set
#     for node in test_nodes:
#         if node in G:
#             neighbors = set(G.neighbors(node)).union(set(G.predecessors(node)))
#             test_neighbors = neighbors.intersection(test_nodes)
            
#             if test_neighbors:
#                 selected_neighbor = random.choice(list(test_neighbors))
#             elif neighbors:
#                 selected_neighbor = random.choice(list(neighbors))
#             else:
#                 continue
            
#             edge = (node, selected_neighbor) if G.has_edge(node, selected_neighbor) else (selected_neighbor, node)
#             sampled_edges.add(edge)
            
#             if len(sampled_edges) >= max_edges:
#                 break
    
#     # Perform uniform random sampling for the remaining edges amongst the test set nodes
#     test_edges = [(u, v) for u in test_nodes for v in (set(G.neighbors(u)).union(set(G.predecessors(u))) & test_nodes) if u != v]
#     while len(sampled_edges) < max_edges and test_edges:
#         edge = random.choice(test_edges)
#         sampled_edges.add(edge)
#         test_edges.remove(edge)
    
#     # If the total number of edges is still below max_edges, perform uniform random sampling for the remaining edges amongst all nodes
#     all_edges = set(G.edges()).difference(sampled_edges)
#     while len(sampled_edges) < max_edges and all_edges:
#         edge = random.choice(list(all_edges))
#         sampled_edges.add(edge)
#         all_edges.remove(edge)
    
#     sampled_graph = G.edge_subgraph(sampled_edges)
    
#     return sampled_graph