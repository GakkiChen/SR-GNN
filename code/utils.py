import torch

def adjacency_matrix(rl_graph_list):
	size = len(rl_graph_list)
	adj = torch.zeros((size, size))
	for i, node in enumerate(rl_graph_list):
		for j, other_node in enumerate(rl_graph_list):
			if any(list(e in node for e in other_node)):
				adj[i, j] = 1
	return adj 

def my_collate(batch):
	return [(dp[0], dp[1], torch.tensor(dp[2]), dp[3], dp[4]) for dp in batch]

def vis_collate(batch):
	return [(dp[0], dp[1], dp[2], dp[3], dp[4], dp[5]) for dp in batch]