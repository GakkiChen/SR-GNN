import argparse
import os, sys
import time
import numpy as np
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
from networks.GRM_large import GRM
from utils import adjacency_matrix, my_collate
from my_test_ds import SRDataset
from sklearn.metrics import average_precision_score
# from tqdm import tqdm

class output_check():
	def __init__(self):
		self.print_AP = []

	def update(self, target_var, output, y_onehot):
		target_var = target_var.cpu().view(-1, 1)
		y_onehot.zero_()
		y_onehot.scatter_(1, target_var, 1)
		y_true = y_onehot.numpy().reshape(-1)
		y_scores = output.cpu().detach().numpy().reshape(-1)
		ap_val = average_precision_score(y_true, y_scores)
		self.print_AP.append(ap_val) 

	def clear(self):
		self.print_AP = []

	def compute_mean(self):
		return np.nanmean(self.print_AP)

def get_test_set(data_dir, test_list, feat_path, scale_size=256, crop_size=224):
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])
	scale_size = scale_size
	crop_size = crop_size

	test_data_transform = transforms.Compose([
			transforms.Resize((crop_size, crop_size)),
			transforms.ToTensor(),
			normalize])  # what about horizontal flip
	
	test_set = SRDataset(data_dir, test_list, feat_path, test_data_transform)
	test_loader = DataLoader(dataset=test_set, batch_size=10, shuffle=True, 
							num_workers=4, pin_memory=True, collate_fn=my_collate)
	return test_loader

def main():
	# Create dataloader
	print ('====> Creating dataloader...')
	data_dir = '/data/chen/ggnn_rl/PISC/image'
	test_list = '/data/chen/ggnn_rl/PISC/data/pisc_relationships_test.json'
	feat_path = '/data/chen/ggnn_rl/semantic-segmentation/feature_layer/avgpool_features_all.npy'
	test_loader = get_test_set(data_dir, test_list, feat_path)

	rl_dict = {}
	for epoch in range(50):
		print ('====> Loading the network...')
		loadFilename = '/data/chen/ggnn_rl/gat_codebook/model_checkpoints/larger_net/multi_head_{}.tar'.format(epoch)
		checkpoint = torch.load(loadFilename)
		model_sd = checkpoint['weights']
		model = GRM()
		# gpu_ids = [0, 1]
		# if len(gpu_ids)> 1:
		# 	model = nn.DataParallel(model, device_ids = gpu_ids)
		model.load_state_dict(model_sd)
		model.to(device)
		# cudnn.benchmark = True
		rl_map = val(test_loader, model)
		print('====> epoch: {}; rl_map: {:.4f}.'.format(epoch, rl_map))
		rl_dict[epoch] = rl_map
	sorted_x = sorted(rl_dict.items(), key=lambda kv: kv[1], reverse=True)
	print(sorted_x)

def val(test_loader, model, class_num=6):
	model.eval()
	print('====> Total data points number: ', len(test_loader.dataset))
	rl_check = output_check()
	# pbar = tqdm(total=124)
	for iterr, batch in enumerate(test_loader):		
		for graph in batch:
			batch_size = len(graph[2])
			rl_onehot = torch.FloatTensor(batch_size, class_num)
			with torch.no_grad():
				target_var = torch.LongTensor(graph[2]).squeeze().to(device) - 1
				x = graph[0].to(device)
				seg_feature = graph[3].to(device)
				adj_mat = adjacency_matrix(graph[1]).to(device)
			rl_out, _ = model(x, seg_feature, adj_mat)
			rl_check.update(target_var, rl_out, rl_onehot)
		# pbar.update(1)
	rl_map = rl_check.compute_mean()
	return rl_map


if __name__=='__main__':
	main()
