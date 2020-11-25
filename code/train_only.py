import argparse
import os, sys
import time
import numpy as np
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
from networks.GRM_lite import GRM
from my_test_ds import SRDataset
from utils import adjacency_matrix, my_collate
from sklearn.metrics import average_precision_score
from tensorboardX import SummaryWriter
import copy 
from torchviz import make_dot

# writer = SummaryWriter('runs/exp-2')

print(os.environ["CUDA_VISIBLE_DEVICES"])
def get_train_set(data_dir, train_list, feat_path, scale_size=256, crop_size=224):
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])
	scale_size = scale_size
	crop_size = crop_size

	train_data_transform = transforms.Compose([
			transforms.Resize((crop_size, crop_size)),
			transforms.ToTensor(),
			normalize])  # what about horizontal flip

	train_set = SRDataset(data_dir, train_list, feat_path, train_data_transform)
	train_loader = DataLoader(train_set, batch_size=10, shuffle=True, 
							num_workers=4, pin_memory=True, collate_fn=my_collate)
	return train_loader

class output_check():
	def __init__(self):
		self.print_AP = []

	def update(self, target_var, output, y_onehot):
		y_onehot = self.covert_onehot(target_var, y_onehot)
		y_true = y_onehot.numpy().reshape(-1)
		y_scores = output.cpu().detach().numpy().reshape(-1)
		ap_val = average_precision_score(y_true, y_scores)
		self.print_AP.append(ap_val) 

	def covert_onehot(self, target_var, y_onehot):
		target_var = target_var.cpu().view(-1, 1)
		y_onehot.zero_()
		y_onehot.scatter_(1, target_var, 1)
		return y_onehot

	def clear(self):
		self.print_AP = []

	def compute_mean(self):
		return np.nanmean(self.print_AP)

def main():
	# Create dataloader
	print ('====> Creating dataloader...')
	data_dir = '/data/chen/ggnn_rl/PISC/image'
	train_list = '/data/chen/ggnn_rl/PISC/data/pisc_relationships_train.json'
	feat_path = '/data/chen/ggnn_rl/semantic-segmentation/feature_layer/avgpool_features_all.npy'
	train_loader = get_train_set(data_dir, train_list, feat_path)

	print ('====> Loading the network...')

	model = GRM()
	# gpu_ids = [0, 1]
	# if len(gpu_ids)> 1:
	# 	print("Let's use", len(gpu_ids), "GPUs!")
	# 	model = nn.DataParallel(model, device_ids = gpu_ids)
	model.to(device)
	optimizer = optim.Adam(model.parameters(), lr=0.000002)
	class_weights_relation = torch.tensor([0.02723271, 0.04418959, 0.22259935, 0.01657587, 0.6605625, 0.02883999])

	criterion_rl = nn.CrossEntropyLoss(class_weights_relation).to(device)
	# cudnn.benchmark = True

	train(train_loader, model, criterion_rl, optimizer)

def train(train_loader, model, criterion_rl, optimizer, print_every=10, epoch_num=50, class_num=6):

	model.train()

	start = time.time()
	print('====> Total data points number: ', len(train_loader.dataset))
	directory = 'model_checkpoints/modified'
	if not os.path.exists(directory):
		os.makedirs(directory)
	iterration = 0
	for epoch in range(epoch_num):
		rl_check = output_check()
		print_loss = []
		best_acc = 0
		for iterr, batch in enumerate(train_loader):
			iterration += 1 
			batch_size = len(batch)
			for graph in batch:
				batch_size = len(graph[2])
				rl_onehot = torch.FloatTensor(batch_size, class_num)
				optimizer.zero_grad()
				with torch.no_grad():
					target_var = torch.LongTensor(graph[2]).squeeze().to(device) - 1
					x = graph[0].to(device)
					# print('input shape: ', x.shape)
					seg_feature = graph[3].to(device)
					# print('segmentation shape: ', seg_feature.shape)
					adj_mat = adjacency_matrix(graph[1]).to(device)
					# print('adj matrix shape: ', adj_mat.shape)
				# ground_domain = domain_check.covert_onehot(domain_var, dm_onehot)
				rl_out = model(x, seg_feature, adj_mat)
				# p = make_dot(rl_out, params=dict(model.named_parameters()))
				# print(type(p))
				# p.format = 'svg'
				# p.render()
				# input('enter')
				rl_check.update(target_var, rl_out, rl_onehot)
				loss = criterion_rl(rl_out, target_var)
				print_loss.append(loss.item())
				loss.backward()
				optimizer.step()				
				end = time.time()

			n_iteration = len(train_loader.dataset)/10
			# Print progress
			if (iterr % print_every == 0 and iterr != 0):
				time_avg = (end - start) / print_every
				rl_map = rl_check.compute_mean()
				if rl_map > best_acc:
					best_model = copy.deepcopy(model)
					best_acc = rl_map
				mloss = np.nanmean(print_loss)
				print("===> Avg iterration time: {:.4f}; Epoch: {}; Iteration: {}; Percent complete: {:.1f}%.".format(time_avg, epoch, iterr, iterr / n_iteration * 100))
				rl_check.clear()
				print_loss = []
				start = end
				writer.add_scalar('rl_mAP', rl_map, iterration)
				writer.add_scalar('loss', mloss, iterration)

		# Save checkpoint
		print('====> Saving checkpoints')
		torch.save({
					'epoch': epoch,
					'weights': best_model.state_dict(),
					# 'opt': optimizer.state_dict(),
				}, os.path.join(directory, 'single_head_{}.tar'.format(epoch)))

if __name__=='__main__':
	# torch.multiprocessing.set_start_method('spawn')
	main()
