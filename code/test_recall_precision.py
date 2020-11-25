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
from networks.GRM_large import GRM
from utils import adjacency_matrix, my_collate
from my_test_ds import SRDataset
from sklearn.metrics import precision_recall_fscore_support


CATEGORIES = ['friends', 'family', 'couple', 'professional', 'commercial', 'no-relation']

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

	print ('====> Loading the network...')
	loadFilename = '/data/chen/ggnn_rl/gat_codebook/model_checkpoints/larger_net/multi_head_{}.tar'.format(10)
	checkpoint = torch.load(loadFilename)
	model_sd = checkpoint['weights']
	model = GRM()
	# gpu_ids = [0, 1]
	# if len(gpu_ids)> 1:
	# 	model = nn.DataParallel(model, device_ids = gpu_ids)
	model.load_state_dict(model_sd)
	model.to(device)
	# cudnn.benchmark = True
	y_pred, y_true, save_dict = val(test_loader, model)
	np.save('save_dict.npy', save_dict)
	pred_dict = {}
	for pred in y_pred:
		if pred in pred_dict:
			pred_dict[pred] += 1
		else:
			pred_dict[pred] = 1
	print(pred_dict)
	out = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4, 5])
	print(out)

def val(test_loader, model, print_every=100, batch_size=32, class_num=6):
	y_pred = []
	y_true = []
	save_dict = {}
	model.eval()
	print('====> Total data points number: ', len(test_loader.dataset))

	for iterr, batch in enumerate(test_loader):		
		for graph in batch:
			with torch.no_grad():
				target_var = torch.LongTensor(graph[2]).squeeze().to(device) - 1
				x = graph[0].to(device)
				seg_feature = graph[3].to(device)
				adj_mat = adjacency_matrix(graph[1]).to(device)
			img_id = graph[4]
			rl_out, _ = model(x, seg_feature, adj_mat)
			_, predicted = torch.max(rl_out.data, 1)
			y_pred += list(pred.item() for pred in predicted)	
			y_true += list(tar.item() for tar in target_var)
			save_dict[img_id] = {'pred': list(pred.item() for pred in predicted),
								 'true': list(tar.item() for tar in target_var)}
		# img_list += list(img.item() for img in img_id)

	y_pred = np.asarray(y_pred)
	y_true = np.asarray(y_true)
	# img_list = np.asarray(img_list)
	return y_pred, y_true, save_dict

if __name__=='__main__':
	main()
