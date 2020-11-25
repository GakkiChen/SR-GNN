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
from utils import adjacency_matrix, vis_collate
from vis_ds import SRDataset
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

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
							num_workers=4, pin_memory=True, collate_fn=vis_collate)
	return test_loader

def draw_bbox(node_dict, img_id):
	image_dir = '/data/chen/ggnn_rl/PISC/image'
	img_path = os.path.join(image_dir, img_id.zfill(5) + '.jpg')
	source_img = Image.open(img_path).convert('RGB') # convert gray to rgb
	draw = ImageDraw.Draw(source_img)
	for k, v in node_dict.items():
		draw.rectangle(v, outline='green', width=3)
		draw.text((v[0], v[1]), str(k), fill='red')
	return source_img

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

	model.load_state_dict(model_sd)
	model.to(device)

	for iterr, batch in enumerate(test_loader):
		for graph in batch:
			val(graph, model)
			input('enter')

def val(graph, model):
	model.eval()
	
	with torch.no_grad():
		target_var = graph[2]
		x = graph[0].to(device)
		seg_feature = graph[3].to(device)
		adj_mat = adjacency_matrix(graph[1]).to(device)
	img_id = graph[4]
	node_dict = graph[5]
	rl_out, _ = model(x, seg_feature, adj_mat)
	_, predicted = torch.max(rl_out.data, 1)
	y_pred = list(CATEGORIES[pred.item()] for pred in predicted)	
	y_true = list(CATEGORIES[tar[2] - 1] for tar in target_var)
	pair = list((tar[0], tar[1]) for tar in target_var)
	log = list('Pair {} and {} Pred: {} True: {}'.format(pair[i][0], pair[i][1], y_pred[i], y_true[i]) for i in range(len(pair)))
	
	# print('====> y_true: ', y_true)
	# print('====> y_pred: ', y_pred)
	for l in log:
		print(l)
	source_img = draw_bbox(node_dict, img_id)
	source_img.show()

if __name__=='__main__':
	main()
