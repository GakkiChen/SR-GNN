import os, sys
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import json
import random
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import pylab
from utils import adjacency_matrix
# np.set_printoptions(threshold=sys.maxsize)

class SRDataset(Dataset):
	def __init__(self, image_dir, list_path, feat_path, input_transform = None):
		super(SRDataset, self).__init__()

		self.image_dir = image_dir
		self.input_transform = input_transform
		self.names = []
		self.relationships = []
		self.max_num_node = 0
		self.seg_feat = np.load(feat_path, allow_pickle=True).item()
		relationship_list = json.load(open(list_path))

		for rl in relationship_list:
			if len(rl['relationships']) >= 3:
				self.names.append(rl['image_id'])
				self.relationships.append(rl['relationships'])

	def __getitem__(self, index):
		img_id = self.names[index]
		img_path = os.path.join(self.image_dir, img_id.zfill(5) + '.jpg')
		img = Image.open(img_path).convert('RGB') # convert gray to rgb
		all_rls = []
		target_list = []
		person_node_idx = 1
		id_to_index = {}
		rl_graph_list = []

		for pairs in self.relationships[index]:
			obj1 = pairs['object']
			obj2 = pairs['subject']
			node1 = obj1['object_id']
			node2 = obj2['object_id']
			if node1 not in id_to_index:
				id_to_index[node1] = person_node_idx
				person_node_idx += 1
			if node2 not in id_to_index:
				id_to_index[node2] = person_node_idx
				person_node_idx += 1

			src_idx = id_to_index[node1]
			tgt_idx = id_to_index[node2]
			rl_graph_list.append([src_idx, tgt_idx])

			box1 = [obj1['x'], obj1['y'], obj1['x'] + obj1['w'], obj1['y'] + obj1['h']]
			box2 = [obj2['x'], obj2['y'], obj2['x'] + obj2['w'], obj2['y'] + obj2['h']]
			obj1 = img.crop((box1[0], box1[1], box1[2], box1[3]))
			obj2 = img.crop((box2[0], box2[1], box2[2], box2[3]))

			# union
			u_x1 = min(box1[0], box2[0])
			u_y1 = min(box1[1], box2[1])
			u_x2 = max(box1[2], box2[2])
			u_y2 = max(box1[3], box2[3])
			union = img.crop((u_x1, u_y1, u_x2, u_y2))
						
			if self.input_transform:
				obj1 = self.input_transform(obj1)
				obj2 = self.input_transform(obj2)
				union = self.input_transform(union)

			target = pairs['relationship_id']
			target_list.append(int(target))
			all_rls.append(torch.stack([union, obj1, obj2]))
		
		all_rls = torch.cat(all_rls)
		seg_feat = torch.from_numpy(self.seg_feat[img_path])

		return all_rls, rl_graph_list, target_list, seg_feat, img_id

	def __len__(self):
		return len(self.names)

# data_dir = '/data/chen/ggnn_rl/PISC/image/'
# train_list = '/data/chen/ggnn_rl/PISC/data/pisc_relationships_train.json'
# feat_path = '/data/chen/ggnn_rl/semantic-segmentation/feature_layer/avgpool_features_all.npy'
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
# 									 std=[0.229, 0.224, 0.225])
# crop_size = 224

# train_data_transform = transforms.Compose([
# 		transforms.Resize((crop_size, crop_size)),
# 		transforms.ToTensor(),
# 		normalize])  # what about horizontal flip
# def my_collate(batch):
# 	print(type(batch))
# 	return [(dp[0], dp[1], torch.tensor(dp[2])) for dp in batch]

# train_set = SRDataset(data_dir, train_list, feat_path, train_data_transform)
# loader = DataLoader(train_set, batch_size=10, shuffle=True, num_workers=2, pin_memory=True, collate_fn=my_collate)
# print('dataset loaded')
# for batch in loader:
# 	for i in range(len(batch)):
# 		all_rls = batch[i][1]
# 		print(all_rls.shape)
# 		input('enter')