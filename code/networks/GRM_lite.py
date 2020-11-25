import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch.autograd import Variable
from networks.VGG import person_pair
from networks.models import GAT
import math
import numpy as np
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


class GRM(nn.Module):
	def __init__(self, 
				input_channel = 2048,
				feature_channel = 4096,
				max_rl_num = 6,
				gat_input_channel = 32,
				gat_output_channel = 64,
				gat_dropout = 0.6,
				gat_leaky_relu_alpha = 0.2,
				gat_head_count = 3):

		super(GRM, self).__init__()
		self._input_channel = input_channel
		self._inter_channel = input_channel * 3
		self._max_rl_num = max_rl_num
		self.ReLU = nn.ReLU(True)
		self.fg = person_pair(input_channel).to(device)
		self.gat = GAT(gat_input_channel, gat_output_channel, 
					max_rl_num, gat_dropout, gat_leaky_relu_alpha, gat_head_count).to(device)

		self.classifier1 = nn.Sequential(
			nn.Linear(self._inter_channel , self._inter_channel // 4),
			nn.ReLU(True),
			nn.Dropout(0.3),
			nn.Linear(self._inter_channel // 4 , self._inter_channel // 8),
			nn.ReLU(True),
		)
		concat_channel = self._inter_channel // 8 + feature_channel

		self.classifier2 = nn.Sequential(
			nn.Linear(concat_channel, concat_channel // 4),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(concat_channel // 4, concat_channel // 8),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(concat_channel // 8, gat_input_channel),
			nn.ReLU(True),
		)

		self._initialize_weights()

	def forward(self, person_input, seg_feature, adj_mat):
		contextual = self.fg(person_input)
		# print('contextual shape: ', contextual.shape)
		rl_count = contextual.shape[0]
		person_feature = self.classifier1(contextual)
		# print('person feature: ', person_feature.shape)
		combined_feature = torch.cat((person_feature, seg_feature.expand(rl_count, seg_feature.shape[0])), dim=1)
		# print('combined feature: ', combined_feature.shape)
		gat_input = self.classifier2(combined_feature)
		# print('gat input shape: ', gat_input.shape)
		rl_output = self.gat(gat_input, adj_mat)
		# print('output shape: ', rl_output.shape)
		# print('pred: ', torch.argmax(rl_output, dim=1))
		# input('enter')
		return rl_output
		
	def _initialize_weights(self):
		
		for m in self.classifier1.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight.data)
				m.bias.data.fill_(0.0001)

		for m in self.classifier2.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight.data)
				m.bias.data.fill_(0.0001)	
			