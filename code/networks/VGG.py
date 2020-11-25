import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torchvision import models

class person_pair(nn.Module):
    def __init__(self, output_channel=2048):
        super(person_pair, self).__init__()
        self._output_channel = output_channel
        self.model_vgg = models.vgg16(pretrained=True).cuda()
        for param in self.model_vgg.features.parameters():
            param.requires_grad = False
        for param in self.model_vgg.classifier.parameters():
            param.requires_grad = True
        num_ftrs = self.model_vgg.classifier[3].out_features
        self.model_vgg.classifier[6] = nn.Linear(num_ftrs, output_channel)

    # x1 = union, x2 = object1, x3 = object2
    def forward(self, x):
        node_count = x.shape[0] 
        x = self.model_vgg(x)
        x = x.view(node_count//3, 3*self._output_channel)
        
        return x
        
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2./n))
    #             if m.bias is not None:
    #                 m.bias.data.zero_()

    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             m.weight.data.normal_(0, 0.01)
    #             m.bias.data.zero_()       

    # def vgg_conv(self, inputs):
    #     model_vgg = models.vgg16(pretrained=True).cuda()
    #     for param in model_vgg.parameters():
    #         param.requires_grad = False
    #     x = model_vgg.features(inputs)
    #     x = x.view(x.size(0), -1)
    #     return x     

    # def    