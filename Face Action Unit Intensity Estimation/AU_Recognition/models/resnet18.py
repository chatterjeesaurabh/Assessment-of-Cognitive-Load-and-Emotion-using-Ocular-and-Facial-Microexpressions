import torch.nn as nn
import torch
import torchvision
import pdb

class ResNet18(nn.Module):
	def __init__(self, opts):
		super(ResNet18, self).__init__()
		self.fm_distillation = opts.fm_distillation
		self.dropout = opts.dropout
		resnet18 = torchvision.models.resnet18(pretrained=True)		# ResNet-18 load (TorchVision Pre-Trained)
		resnet18_layers = list(resnet18.children())[:-1]
		self.encoder = nn.Sequential(*resnet18_layers)				# Encoder = ResNet-18

		self.classifier = nn.Sequential(							# next Linear Classifier Layer
					nn.Linear(512, 128),
					nn.ReLU(),
					nn.BatchNorm1d(num_features=128),
					nn.Dropout(p=self.dropout),
					nn.Linear(128, opts.num_labels),				# Output Dim =  12
					nn.Sigmoid()									# each of the 12 output value range : 0-1 **  (later needs to be multiplied by 5)
					)
				

   
	def forward(self, images):
		batch_size = images.shape[0]
		features = self.encoder(images).reshape(batch_size, -1)

		labels = self.classifier(features)

		if not self.fm_distillation:
			return labels
		else:
			return labels, features
