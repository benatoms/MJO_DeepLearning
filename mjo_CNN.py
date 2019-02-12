import sys
import math

import torch.nn as nn

class MJONet(nn.Module):
	def __init__(self, num_channels, num_classes, dropout_rate):
		super().__init__()

		"""
		Initialize the neural network 

		Parameters
		----------

		num_channels : int
		Number of input channels

		num_classes : int
		Number of classes in final layer

		dropout_rate : float
		dropout rate for each dropout instance

		"""

		#Convolutional layer one
		self.add_module('conv0', nn.Conv2d(num_channels, 24, kernel_size=5, padding=5//2, stride=1, bias=False))
		self.add_module('act0', nn.LeakyReLU(negative_slope=0.003, inplace=True))
		self.add_module('drop0', nn.Dropout(p=dropout_rate))
		self.add_module('pool0', nn.AvgPool2d(kernel_size=3, stride=3, padding=1)),

		#Convolutional layer two
		self.add_module('conv1', nn.Conv2d(24, 36, kernel_size=5, padding=5//2, stride=1, bias=False))
		self.add_module('act1', nn.LeakyReLU(negative_slope=0.003, inplace=True))
		self.add_module('drop1', nn.Dropout(p=dropout_rate))
		self.add_module('pool1', nn.AvgPool2d(kernel_size=3, stride=3, padding=1)),

		#Convolutional layer three
		self.add_module('conv2', nn.Conv2d(36, 54, kernel_size=3, padding=3//2, stride=1, bias=False))
		self.add_module('act2', nn.LeakyReLU(negative_slope=0.003, inplace=True))

		#Fully connected layer
		self.add_module('linear', nn.Linear(15120, num_classes))
		self.add_module('logsoftmax', nn.LogSoftmax())

		#Randomly initialize the weights of the CNN
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.bias.data.zero_()

	def forward(self, x):
		"""
		Pass a sample forward through the neural network

		Parameters
		----------

		x : ndarray, initial shape (n_samples, n_channels, n_latitude, n_longitude)
		Input batch

		"""

		#Gather information on batch size (in index 0)
		x_init_shape = x.shape

		#Pass through first convolution layer
		x = self.pool0(self.drop0(self.act0(self.conv0(x))))

		#...and second convolution layer
		x = self.pool1(self.drop1(self.act1(self.conv1(x))))

		#...and third convolution layer
		x = self.act2(self.conv2(x))

		#...reshape for the fully connected layer
		x = x.view(x_init_shape[0],-1)

		#...and pass through the fully connected layer
		x = self.linear(x)
		x = self.logsoftmax(x)

		#Return the output of the CNN
		return x

