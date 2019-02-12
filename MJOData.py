import os

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np


class MJODataset(Dataset):
	"""
	This class is used to load the data
		into the CNN.
	"""

	def __init__(self, events_file, root_dir, channels):
		"""
		Initialize the MJODataset instance

		Parameters
		----------

		events_file : string
		The location of the .npy file to be loaded

		root_dir: string
		The directory containing the variable .npy files

		root_dir : ndarray, shape (n_examples,)
		Array of labels.

		channels: list
		List of which variables to include within the dataset

		"""

		#Load the .npy data file for the given sample...
		MJO_frame = np.load(events_file)

		self.MJO_frame = MJO_frame
		self.root_dir = root_dir
		self.channels = channels


	def __getitem__(self, index):
		"""
		Initialize the MJODataset instance

		Parameters
		----------

		index: int
		The index of the sample that is being extracted

		"""

		#Declare the file path for the sample
		sample_name = os.path.join(self.root_dir,
								self.MJO_frame[index,0])
		#Extract the image in (n_channels, n_latitudes, n_longitudes) format
		sample = torch.from_numpy(np.load(img_name + '.npy'))
		#Now extract only the channels (variables) that we want to use
		sample = sample[self.channels]
		#Extract the label for the image
		label = int(float(self.MJO_frame[index,1]))

		#Return the sample and its label
		return sample, label

	def __len__(self):
		return len(self.MJO_frame)



