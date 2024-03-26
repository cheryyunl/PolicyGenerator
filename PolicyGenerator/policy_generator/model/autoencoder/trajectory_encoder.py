import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

class TrajectoryEncoder(nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.ln = nn.Linear(input_dim, output_dim)
		# raise NotImplementedError("Encoder not implemented.")
	def forward(self, traj_dict):
		# traj_dict: must include "trajectory" key
		return self.ln(traj_dict["trajectory"])
	def output_shape(self):
		return self.output_dim