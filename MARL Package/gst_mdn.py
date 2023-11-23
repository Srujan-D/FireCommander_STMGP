import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
# import gpytorch
import torch
# from sklearn.mixture import GaussianMixture
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import make_spd_matrix
from WildFire_Model import WildFire
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import torch.nn.functional as F

# from GP_mixture import mix_GPs
from tqdm import tqdm
# import plotly.graph_objs as go

import matplotlib.pyplot as plt # creating visualizations
import numpy as np # basic math and random numbers
import torch # package for building functions with learnable parameters
import torch.nn as nn # prebuilt functions specific to neural networks
# from torch.autograd import Variable # storing data while learning
from torch.utils.data import DataLoader


# from pytorch_tabular import TabularModel
# from pytorch_tabular.models import (
# 	CategoryEmbeddingModelConfig,
# 	GatedAdditiveTreeEnsembleConfig,
# 	MDNConfig
# )
# from pytorch_tabular.config import (
# 	DataConfig,
# 	OptimizerConfig,
# 	TrainerConfig,
# 	ExperimentConfig,
# )
# # from pytorch_tabular.categorical_encoders import CategoricalEmbeddingTransformer
# from pytorch_tabular.models.common.heads import LinearHeadConfig, MixtureDensityHeadConfig
# np.random.seed(42)

class Fire(object):
	def __init__(self, world_size=250, duration=200, fireAreas_Num=5, episodes=200):
		self.world_size = world_size
		self.duration = duration
		self.fireAreas_Num = fireAreas_Num
		self.episodes = episodes

		# fire model parameters
		areas_x = np.random.randint(20, self.world_size - 20, self.fireAreas_Num)
		areas_y = np.random.randint(20, self.world_size - 20, self.fireAreas_Num)
		area_delays = [0] * self.fireAreas_Num
		area_fuel_coeffs = [10] * self.fireAreas_Num
		area_wind_speed = [10] * self.fireAreas_Num
		area_wind_directions = []
		area_centers = []
		num_firespots = []
		self.fire_map = []

		for i in range(self.fireAreas_Num):
			area_centers.append([areas_x[i], areas_y[i]])
			num_firespots.append(10)
			area_wind_directions.append(random.choice([0, 45, 90, 135, 180]))
		self.fire_info = [area_centers,            # [[area1_center_x, area1_center_y], [area2_center_x, area2_center_y], ...],
						  [num_firespots,          # [[num_firespots1, num_firespots2, ...],
						   area_delays,            # [area1_start_delay, area2_start_delay, ...],
						   area_fuel_coeffs,       # [area1_fuel_coefficient, area2_coefficient, ...],
						   area_wind_speed,        # [area1_wind_speed, area2_wind_speed, ...],
						   area_wind_directions,   # [area1_wind_direction, area2_wind_direction, ...],
						   1.25,                   # temporal penalty coefficient,
						   0.1,                    # fire propagation weight,
						   90,                     # Action Pruning Confidence Level (In percentage),
						   80,                     # Hybrid Pruning Confidence Level (In percentage),
						   1]]
		
		self.gaussian_processes = []

	# initialize the fire model
	def fire_init(self):
		# Fire region (Color: Red (255, 0, 0))
		# The wildfire generation and propagation utilizes the FARSITE wildfire mathematical model
		# To clarify the fire state data, the state of the fire spot at each moment is stored in the dictionary list separately
		# Besides, the current fire map will also be stored as a matrix with the same size of the simulation model, which
		# reflects the fire intensity of each position on the world

		# create the fire state dictionary list
		self.fire_States_List = []
		for i in range(self.fireAreas_Num):
			self.fire_States_List.append([])
		# length and width of the terrain as a list [length, width]
		terrain_sizes = [self.world_size, self.world_size]
		hotspot_areas = []
		for i in range(self.fireAreas_Num):
			hotspot_areas.append([self.fire_info[0][i][0] - 5, self.fire_info[0][i][0] + 5,
								  self.fire_info[0][i][1] - 5, self.fire_info[0][i][1] + 5])

		# checking fire model setting mode and initializing the fire model
		if self.fire_info[1][9] == 0:  # when using "uniform" fire setting (all fire areas use the same parameters)
			# initial number of fire spots (ignition points) per hotspot area
			num_ign_points = self.fire_info[1][0]
			# fuel coefficient for vegetation type of the terrain (higher fuel_coeff:: more circular shape fire)
			fuel_coeff = self.fire_info[1][2]
			# average mid-flame wind velocity (higher values streches the fire more)
			wind_speed = self.fire_info[1][3]
			# wind azimuth
			wind_direction = np.pi * 2 * self.fire_info[1][4] / 360  # converting degree to radian

			# Init the wildfire model
			self.fire_mdl = WildFire(terrain_sizes=terrain_sizes, hotspot_areas=hotspot_areas, num_ign_points=num_ign_points, duration=self.duration,
									 time_step=1, radiation_radius=10, weak_fire_threshold=5, flame_height=3, flame_angle=np.pi / 3)
			print('all fires same..')
			self.ign_points_all = self.fire_mdl.hotspot_init()      # initializing hotspots
			self.fire_map = self.ign_points_all                     # initializing fire-map
			self.previous_terrain_map = self.ign_points_all.copy()  # initializing the starting terrain map
			self.geo_phys_info = self.fire_mdl.geo_phys_info_init(max_fuel_coeff=fuel_coeff, avg_wind_speed=wind_speed,
																  avg_wind_direction=wind_direction)  # initialize geo-physical info
		else:  # when using "Specific" fire setting (each fire area uses its own parameters)
			self.fire_mdl = []
			self.geo_phys_info = []
			self.ign_points_all = []
			self.previous_terrain_map = []
			self.new_fire_front_temp = []
			self.current_geo_phys_info = []
			# initialize fire areas separately
			for i in range(self.fireAreas_Num):
				self.new_fire_front_temp.append([])
				self.current_geo_phys_info.append([])
				# initial number of fire spots (ignition points) per hotspot area
				num_ign_points = self.fire_info[1][0][i]
				# fuel coefficient for vegetation type of the terrain (higher fuel_coeff:: more circular shape fire)
				fuel_coeff = self.fire_info[1][2][i]
				# average mid-flame wind velocity (higher values streches the fire more)
				wind_speed = self.fire_info[1][3][i]
				# wind azimuth
				wind_direction = np.pi * 2 * self.fire_info[1][4][i] / 360  # converting degree to radian

				# init the wildfire model
				self.fire_mdl.append(WildFire(
					terrain_sizes=terrain_sizes, hotspot_areas=[hotspot_areas[i]], num_ign_points=num_ign_points, duration=self.duration, time_step=1,
					radiation_radius=10, weak_fire_threshold=5, flame_height=3, flame_angle=np.pi / 3))
				self.ign_points_all.append(self.fire_mdl[i].hotspot_init(cluster_num=i+1))        # initializing hotspots
				self.previous_terrain_map.append(self.fire_mdl[i].hotspot_init(cluster_num=i+1))  # initializing the starting terrain map
				self.geo_phys_info.append(self.fire_mdl[i].geo_phys_info_init(max_fuel_coeff=fuel_coeff, avg_wind_speed=wind_speed,
																			  avg_wind_direction=wind_direction))  # initialize geo-physical info
			# initializing the fire-map
			# self.fire_map = []
			for i in range(self.fireAreas_Num):
				for j in range(len(self.ign_points_all[i])):
					self.fire_map.append(self.ign_points_all[i][j])
			self.fire_map = np.array(self.fire_map)
			self.fire_map_spec = self.ign_points_all

		# the lists to store the firespots in different state, coordinates only
		self.onFire_List = []  # the onFire_List, store the points currently on fire (sensed points included, pruned points excluded)

	# propagate fire one step forward according to the fire model
	def fire_propagation(self):
		self.pruned_List = []
		# checking fire model setting mode and initializing the fire model
		if self.fire_info[1][9] == 0:  # when using "uniform" fire setting (all fire areas use the same parameters)
			self.new_fire_front, current_geo_phys_info =\
				self.fire_mdl.fire_propagation(self.world_size, ign_points_all=self.ign_points_all, geo_phys_info=self.geo_phys_info,
											   previous_terrain_map=self.previous_terrain_map, pruned_List=self.pruned_List)
			updated_terrain_map = self.previous_terrain_map
		else:  # when using "Specific" fire setting (each fire area uses its own parameters)
			updated_terrain_map = self.previous_terrain_map
			for i in range(self.fireAreas_Num):
				self.new_fire_front_temp[i], self.current_geo_phys_info[i] =\
					self.fire_mdl[i].fire_propagation(self.world_size, ign_points_all=self.ign_points_all[i], geo_phys_info=self.geo_phys_info[i],
													  previous_terrain_map=self.previous_terrain_map[i], pruned_List=self.pruned_List)

			# update the new firefront list by combining all region-wise firefronts
			self.new_fire_front = []
			for i in range(self.fireAreas_Num):
				for j in range(len(self.new_fire_front_temp[i])):
					self.new_fire_front.append(self.new_fire_front_temp[i][j])
			self.new_fire_front = np.array(self.new_fire_front)

		# update the region-wise fire map
		if self.fire_info[1][9] == 1:
			for i in range(self.fireAreas_Num):
				self.fire_map_spec[i] = np.concatenate([self.fire_map_spec[i], self.new_fire_front_temp[i]], axis=0)
		else:
			self.fire_map_spec = self.fire_map

		# updating the fire-map data for next step
		if self.new_fire_front.shape[0] > 0:
			self.fire_map = np.concatenate([self.fire_map, self.new_fire_front], axis=0)  # raw fire map without fire decay

		# update the fire propagation information
		if self.fire_info[1][9] == 1:
			ign_points_all_temp = []
			for i in range(self.fireAreas_Num):
				if self.new_fire_front_temp[i].shape[0] > 0:
					# fire map with fire decay
					self.previous_terrain_map[i] = np.concatenate((updated_terrain_map[i], self.new_fire_front_temp[i]), axis=0)
				ign_points_all_temp.append(self.new_fire_front_temp[i])
			self.ign_points_all = ign_points_all_temp
		else:
			if self.new_fire_front.shape[0] > 0:
				self.previous_terrain_map = np.concatenate((updated_terrain_map, self.new_fire_front))  # fire map with fire decay
				self.ign_points_all = self.new_fire_front
	
	def generate_fire_data(self):
		# print('Generating fire data...')
		self.fire_init()
		means = self.ign_points_all.copy()
		for _ in range(self.episodes):
			self.fire_propagation()
	
		# return fire map: X,Y, Intensity, fire cluster number (index of fire_map)
		return self.fire_map[:, 0], self.fire_map[:, 1], self.fire_map[:, 2], self.fire_map[:, 3], means

'''
class MDN(nn.Module):
	def __init__(self, n_hidden=20, n_gaussians=5, n_input=3, mu_bias=None):
		super(MDN, self).__init__()
		self.neurons = n_hidden
		self.components = n_gaussians
		self.mu_bias = mu_bias
		
		# self.h1 = nn.Linear(2, n_hidden)	#if intensity is not in input
		self.h1 = nn.Linear(n_input, n_hidden)	#if intensity is in input
		self.h2 = nn.Linear(n_hidden, n_hidden)
		# self.h3 = nn.Linear(n_hidden, n_hidden)
	
		self.alphas = nn.Linear(n_hidden, n_gaussians)
		self.mus = nn.Linear(n_hidden, n_gaussians, bias=mu_bias)
		self.sigmas = nn.Linear(n_hidden, n_gaussians)

		self.ONEDIVSQRT2PI = 1.0 / np.sqrt(2.0*np.pi)

		# self.pi = nn.Sequential(nn.Linear(n_input, n_gaussians*))

		
		# Weight Regularization – 
		# 	Applying L1 or L2 regularization to the weights of the neurons which compute the mean, variances and mixing components.
		# Bias Initialization – 
		# 	If we precompute the possible centers of the two gaussians, we can initialize the bias of the \mu layers to these centers.
		# 	This has shown to have a strong effect in the separation of the two gaussian kernels/components during training.
		
	
	def forward(self, inputs):
		x = F.tanh(self.h1(inputs))
		x = F.relu(self.h2(x))
		# x = F.relu(self.h3(x))
		
		pi = torch.softmax(self.alphas(x), dim=1)
		mu = self.mus(x)
		# for multilabel output, should i put softmax?
		# mu = torch.softmax(self.mus(x), dim=1)

		# use ELU for sigma
		sigma = nn.ELU()(self.sigmas(x)) + 1 + 1e-15
		
		return pi, sigma, mu
	
	def gaussian_distribution(self, y, mu, sigma):
		# make |mu|=K copies of y, subtract mu, divide by sigma
		y = y.expand_as(sigma)
		res = -torch.log(sigma) - 0.5 * torch.log(2 * torch.tensor(np.pi)) - 0.5 * torch.pow((y - mu) / sigma, 2)
		# res = (self.ONEDIVSQRT2PI/sigma) * torch.exp((-0.5 * (y-mu)/sigma)**2)
		return res

	def mdn_loss_fn(self, y, mu, sigma, pi):
		
		# use LogsumExp trick for numerical stability: https://en.wikipedia.org/wiki/LogSumExp
		
		log_component_prob = self.gaussian_distribution(y, mu, sigma)
		log_mix_prob = torch.log(nn.functional.gumbel_softmax(pi, tau=1, dim=-1)) + 1e-15
		# TODO: # add 1e-15 inside the torch.log(pi + 1e-15)
		# log_mix_prob = torch.log(pi + 1e-15)
		# log_mix_prob = torch.log(pi) + 1e-15 

		result = torch.logsumexp(log_component_prob + log_mix_prob, dim=-1)		
		return torch.mean(-result)
	
	def train_mdn(self, x_variable, y_variable, optimizer):
		# for epoch in range(3001):
		pi_variable, sigma_variable, mu_variable = network(x_variable)
		loss = self.mdn_loss_fn(y_variable, mu_variable, sigma_variable, pi_variable)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# print(loss.item())
			# if epoch % 500 == 0:
			# 	print(epoch, loss.item())
		# if abs(loss)<0.5:
		# 	# plot the prediction
		# 	fig, ax = plt.subplots(1, 2, figsize=(20, 10))
		# 	ax[0].scatter(x_variable[:,0].detach().numpy(), x_variable[:,1].detach().numpy(), c=np.max(pi_variable.detach().numpy(), axis=1), cmap='jet', s=1)
		# 	ax[1].scatter(x_variable[:,0].detach().numpy(), x_variable[:,1].detach().numpy(), c=y_variable[:,0].detach().numpy(), cmap='viridis', s=1)
		# 	ax[0].set_title('prediction')
		# 	ax[1].set_title('ground truth')
		# 	plt.savefig(f'results/train/2GP/new_mean/low_loss_mdn_{i}.png')
		# 	plt.close()
		# 	plt.show()
'''

class MDN(nn.Module):
	def __init__(self, n_hidden=200, n_gaussians=2, n_input=1):
		super(MDN, self).__init__()
		self.z_h = nn.Sequential(
			nn.Linear(n_input, n_hidden),
			nn.Tanh()
		)
		self.z_pi = nn.Linear(n_hidden, n_gaussians)
		self.z_mu = nn.Linear(n_hidden, n_gaussians)
		self.z_sigma = nn.Linear(n_hidden, n_gaussians)
	
	def forward(self, x):
		z_h = self.z_h(x)
		pi = F.softmax(self.z_pi(z_h), -1)
		mu = self.z_mu(z_h)
		sigma = torch.exp(self.z_sigma(z_h))
		return pi, mu, sigma

	def gaussian_distribution(self, y, mu, sigma):
		# make |mu|=K copies of y, subtract mu, divide by sigma
		y = y.expand_as(sigma)
		res = -torch.log(sigma) - 0.5 * torch.log(2 * torch.tensor(np.pi)) - 0.5 * torch.pow((y - mu) / sigma, 2)
		# res = (self.ONEDIVSQRT2PI/sigma) * torch.exp((-0.5 * (y-mu)/sigma)**2)
		return res
	
	def mdn_loss_fn(self, y, mu, sigma, pi):
		
		# use LogsumExp trick for numerical stability: https://en.wikipedia.org/wiki/LogSumExp
		
		log_component_prob = self.gaussian_distribution(y, mu, sigma)
		log_mix_prob = torch.log(nn.functional.gumbel_softmax(pi, tau=1, dim=-1)) + 1e-15
		# log_mix_prob = torch.log(pi + 1e-15)

		result = torch.logsumexp(log_component_prob + log_mix_prob, dim=-1)		
		return torch.mean(-result)
	


class BuildData():
	def __init__(self, data_size=10000, world_size=250, episodes=10, fireAreas_Num=5):
		self.world_size = world_size
		self.episodes = episodes
		self.fireAreas_Num = fireAreas_Num

		self.env_list = []

		for i in tqdm(range(data_size)):
			env = Fire(world_size=self.world_size, episodes=self.episodes, fireAreas_Num=self.fireAreas_Num)
			self.env_list.append(env)

	def __len__(self):
		return len(self.env_list)
	
	def __getitem__(self, idx):
		env = self.env_list[idx]
		x, y, intensity, fire_cluster, fire_cluster_means = env.generate_fire_data()
		fire_means = []
		for cluster in fire_cluster_means:
			fire_means.append(np.mean(cluster, axis=0, dtype=np.int32))

		# unique_cluster_num = np.unique(fire_cluster)
		# print('fire shape:', fire_cluster.shape)
		# fire_means1 = np.array(fire_means)
		# print('fire means shape:', fire_means1.shape)
		# for i in range(fire_cluster.shape[0]):
		# 	for j in range(len(unique_cluster_num)):
		# 		if fire_cluster[i] == unique_cluster_num[j]:
		# 			# print(means[j])
		# 			fire_cluster[i] = fire_means1[j]

		
		x_train = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), intensity.reshape(-1, 1)], axis=1)	#with intensity
		# x_train = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)	#without intensity
		y_train = fire_cluster.reshape(-1, 1)
		# y_train = intensity.reshape(-1, 1)

		x_train = torch.tensor(x_train, dtype=torch.float32)
		y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=False)
		fire_means = torch.tensor(np.array(fire_means), dtype=torch.float32)

		if not x_train.shape[0] == y_train.shape[0]:
			print("false")

		return x_train, y_train, fire_means

def collate_fn(batch):
	# Find the maximum size for x_train and y_train
	max_x_size = max(x.size(0) for x, _ in batch)
	max_y_size = max(y.size(0) for _, y in batch)

	# Initialize lists to store padded x_train and y_train
	padded_x = []
	padded_y = []

	# Loop through the batch
	for x, y in batch:
		# Calculate how many times to replicate the data to match the maximum size
		x_replicas = max_x_size // x.size(0)
		y_replicas = max_y_size // y.size(0)

		# Replicate x and y data to match the maximum size
		x_padded = x.repeat(x_replicas, 1)
		y_padded = y.repeat(y_replicas, 1)

		# Append the replicated data to the padded lists
		padded_x.append(x_padded)
		padded_y.append(y_padded)

	# Stack the padded data to form tensors
	x_train = torch.stack(padded_x)
	y_train = torch.stack(padded_y)

	return x_train, y_train

# if __name__ == '__main__':
# 	network = MDN(n_hidden=20, n_gaussians=2)
# 	optimizer = torch.optim.Adam(network.parameters(), lr=0.00001, weight_decay=1e-4)

# 	train_dataset = BuildData(data_size=1000, world_size=100, episodes=10, fireAreas_Num=2)
# 	train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=64) #, collate_fn=collate_fn)
# 	batch_size = 32
# 	n_epochs = 1000
# 	for i in tqdm(range(n_epochs)):
# 		for x_train, y_train, fire_means  in train_dataloader:
# 			# x_train, y_train = batch
# 			# print(x_train.shape, y_train.shape)
			
# 			# network.mu_bias = fire_means
# 			network.train_mdn(x_train, y_train, optimizer)
# 			pi, sigma, mu = network(x_train)

# 			with open('results/train/2GP/new_mean/loss.txt', 'a') as f:
# 				f.write(str(network.mdn_loss_fn(y_train, mu, sigma, pi).item())+'\n')

# 			pi = pi.detach().numpy()
# 			sigma = sigma.detach().numpy()
# 			mu = mu.detach().numpy()
# 			# print('means', mu)
# 			# print('sigma', sigma)
# 			pi1 = np.max(pi, axis=2)
# 			pi2 = np.argmax(pi, axis=2)
# 			# print('alphas', pi)

# 			# mu1 = mu[np.indices(pi2.shape)[0], pi2]
# 			# result = array2[np.arange(32)[:, np.newaxis], np.arange(550), argmax_indices]
# 			# print(pi1.shape, pi2.shape)
# 			mu1 = mu[np.arange(pi1.shape[0])[:, np.newaxis], np.arange(pi1.shape[1]), pi2]
# 			# mu1 = mu[:, pi2]
# 			sigma1 = sigma[np.indices(pi2.shape)[0], pi2]

# 			# print(pi2.shape, pi.shape, pi1.shape, mu1.shape, mu.shape)

# 			if i%10 == 0:
# 				fig, ax = plt.subplots(1, 3, figsize=(45,10))
# 				fig.colorbar(
#         						ax[0].scatter(
# 									x_train[5][:,0].detach().numpy(),
# 									x_train[5][:,1].detach().numpy(),
# 									c=pi2[5],
# 									cmap='jet',
# 									s=5
# 								),
# 								ax = ax[0]
# 							)
# 				fig.colorbar(
#         						ax[1].scatter(
# 									x_train[5][:,0].detach().numpy(),
# 									x_train[5][:,1].detach().numpy(),
# 									c=mu1[5],
# 									cmap='jet',
# 									s=5
# 								),
# 								ax=ax[1]
#        						)
# 				ax[2].scatter(
# 								x_train[5][:,0].detach().numpy(),
# 				  				x_train[5][:,1].detach().numpy(),
# 					  			c=y_train[5][:,0].detach().numpy(),
# 						 		cmap='jet',
# 						   		s=5
# 							)
# 				ax[0].set_title('highest confidence alpha')
# 				ax[1].set_title('prediction mean')
# 				ax[2].set_title('ground truth')
# 				plt.tight_layout()
# 				plt.savefig(f'results/train/2GP/new_mean/mdn_{i}.png')
# 				plt.close()

# 				# fig, ax = plt.subplots(1, 2, figsize=(20, 10))
# 				# ax[0].scatter(x_train[:,0].detach().numpy(), x_train[:,1].detach().numpy(), c=pi1, cmap='jet', s=1)
# 				# ax[1].scatter(x_train[:,0].detach().numpy(), x_train[:,1].detach().numpy(), c=y_train[:,0].detach().numpy(), cmap='viridis', s=1)
# 				# ax[0].set_title('prediction')
# 				# ax[1].set_title('ground truth')
# 				# plt.savefig(f'results/train/2GP/new_mean/low_loss_mdn_{i}.png')
# 				# plt.close()
			

# 				torch.save(network.state_dict(), f'results/train/2GP/new_mean/mdn_{i}.pt')


# 	torch.save(network.state_dict(), f'results/train/2GP/new_mean/mdn_final.pt')


# testing if MDN is working on simple data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.patches as mpatches


def remove_ax_window(ax):
	"""
		Remove all axes and tick params in pyplot.
		Input: ax object.
	"""
	ax.spines["top"].set_visible(False)
	ax.spines["bottom"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["left"].set_visible(False)
	ax.tick_params(axis=u'both', which=u'both',length=0)

if __name__ == '__main__':
	samples = int(1e5)
	dpi = 140
	x_size = 8
	y_size = 4
	alt_font_size = 14
	n_input = 1

	x_data = np.float32(np.random.uniform(-10, 10, (1, samples)))
	r_data = np.array([np.random.normal(scale=np.abs(i)) for i in x_data])
	y_data = np.float32(np.square(x_data)+r_data*2.0)

	x_data2 = np.float32(np.random.uniform(-10, 10, (1, samples)))
	r_data2 = np.array([np.random.normal(scale=np.abs(i)) for i in x_data2])
	y_data2 = np.float32(-np.square(x_data2)+r_data2*2.0)

	x_data = np.concatenate((x_data,x_data2),axis=1).T
	y_data = np.concatenate((y_data,y_data2),axis=1).T

	min_max_scaler = MinMaxScaler()
	y_data = min_max_scaler.fit_transform(y_data)

	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42, shuffle=True)

	# fig = plt.figure(figsize=(x_size,y_size), dpi=dpi)
	# ax = plt.gca()

	# ax.set_title(r"$y = \pm x^2 + \epsilon$"+"\n"+r"$\epsilon\backsim\mathcal{N}(0,|x|)$", fontsize=alt_font_size)
	# ax.plot(x_train,y_train, "x",alpha=1., color=sns.color_palette()[0])

	# remove_ax_window(ax)
	# plt.show()

	components = 2
	neurons = 200

	print('training...')
	network = MDN(n_hidden=neurons, n_gaussians=components, n_input=n_input)
	optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)
	
	x_train = torch.from_numpy(x_train)
	y_train = torch.from_numpy(y_train)
	x_test = torch.from_numpy(x_test)
	y_test = torch.from_numpy(y_test)

	for i in tqdm(range(100)):
		# network.train_mdn(x_train, y_train, optimizer)
		# pi, sigma, mu = network(x_train)
		# print(f'loss: {network.mdn_loss_fn(y_train, mu, sigma, pi).item()}')
		pi, mu, sigma = network(x_train)
		loss = network.mdn_loss_fn(y_train, mu, sigma, pi)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# if i%20 == 0:
		# 	print(loss.data.tolist())

	s = np.linspace(-10,10,int(x_test.shape[0]))[:, np.newaxis].astype(np.float32)

	pi, sigma, mu = network(x_test)

	pi = pi.detach().numpy()
	sigma = sigma.detach().numpy()
	mu = mu.detach().numpy()

	fig = plt.figure(figsize=(x_size,y_size), dpi=dpi)
	ax = plt.gca()

	ax.set_title(r"$y = \pm x^2 + \epsilon$"+"\n"+r"$\epsilon\backsim\mathcal{N}(0,|x|)$", fontsize=alt_font_size)
	ax.plot(x_train,y_train, "x",alpha=0.5, color=sns.color_palette()[0])
	
	for mx in range(components):
		# print(s.shape, mu[:,mx].shape)
		ax.scatter(x_test, mu[:,mx], color=sns.color_palette()[1+mx])
		# plt.plot(x_test, y_test, color=sns.color_palette()[1+mx], linewidth=5, linestyle='-', markersize=3)
		# plt.plot(s,mu[:,mx]-sigma[:,mx], color=sns.color_palette()[1+mx],linewidth=3, linestyle='--', markersize=3)
		# plt.plot(s,mu[:,mx]+sigma[:,mx], color=sns.color_palette()[1+mx],linewidth=3, linestyle='--', markersize=3)

	remove_ax_window(ax)

	data_leg = mpatches.Patch(color=sns.color_palette()[0])
	data_mdn1 = mpatches.Patch(color=sns.color_palette()[1])
	data_mdn2 = mpatches.Patch(color=sns.color_palette()[2])

	ax.legend(handles = [data_leg, data_mdn1, data_mdn2],
			labels = ["Data", "MDN (c=1)", "MDN (c=2)"],
			loc=9, borderaxespad=0.1, framealpha=1.0, fancybox=True,
			bbox_to_anchor=(0.5, -0.05), ncol=6, shadow=True, frameon=False,
			fontsize=alt_font_size)

	plt.tight_layout()

	# if save_figure:
	# 	plt.savefig("graphics/mdn_nonlinear_prediction.png", format='png',dpi=dpi, bbox_inches='tight')

	plt.show()

