import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import gpytorch
import torch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_spd_matrix
from WildFire_Model import WildFire
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from GP_mixture import mix_GPs
from tqdm import tqdm
import plotly.graph_objs as go

import matplotlib.pyplot as plt # creating visualizations
import numpy as np # basic math and random numbers
import torch # package for building functions with learnable parameters
import torch.nn as nn # prebuilt functions specific to neural networks
from torch.autograd import Variable # storing data while learning


from pytorch_tabular import TabularModel
from pytorch_tabular.models import (
	CategoryEmbeddingModelConfig,
	GatedAdditiveTreeEnsembleConfig,
	MDNConfig
)
from pytorch_tabular.config import (
	DataConfig,
	OptimizerConfig,
	TrainerConfig,
	ExperimentConfig,
)
# from pytorch_tabular.categorical_encoders import CategoricalEmbeddingTransformer
from pytorch_tabular.models.common.heads import LinearHeadConfig, MixtureDensityHeadConfig
np.random.seed(42)

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

		for i in range(self.fireAreas_Num):
			area_centers.append([areas_x[i], areas_y[i]])
			num_firespots.append(np.random.randint(low=5, high=15, size=1)[0])
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
				self.ign_points_all.append(self.fire_mdl[i].hotspot_init(i))        # initializing hotspots
				self.previous_terrain_map.append(self.fire_mdl[i].hotspot_init(i))  # initializing the starting terrain map
				self.geo_phys_info.append(self.fire_mdl[i].geo_phys_info_init(max_fuel_coeff=fuel_coeff, avg_wind_speed=wind_speed,
																			  avg_wind_direction=wind_direction))  # initialize geo-physical info
			# initializing the fire-map
			self.fire_map = []
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
		print('Generating fire data...')
		self.fire_init()
		for _ in tqdm(range(self.episodes)):
			self.fire_propagation()

		# return fire map: X,Y, Intensity, fire cluster number (index of fire_map)
		return self.fire_map[:, 0], self.fire_map[:, 1], self.fire_map[:, 2], self.fire_map[:, 3]
	

# class MDN_GP_Model_Spatiotemporal(gpytorch.models.ExactGP):
# 	def __init__(self, x_train, y_train):
# 		super(MDN_GP_Model_Spatiotemporal, self).__init__(x_train, y_train, gpytorch.likelihoods.GaussianLikelihood())
# 		self.mean_module = gpytorch.means.ConstantMean()
# 		self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
		
# 	def forward(self, x):
# 		mean_x = self.mean_module(x)
# 		covar_x = self.covar_module(x)
# 		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

import torch.nn.functional as F

class MDN(nn.Module):
	def __init__(self, n_hidden=100, n_gaussians=5):
		super(MDN, self).__init__()
		self.neurons = n_hidden
		self.components = n_gaussians
		
		self.h1 = nn.Linear(3, n_hidden)
		self.h2 = nn.Linear(n_hidden, n_hidden)
		
		self.alphas = nn.Linear(n_hidden, n_gaussians)
		self.mus = nn.Linear(n_hidden, n_gaussians)
		self.sigmas = nn.Linear(n_hidden, n_gaussians)

		self.ONEDIVSQRTTWOPI = 1.0 / np.sqrt(2.0*np.pi)
	
	def forward(self, inputs):
		x = F.relu(self.h1(inputs))
		x = F.relu(self.h2(x))
		
		pi = torch.softmax(self.alphas(x), dim=1)
		mu = self.mus(x)
		
		# use ELU for sigma
		sigma = nn.ELU()(self.sigmas(x)) + 1 + 1e-15
		
		return pi, sigma, mu	
	
	def gaussian_distribution(self, y, mu, sigma):
		# make |mu|=K copies of y, subtract mu, divide by sigma
		y = y.expand_as(sigma)
		res = (self.ONEDIVSQRTTWOPI/sigma) * torch.exp(-0.5 * (y-mu)**2 / sigma**2)

		return res

	def mdn_loss_fn(self, pi, sigma, mu, y):
		'''
		use LogsumExp trick for numerical stability: https://en.wikipedia.org/wiki/LogSumExp
		'''	
		# sigma = torch.tensor(sigma)
		# mu = torch.tensor(mu)
		# pi = torch.tensor(pi)
		result = -torch.log(sigma) - 0.5 * torch.log(2 * torch.tensor(np.pi)) - torch.pow(y - mu, 2) / (2 * torch.pow(sigma, 2))
		result = torch.logsumexp(torch.log(pi) + result, dim=1)

		# regularization
		# mu_diff = mu.unsqueeze(2) - mu.unsqueeze(1)
		# mu_dist = torch.norm(mu_diff, dim=1)
		# loss_reg = torch.sum(torch.exp(-mu_dist / 0.1))
		# alpha = 0.1
		# loss = -torch.mean(result + alpha * loss_reg)

		# return loss
		
		return -torch.mean(result)
	
	def train_mdn(self, x_variable, y_variable, optimizer):
		for epoch in range(10001):
			pi_variable, sigma_variable, mu_variable = network(x_variable)
			loss = self.mdn_loss_fn(pi_variable, sigma_variable, mu_variable, y_variable)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if epoch % 500 == 0:
				print(epoch, loss.item())
	
	
if __name__ == '__main__':
	network = MDN(n_hidden=5, n_gaussians=5)
	optimizer = torch.optim.Adam(network.parameters(), lr=0.0001, weight_decay=1e-4)

	for i in tqdm(range(50)):
		env = Fire(world_size=150, episodes=20)
		x, y, intensity, fire_cluster = env.generate_fire_data()
		min_max_scaler = MinMaxScaler()
		intensity = min_max_scaler.fit_transform(intensity.reshape(-1, 1)).reshape(-1)

		x_train = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), intensity.reshape(-1, 1)], axis=1)
		y_train = fire_cluster.reshape(-1, 1)

		x_train = torch.tensor(x_train, dtype=torch.float32)
		y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=False)

		network.train_mdn(x_train, y_train, optimizer)

		pi, sigma, mu = network(x_train)
		pi = pi.detach().numpy()
		sigma = sigma.detach().numpy()
		mu = mu.detach().numpy()

		pi = np.argmax(pi, axis=1)

		fig, ax = plt.subplots(1, 2, figsize=(10, 10))
		ax[0].scatter(x, y, c=pi, cmap='viridis', s=1)
		ax[1].scatter(x, y, c=fire_cluster, cmap='viridis', s=1)
		ax[0].set_title('prediction')
		ax[1].set_title('ground truth')
		plt.savefig(f'results/train/mdn_{i}.png')
		plt.close()

	torch.save(network.state_dict(), 'results/train/mdn.pt')

	env_test = Fire(world_size=150, episodes=20)
	x, y, intensity, fire_cluster = env_test.generate_fire_data()
	min_max_scaler = MinMaxScaler()
	intensity = min_max_scaler.fit_transform(intensity.reshape(-1, 1)).reshape(-1)

	x_test = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), intensity.reshape(-1, 1)], axis=1)
	y_test = fire_cluster.reshape(-1, 1)

	x_test = torch.tensor(x_test, dtype=torch.float32)
	y_test = torch.tensor(y_test, dtype=torch.float32, requires_grad=False)

	pi, sigma, mu = network(x_test)
	pi = pi.detach().numpy()
	sigma = sigma.detach().numpy()
	mu = mu.detach().numpy()

	pi = np.argmax(pi, axis=1)

	fig, ax = plt.subplots(1, 2, figsize=(10, 10))
	ax[0].scatter(x, y, c=pi, cmap='jet', s=10)
	ax[1].scatter(x, y, c=fire_cluster, cmap='jet', s=10)
	ax[0].set_title('prediction')
	ax[1].set_title('ground truth')
	plt.savefig(f'results/test/mdn.png')

	# # Train MDN + GP model
	# model_mdn = MDN_GP_Model_Spatiotemporal(x_train, y_train)
	# model_mdn.likelihood.noise = 1e-3
	# model_mdn.train()
	# mll_mdn = gpytorch.mlls.ExactMarginalLogLikelihood(model_mdn.likelihood, model_mdn)
	# optimizer_mdn = torch.optim.Adam(model_mdn.parameters(), lr=0.1)

	# n_epochs = 100
	# for i in tqdm(range(n_epochs)):
	# 	optimizer_mdn.zero_grad()
	# 	output_mdn = model_mdn(x_train)
	# 	loss_mdn = -mll_mdn(output_mdn, y_train.reshape(-1))
	# 	# print(output_mdn, y_train.reshape(-1).shape)
	# 	loss_mdn.backward()
	# 	optimizer_mdn.step()

	# # Initialize a Mixture of GPs model using EM
	# num_mixtures = 5
	# gmm = GaussianMixture(n_components=num_mixtures, covariance_type='full', random_state=0)
	# gmm.fit(x_train)
	# gp_models = []

	# # Create a GP model for each component
	# for i in range(num_mixtures):
	# 	x_train_x = x_train[gmm.predict(x_train) == i]
	# 	y_train_i = y_train[gmm.predict(x_train) == i]

	# 	model_gp_i = MDN_GP_Model_Spatiotemporal(x_train_x, y_train_i)
	# 	model_gp_i.likelihood.noise = 1e-3
	# 	model_gp_i.train()
	# 	mll_gp_i = gpytorch.mlls.ExactMarginalLogLikelihood(model_gp_i.likelihood, model_gp_i)
	# 	optimizer_gp_i = torch.optim.Adam(model_gp_i.parameters(), lr=0.1)

	# 	for epoch in range(n_epochs):
	# 		optimizer_gp_i.zero_grad()
	# 		output_gp_i = model_gp_i(x_train_x)
	# 		loss_gp_i = -mll_gp_i(output_gp_i, y_train_i.reshape(-1))
	# 		loss_gp_i.backward()
	# 		optimizer_gp_i.step()

	# 	gp_models.append(model_gp_i)

	# # Test data
	# test_y = gmm.predict(x_test)
	# x_test = x_test.numpy()  # Convert to NumPy

	# # Convert x_test to PyTorch tensor
	# x_test = torch.tensor(x_test, dtype=torch.float32)

	# # Generate predictions for MDN + GP model
	# model_mdn.eval()
	# with torch.no_grad():
	# 	predictions_mdn = model_mdn(x_test)

	# # Extract the means of MDN
	# mdn_means = predictions_mdn.mean.reshape(-1, 1).detach().numpy()

	# print(mdn_means)