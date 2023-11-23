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


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.datasets import fetch_california_housing
from scipy.special import softmax

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import multiprocessing
from torch.multiprocessing import Pool

import plotly.express as px
import plotly.graph_objs as go

from pytorch_tabular import TabularModel
from pytorch_tabular.models import (
	CategoryEmbeddingModelConfig,
	GatedAdditiveTreeEnsembleConfig,
	MDNConfig
)
from pytorch_tabular.utils import get_gaussian_centers

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
		X = []
		Y = []
		intensity = []
		fire_cluster_id = []
		time_stamp = []
		self.fire_init()
		means = self.ign_points_all.copy()
		for i in range(self.episodes):
			self.fire_propagation()
			# print(self.fire_map[:, 0], self.fire_map[:, 0].shape)
			X.extend(self.fire_map[:, 0])
			Y.extend(self.fire_map[:, 1])
			intensity.extend(self.fire_map[:, 2])
			fire_cluster_id.extend(self.fire_map[:, 3])
			time_stamp.extend(np.ones(self.fire_map.shape[0]) * i)
	
		# return fire map: X,Y, Intensity, time, fire cluster number (index of fire_map)
		# return self.fire_map[:, 0], self.fire_map[:, 1], self.fire_map[:, 2], self.fire_map[:, 3], means
		X = np.array(X)
		Y = np.array(Y)
		intensity = np.array(intensity)	
		fire_cluster_id = np.array(fire_cluster_id)
		time_stamp = np.array(time_stamp)
		
		return X, Y, intensity, fire_cluster_id, time_stamp, means

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

		
		# x_train = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), intensity.reshape(-1, 1)], axis=1)	#with intensity
		x_train = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)	#without intensity
		# y_train = fire_cluster.reshape(-1, 1)
		y_train = intensity.reshape(-1, 1)

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

# Function to generate data
def generate_data(_):
    env = Fire(world_size=world_size, episodes=episodes, fireAreas_Num=fireAreas_Num)
    x, y, intensity, fire_cluster_id, time_stamp, fire_cluster_means = env.generate_fire_data()
    return {'x': x, 'y': y, 'time_stamp': time_stamp, 'target': intensity}

if __name__ == '__main__':

	# target_col = "target"
	# data = fetch_california_housing(return_X_y=False)
	# X = pd.DataFrame(data['data'], columns=data['feature_names'])
	# cont_cols = X.columns.tolist()
	# cat_cols = []
	# y = data['target']
	# X[target_col] = y
	# df_train, df_test = train_test_split(X, test_size=0.2, random_state=42)
	# print(df_train)

	data_size = 100
	valid_size = 10
	test_size = 1
	world_size = 100
	fireAreas_Num = 7
	episodes = 20
	env_list = []
	df_train = pd.DataFrame(columns=['x', 'y', 'time_stamp', 'target'])
	df_valid = pd.DataFrame(columns=['x', 'y', 'time_stamp', 'target'])
	df_test = pd.DataFrame(columns=['x', 'y', 'time_stamp', 'target'])

	pool = Pool(multiprocessing.cpu_count()-10)

	train_data = list(range(data_size))
	results = list(tqdm(pool.imap(generate_data, train_data), total=data_size))
	for result in results:
		df_train = pd.concat([df_train, pd.DataFrame({'x': result['x'], 'y': result['y'], 'time_stamp': result['time_stamp'], 'target': result['target']})], ignore_index=True)

	# for i in tqdm(range(data_size)):
	# 	env = Fire(world_size=world_size, episodes=episodes, fireAreas_Num=fireAreas_Num)
	# 	# env_list.append(env)
	# 	x, y, intensity, fire_cluster, fire_cluster_means = env.generate_fire_data()
	# 	df_train = pd.concat([df_train, pd.DataFrame({'x': x, 'y': y, 'target': intensity})], ignore_index=True)

	# print(df_train)
	target_col = "target"
	# X = df_train.drop(target_col, axis=1)
	X = df_train
	y = df_train[target_col]

	cont_cols = X.columns.tolist()

	# cont_cols.remove('time_stamp')
	# cat_cols = ['time_stamp']
	cat_cols = []

	# print('cont', cont_cols)

	valid_data = list(range(valid_size))
	results = list(tqdm(pool.imap(generate_data, valid_data), total=data_size))
	for result in results:
		df_valid = pd.concat([df_valid, pd.DataFrame({'x': result['x'], 'y': result['y'], 'time_stamp': result['time_stamp'], 'target': result['target']})], ignore_index=True)

	# for i in tqdm(range(valid_size)):
	# 	env = Fire(world_size=world_size, episodes=episodes, fireAreas_Num=fireAreas_Num)
	# 	# env_list.append(env)
	# 	x, y, intensity, fire_cluster, fire_cluster_means = env.generate_fire_data()
	# 	df_valid = pd.concat([df_valid, pd.DataFrame({'x': x, 'y': y, 'target': intensity})], ignore_index=True)


	# test_data = list(range(test_size))
	# results = list(tqdm(pool.imap(generate_data, test_data), total=test_size))
	# df_test = pd.DataFrame(results)

	for i in tqdm(range(test_size)):
		env = Fire(world_size=world_size, episodes=episodes, fireAreas_Num=fireAreas_Num)
		# env_list.append(env)
		x, y, intensity, fire_cluster, time_stamp, fire_cluster_means = env.generate_fire_data()
		df_test = pd.concat([df_test, pd.DataFrame({'x': x, 'y': y, 'time_stamp': time_stamp, 'target': intensity})], ignore_index=True)

	# mu_init = get_gaussian_centers(df_train['y'][0], n_components=2)

	epochs = 1
	batch_size = 32
	steps_per_epoch = int((len(df_train)//batch_size)*0.9)
	data_config = DataConfig(
					target=['target'],
					continuous_cols=cont_cols,
					categorical_cols=cat_cols,
					num_workers=64
				)
	
	trainer_config = TrainerConfig(
						auto_lr_find=True, # Runs the LRFinder to automatically derive a learning rate
						batch_size=batch_size,
						max_epochs=epochs,
						early_stopping="valid_loss",
						early_stopping_patience=5,
						checkpoints="valid_loss",
						load_best=True
					)
	
	optimizer_config = OptimizerConfig(lr_scheduler="ReduceLROnPlateau", lr_scheduler_params={"patience":3})

	mdn_head_config = MixtureDensityHeadConfig(
						num_gaussian=3, 
						weight_regularization=2,
						lambda_mu=5,
						lambda_pi=2.0,
						lambda_sigma=5,
						# mu_bias_init=mu_init
					).__dict__


	backbone_config_class = "CategoryEmbeddingModelConfig"
	backbone_config = dict(
		task="backbone",
		layers="64",  # Number of nodes in each layer
		activation="ReLU",  # Activation between each layers
		head=None,
	)

	model_config = MDNConfig(
		task="regression",
		backbone_config_class=backbone_config_class,
		backbone_config_params=backbone_config,
		head_config=mdn_head_config,
		learning_rate=1e-3,
	)

	tabular_model = TabularModel(
		data_config=data_config,
		model_config=model_config,
		optimizer_config=optimizer_config,
		trainer_config=trainer_config,
	)

	# loaded_model = TabularModel.load_from_checkpoint("imgs/2h_mdn_model100")

	tabular_model.fit(train=df_train, validation=df_valid)
	# loaded_model.fit(train=df_train, validation=df_valid)

	tabular_model.save_model('imgs/lr4_3g_5f_100ws/mdn_model10')
	print('model saved')
	# pred_df = tabular_model.predict(df_test, quantiles=[0.25,0.5,0.75], n_samples=100, ret_logits=True)
	# # print(pred_df.keys())

	# # pi is the max of the two series for every row
	# # pred_df[['pi_0','pi_1']] = softmax(pred_df[['pi_0','pi_1']].values, axis=-1)
	# # print(pred_df['pi_0'], pred_df['pi_1'])
	# # print(pred_df['mu_0'], pred_df['mu_1'])
	# # print(pred_df['sigma_0'], pred_df['sigma_1'])
	# pi = np.where(pred_df['pi_0'] > pred_df['pi_1'], 0, 1)
	# # print(pi)
	
	# # mu corresponding to the max pi
	# mu = np.where(pi == pred_df['pi_0'], pred_df['mu_0'], pred_df['mu_1'])

	# # sigma corresponding to the max pi
	# sigma = np.where(pi == pred_df['pi_0'], pred_df['sigma_0'], pred_df['sigma_1'])

	# # plot
	# fig, ax = plt.subplots(1, 3, figsize=(45, 15))
	# x_test = pred_df['x']
	# y_test = pred_df['y']
	# intensity = pred_df['target']
	# fig.colorbar(ax[0].scatter(x_test, y_test, c=pi, cmap='jet', s=10))
	# fig.colorbar(ax[1].scatter(x_test, y_test, c=mu, cmap='jet', s=10))
	# fig.colorbar(ax[2].scatter(x_test, y_test, c=intensity, cmap='jet', s=10))

	# ax[0].set_title('confidence')
	# ax[1].set_title('predicted mean')
	# ax[2].set_title('ground truth intensity')

	# plt.savefig('imgs/int_in_x/testing_fire10.png')
	# print('plot saved')		
	# pool.close()
