import pygame
from pygame.locals import *
import numpy as np
import random
import matplotlib.pyplot as plt
from WildFire_Model import WildFire
from FireCommander_Cmplx2_Utilities import EnvUtilities

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from GP_mixture import mix_GPs

from agent import Agent
from world import World

Agent_Util = EnvUtilities()

class FireCommander(object):
	def __init__(self, world_size=None, duration=None, fireAreas_Num=None, agent_num=None, online_vis=False):
		# pars parameters
		self.world_size = 100 if world_size is None else world_size           	# world size
		self.duration = 200 if duration is None else duration                	# number of steps per game
		self.fireAreas_Num = 5 if fireAreas_Num is None else fireAreas_Num		# number of fire areas
		self.agent_num = 3 if agent_num is None else agent_num  				# number of agents
		self.agent_list = []
		self.world = None

		# fire model parameters
		areas_x = np.random.randint(20, self.world_size - 20, self.fireAreas_Num)
		areas_y = np.random.randint(20, self.world_size - 20, self.fireAreas_Num)
		area_delays = [0] * self.fireAreas_Num
		area_fuel_coeffs = [5] * self.fireAreas_Num
		area_wind_speed = [5] * self.fireAreas_Num
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
						   1]]                     # mode]

		# the number of stacked frames for training
		self.stack_num = 4

		# initialize the pygame for online visualization
		if online_vis:
			pygame.init()
			# The simulation time is counted in seconds, while the actual time is counted in milliseconds
			clock = pygame.time.Clock()
			# Create a screen (Width * Height) = (1024 * 1024)
			self.screen = pygame.display.set_mode((self.world_size, self.world_size), 0, 32)

	# initialize the environment
	def env_init(self, comm_range=30, init_alt=10):
		for i in range(self.agent_num):
			# initialize the agent
			agent = Agent(id=i, X=int(self.world_size-10), Y=int(self.world_size-10), init_alt=init_alt)
			self.agent_list.append(agent)

		# initialize the world
		self.world = World(world_size=self.world_size, agent_list=self.agent_list, fire_list=self.fire_info)

		# initialize the communication range
		self.comm_range = comm_range

		
	
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
				self.ign_points_all.append(self.fire_mdl[i].hotspot_init())        # initializing hotspots
				self.previous_terrain_map.append(self.fire_mdl[i].hotspot_init())  # initializing the starting terrain map
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
		self.sensed_List = []  # the sensed_List, store the points currently on fire and have been sensed by agents
		self.pruned_List = []  # the pruned_List, store the pruned fire spots

		# keeping track of agents' contributions (e.g. number of sensed/pruned firespot by each Perception/Action agent)
		self.sensed_contribution = [0] * self.perception_agent_num
		self.pruned_contribution = [0] * self.action_agent_num