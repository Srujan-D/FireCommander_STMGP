import numpy as np
import time
import random
from WildFire_Model import WildFire
from tqdm import tqdm

from scipy import stats

import torch

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
from torch.multiprocessing import Pool
import multiprocessing

import gpytorch
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.mixture import GaussianMixture
from GP_mixture import mix_GPs


# Function to create a 2D Gaussian map
def create_gaussian_map(
    x, y, intensity, world_size, n_inducing=10, n_iterations=100, learning_rate=0.1
):
    # print("creating gp")
    st = time.time()
    dtype = torch.float

    x_tensor = torch.tensor(x, dtype=dtype)
    y_tensor = torch.tensor(y, dtype=dtype)
    intensity_tensor = torch.tensor(intensity, dtype=dtype)

    data = torch.stack([x_tensor, y_tensor, intensity_tensor], dim=1)

    class GPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, inducing_points):
            super(GPModel, self).__init__(train_x, train_y, likelihood)

            batch_shape = torch.Size([train_x.shape[1]])
            # self.mean_module = gpytorch.means.ConstantMean()
            self.mean_module = gpytorch.means.ZeroMean()
            # self.mean_module = gpytorch.means.LinearMeanGradGrad(
            # 	input_size=train_x.shape[1], batch_shape=batch_shape
            # )
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    lengthscale=20 * torch.ones_like(train_x),
                    # batch_shape=batch_shape,
                ),
                outputscale=0.002 * torch.tensor(1.0),
                inducing_points=inducing_points,
                # batch_shape=batch_shape,
            )

        def forward(self, x):
            # x = x.T
            # print(x.shape)
            # x = x.reshape((1, -1))
            # print(x.shape)
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(data[:, :2], data[:, 2], likelihood, inducing_points=n_inducing)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
        ],
        lr=learning_rate,
    )

    for i in range(n_iterations):
        optimizer.zero_grad()
        output = model(data[:, :2])
        loss = -likelihood(output).log_prob(data[:, 2]).mean()
        # print("Mean shape:", output.mean.shape)
        # print("Covariance matrix shape:", output.covariance_matrix.shape)
        # loss = -likelihood(output).log_prob(data[:, 2].unsqueeze(-1).flatten()).mean()

        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    print("model created. now predicting: ", time.time() - st)
    st = time.time()

    x_grid, y_grid = torch.meshgrid(
        torch.arange(0, world_size), torch.arange(0, world_size)
    )
    test_x = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1).type(dtype)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(test_x))

    print("done with predicting at ", time.time() - st)
    gaussian_map = predictions.mean.view((world_size, world_size))

    return gaussian_map.numpy(), model


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
            num_firespots.append(25)
            area_wind_directions.append(random.choice([0, 45, 90, 135, 180]))
        self.fire_info = [
            area_centers,  # [[area1_center_x, area1_center_y], [area2_center_x, area2_center_y], ...],
            [
                num_firespots,  # [[num_firespots1, num_firespots2, ...],
                area_delays,  # [area1_start_delay, area2_start_delay, ...],
                area_fuel_coeffs,  # [area1_fuel_coefficient, area2_coefficient, ...],
                area_wind_speed,  # [area1_wind_speed, area2_wind_speed, ...],
                area_wind_directions,  # [area1_wind_direction, area2_wind_direction, ...],
                1.25,  # temporal penalty coefficient,
                0.1,  # fire propagation weight,
                90,  # Action Pruning Confidence Level (In percentage),
                80,  # Hybrid Pruning Confidence Level (In percentage),
                1,
            ],
        ]

        # self.gaussian_process = GaussianProcessRegressor(kernel=self.kernel_initial(), n_restarts_optimizer=10, normalize_y=False)

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
            hotspot_areas.append(
                [
                    self.fire_info[0][i][0] - 5,
                    self.fire_info[0][i][0] + 5,
                    self.fire_info[0][i][1] - 5,
                    self.fire_info[0][i][1] + 5,
                ]
            )

        # checking fire model setting mode and initializing the fire model
        if (
            self.fire_info[1][9] == 0
        ):  # when using "uniform" fire setting (all fire areas use the same parameters)
            # initial number of fire spots (ignition points) per hotspot area
            num_ign_points = self.fire_info[1][0]
            # fuel coefficient for vegetation type of the terrain (higher fuel_coeff:: more circular shape fire)
            fuel_coeff = self.fire_info[1][2]
            # average mid-flame wind velocity (higher values streches the fire more)
            wind_speed = self.fire_info[1][3]
            # wind azimuth
            wind_direction = (
                np.pi * 2 * self.fire_info[1][4] / 360
            )  # converting degree to radian

            # Init the wildfire model
            self.fire_mdl = WildFire(
                terrain_sizes=terrain_sizes,
                hotspot_areas=hotspot_areas,
                num_ign_points=num_ign_points,
                duration=self.duration,
                time_step=1,
                radiation_radius=10,
                weak_fire_threshold=5,
                flame_height=3,
                flame_angle=np.pi / 3,
            )
            print("all fires same..")
            self.ign_points_all = self.fire_mdl.hotspot_init()  # initializing hotspots
            self.fire_map = self.ign_points_all  # initializing fire-map
            self.previous_terrain_map = (
                self.ign_points_all.copy()
            )  # initializing the starting terrain map
            self.geo_phys_info = self.fire_mdl.geo_phys_info_init(
                max_fuel_coeff=fuel_coeff,
                avg_wind_speed=wind_speed,
                avg_wind_direction=wind_direction,
            )  # initialize geo-physical info
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
                wind_direction = (
                    np.pi * 2 * self.fire_info[1][4][i] / 360
                )  # converting degree to radian

                # init the wildfire model
                self.fire_mdl.append(
                    WildFire(
                        terrain_sizes=terrain_sizes,
                        hotspot_areas=[hotspot_areas[i]],
                        num_ign_points=num_ign_points,
                        duration=self.duration,
                        time_step=1,
                        radiation_radius=10,
                        weak_fire_threshold=5,
                        flame_height=3,
                        flame_angle=np.pi / 3,
                    )
                )
                self.ign_points_all.append(
                    self.fire_mdl[i].hotspot_init(cluster_num=i + 1)
                )  # initializing hotspots
                self.previous_terrain_map.append(
                    self.fire_mdl[i].hotspot_init(cluster_num=i + 1)
                )  # initializing the starting terrain map
                self.geo_phys_info.append(
                    self.fire_mdl[i].geo_phys_info_init(
                        max_fuel_coeff=fuel_coeff,
                        avg_wind_speed=wind_speed,
                        avg_wind_direction=wind_direction,
                    )
                )  # initialize geo-physical info
            # initializing the fire-map
            # self.fire_map = []
            for i in range(self.fireAreas_Num):
                for j in range(len(self.ign_points_all[i])):
                    self.fire_map.append(self.ign_points_all[i][j])
            self.fire_map = np.array(self.fire_map)
            self.fire_map_spec = self.ign_points_all

        # the lists to store the firespots in different state, coordinates only
        self.onFire_List = (
            []
        )  # the onFire_List, store the points currently on fire (sensed points included, pruned points excluded)

    # propagate fire one step forward according to the fire model
    def fire_propagation(self):
        self.pruned_List = []
        # checking fire model setting mode and initializing the fire model
        if (
            self.fire_info[1][9] == 0
        ):  # when using "uniform" fire setting (all fire areas use the same parameters)
            self.new_fire_front, current_geo_phys_info = self.fire_mdl.fire_propagation( # type: ignore
                self.world_size,
                ign_points_all=self.ign_points_all,
                geo_phys_info=self.geo_phys_info,
                previous_terrain_map=self.previous_terrain_map,
                pruned_List=self.pruned_List,
            )
            updated_terrain_map = self.previous_terrain_map
        else:  # when using "Specific" fire setting (each fire area uses its own parameters)
            updated_terrain_map = self.previous_terrain_map
            for i in range(self.fireAreas_Num):
                (
                    self.new_fire_front_temp[i],
                    self.current_geo_phys_info[i],
                ) = self.fire_mdl[i].fire_propagation( # type: ignore
                    self.world_size,
                    ign_points_all=self.ign_points_all[i],
                    geo_phys_info=self.geo_phys_info[i], # type: ignore
                    previous_terrain_map=self.previous_terrain_map[i],
                    pruned_List=self.pruned_List,
                )

            # update the new firefront list by combining all region-wise firefronts
            self.new_fire_front = []
            for i in range(self.fireAreas_Num):
                for j in range(len(self.new_fire_front_temp[i])):
                    self.new_fire_front.append(self.new_fire_front_temp[i][j])
            self.new_fire_front = np.array(self.new_fire_front)

        # update the region-wise fire map
        if self.fire_info[1][9] == 1:
            for i in range(self.fireAreas_Num):
                self.fire_map_spec[i] = np.concatenate(
                    [self.fire_map_spec[i], self.new_fire_front_temp[i]], axis=0
                )
        else:
            self.fire_map_spec = self.fire_map

        # updating the fire-map data for next step
        if self.new_fire_front.shape[0] > 0:
            self.fire_map = np.concatenate(
                [self.fire_map, self.new_fire_front], axis=0
            )  # raw fire map without fire decay

        # update the fire propagation information
        if self.fire_info[1][9] == 1:
            ign_points_all_temp = []
            for i in range(self.fireAreas_Num):
                if self.new_fire_front_temp[i].shape[0] > 0:
                    # fire map with fire decay
                    self.previous_terrain_map[i] = np.concatenate(
                        (updated_terrain_map[i], self.new_fire_front_temp[i]), axis=0
                    )
                ign_points_all_temp.append(self.new_fire_front_temp[i])
            self.ign_points_all = ign_points_all_temp
        else:
            if self.new_fire_front.shape[0] > 0:
                self.previous_terrain_map = np.concatenate(
                    (updated_terrain_map, self.new_fire_front)
                )  # fire map with fire decay
                self.ign_points_all = self.new_fire_front

    def kernel_initial(self, sf_initial=1.0, ell_initial=1.0, sn_initial=0.1):
        return sf_initial**2 * RBF(length_scale=ell_initial) + WhiteKernel(
            noise_level=sn_initial
        )

    def calc_fire_intensity_in_field(self, fire_map):
        """
        input: fire_map, shape = (n, 4), n: number of fire spots, 4: (x, y, intensity, cluster_id)
        output: field_intensity - fire_intensity at each location in field, shape: (world_size, world_size)
        """
        field_intensity = np.zeros((self.world_size, self.world_size))
        for i in range(fire_map.shape[0]):
            x, y, intensity, _ = fire_map[i]
            if field_intensity[int(x), int(y)] > 0:
                # field_intensity[int(x), int(y)] = np.mean(np.array([field_intensity[int(x), int(y)], intensity]))
                field_intensity[int(x), int(y)] = max(
                    field_intensity[int(x), int(y)], intensity
                )
            else:
                field_intensity[int(x), int(y)] = intensity

        not_on_fire = np.ones((self.world_size, self.world_size))
        not_on_fire[field_intensity > 0] = 0
        not_on_fire = np.argwhere(not_on_fire == 1)

        # get the points that are on fire
        on_fire = np.argwhere(field_intensity > 0)

        for i in range(not_on_fire.shape[0]):
            x, y = not_on_fire[i]

            # # calculate the distance between the point and all the points that are on fire
            # dist = np.linalg.norm(on_fire - np.array([x, y]), axis=1)
            # # calculate the intensity of the point by interpolation
            # field_intensity[x, y] = np.sum(field_intensity[on_fire[:, 0], on_fire[:, 1]] / dist)

            neighboring_points = []
            for j in range(on_fire.shape[0]):
                x_, y_ = on_fire[j]
                if np.linalg.norm(np.array([x, y]) - np.array([x_, y_])) <= 2:
                    neighboring_points.append(field_intensity[x_, y_])

            if len(neighboring_points) != 0:
                field_intensity[x, y] = np.mean(neighboring_points)
            else:
                field_intensity[x, y] = 0

        return field_intensity

    def generate_fire_data(self, key="gpytorch"):
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
            X.extend(self.fire_map[:, 0]) # type: ignore
            Y.extend(self.fire_map[:, 1]) # type: ignore
            intensity.extend(self.fire_map[:, 2]) # type: ignore
            fire_cluster_id.extend(self.fire_map[:, 3]) # type: ignore
            time_stamp.extend(np.ones(self.fire_map.shape[0]) * i) # type: ignore

            # start_time = time.time()
            # gaussian_map, model = create_gaussian_map(X, Y, intensity, world_size)
            # gp_time = time.time() - start_time
            # print("gaussian map created ", gp_time)

            if (i % 1) == 0:
                if key == "interp":
                    xx = []
                    yy = []
                    ii = []
                    field_intensity = self.calc_fire_intensity_in_field(self.fire_map)
                    for m in range(field_intensity.shape[0]):
                        for n in range(field_intensity.shape[1]):
                            if field_intensity[m][n] != 0:
                                xx.append(m)
                                yy.append(n)
                                ii.append(field_intensity[m][n])

                    fig, ax = plt.subplots(1, 2, figsize=(30, 15))
                    fig.colorbar(
                        ax[0].scatter(
                            np.array(Y), np.array(X), c=intensity, cmap="viridis"
                        ),
                        ax=ax[0],
                    )
                    fig.colorbar(
                        ax[1].scatter(
                            np.array(yy), np.array(xx), c=np.array(ii), cmap="viridis"
                        ),
                        ax=ax[1],
                    )

                    for a in ax:
                        a.set_xlim(0, self.world_size)
                        a.set_ylim(0, self.world_size)

                    plt.title("2D fire intensity map by interpolation")
                    plt.savefig(f"gp_fire_img/interp/episode_{i//1}.png")
                    plt.close()
                    print(f"saved episode {i//1} image")

                # fig, ax = plt.subplots(1, 2, figsize=(30, 15), sharey=True, gridspec_kw={'width_ratios': [1, 1]})
                # fig.colorbar(ax[0].scatter(np.array(Y), np.array(X), c=intensity, cmap='viridis'), ax=ax[0])
                # fig.colorbar(ax[1].imshow(gaussian_map, cmap='viridis', origin='lower', extent=(0, world_size, 0, world_size)), ax=ax[1])
                # ax[0].set_aspect('equal', adjustable='box')
                # ax[1].set_aspect('equal', adjustable='box')
                # # plt.imshow(gaussian_map, cmap='viridis', origin='lower', extent=(0, world_size, 0, world_size))
                # plt.title('2D Gaussian Map (GPyTorch)')
                # plt.savefig(f'gp_fire_img/episode_{i/10}.png')

                """
				using gpytorch
				"""
                # '''
                if key == "gpytorch":
                    start_time = time.time()
                    gaussian_map, model = create_gaussian_map(
                        X, Y, intensity, world_size
                    )
                    gp_time = time.time() - start_time
                    print("gaussian map created ", gp_time)

                    fig, ax = plt.subplots(1, 2, figsize=(30, 15))

                    scatter = ax[0].scatter(
                        np.array(Y), np.array(X), c=intensity, cmap="viridis", s=20
                    )
                    fig.colorbar(scatter, ax=ax[0])

                    for a in ax:
                        a.set_xlim(0, self.world_size)
                        a.set_ylim(0, self.world_size)

                    img = ax[1].imshow(
                        gaussian_map,
                        cmap="viridis",
                        origin="lower",
                        extent=(0, world_size, 0, world_size),
                    )
                    fig.colorbar(img, ax=ax[1])

                    plt.suptitle("2D Gaussian Map (GPyTorch)")
                    plt.savefig(f"gp_fire_img/gpytorch/episode_{i//10}.png")

                    print(f"episode {i} saved")
                # '''

                """
				using scipy GP instead of gpytorch
				"""
                # """
                if key == "scipyGP":
                    input_x = np.array([X, Y]).T
                    input_y = np.array(intensity)

                    # for single GP:
                    # self.gaussian_process = GaussianProcessRegressor(
                    #     kernel=self.kernel_initial(),
                    #     n_restarts_optimizer=10,
                    #     normalize_y=False,
                    # )
                    # self.gaussian_process.fit(input_x, input_y)

                    n = 3
                    GPs = []
                    for i in range(n):
                        gp = GaussianProcessRegressor(
                            kernel=self.kernel_initial(), n_restarts_optimizer=10
                        )
                        GPs.append(gp)

                        GPs[i].fit(input_x, input_y)

                    x_ = np.linspace(0, self.world_size, self.world_size)
                    y_ = np.linspace(0, self.world_size, self.world_size)
                    X_, Y_ = np.meshgrid(x_, y_)
                    X_test = np.vstack(np.dstack((X_, Y_)))

                    GP_mixture_model = mix_GPs(GPs)
                    μ_test, σ_test = GP_mixture_model(X_test)

                    n_test = (self.world_size, self.world_size)

                    μ_test_2D = μ_test.reshape(n_test)
                    σ_test_2D = σ_test.reshape(n_test)

                    # fig, ax = plt.subplots(1, 3, figsize=(45, 15))
                    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(12, 4))

                    scatter = ax[0].scatter(
                        np.array(Y), np.array(X), c=intensity, cmap="viridis", s=10
                    )
                    fig.colorbar(scatter, ax=ax[0])

                    for a in ax:
                        a.set_xlim(0, self.world_size)
                        a.set_ylim(0, self.world_size)

                    # mean, std = self.gaussian_process.predict(
                    # 	np.c_[X_.ravel(), Y_.ravel()], return_std=True
                    # )
                    # # print(mean, std)
                    # mean = mean.reshape(X_.shape)
                    # std = std.reshape(X_.shape)

                    # # plot the mean and variance
                    # ax[1].contourf(X_, Y_, mean)  # , 1, colors='green', linewidths=1)
                    # ax[2].contourf(X_, Y_, std)  # , 1, colors='green', linewidths=1)

                    ax[1].set_title("Posterior Mean\n$\mu_{2|1}$")
                    ax[1].contourf(X_, Y_, μ_test_2D)

                    ax[2].set_title("Posterior Variance\n$\sigma^2_{2|1}$")
                    ax[2].contourf(X_, Y_, σ_test_2D)

                    ax[0].set_title("Fire Contour")
                    ax[1].set_title("Fire Mean")
                    ax[2].set_title("Fire Variance")

                    plt.savefig(f"gp_fire_img/changed/episode_{i//10}.png")
                    print(f"saved at {i}")
                # """

                """
				using GP mixture from scipy
				"""
                # '''
                if key == "mixGP":
                    self.gp_mixture = GaussianMixture(
                        n_components=1, covariance_type="full", max_iter=1000
                    )

                    input_x = np.array([X, Y]).T
                    input_y = np.array(intensity)

                    self.gp_mixture.fit(input_x, input_y)

                    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

                    for a in ax:
                        a.set_xlim(0, self.world_size)
                        a.set_ylim(0, self.world_size)

                    scatter = ax[0].scatter(
                        np.array(X), np.array(Y), c=intensity, cmap="viridis", s=20
                    )
                    fig.colorbar(scatter, ax=ax[0])

                    x_ = np.linspace(0, self.world_size, 500)
                    y_ = np.linspace(0, self.world_size, 500)
                    X_, Y_ = np.meshgrid(x_, y_)

                    Z = np.exp(
                        self.gp_mixture.score_samples(np.c_[X_.ravel(), Y_.ravel()])
                    )
                    Z = Z.reshape(X_.shape)

                    # ax[1].contourf(X_, Y_, Z, cmap='viridis')
                    fig.colorbar(ax[1].contourf(X_, Y_, Z, cmap="viridis"), ax=ax[1])

                    means = self.gp_mixture.means_
                    ax[1].scatter(
                        means[:, 0], means[:, 1], marker="x", color="red", s=20
                    )

                    # for i, cov_matrix in enumerate(self.gp_mixture.covariances_):
                    # 	variance_contour = ax[2].contourf(X_, Y_, self.gp_mixture.weights_[i] * np.exp(-0.5 * self._mahalanobis(X_, means[i], cov_matrix)), cmap='Blues', alpha=0.3)

                    # fig.colorbar(variance_contour, ax=ax[2])
                    # ax[2].scatter(means[:, 0], means[:, 1], marker='x', color='red', s=20)

                    ax[0].set_title("Fire Contour")
                    ax[1].set_title("Density Estimate")
                    # ax[2].set_title('Variance')

                    plt.suptitle(f"2D Gaussian Mixture Model - Episode {i}")
                    plt.savefig(f"gp_fire_img/1gp/episode_{i}.png")
                    plt.close()
                    # plt.show()
                    print(f"Saved at episode {i}")

                    del self.gp_mixture
                # '''

        # return fire map: X,Y, Intensity, time, fire cluster number (index of fire_map)
        # return self.fire_map[:, 0], self.fire_map[:, 1], self.fire_map[:, 2], self.fire_map[:, 3], means
        X = np.array(X)
        Y = np.array(Y)
        intensity = np.array(intensity)
        fire_cluster_id = np.array(fire_cluster_id)
        time_stamp = np.array(time_stamp)

        return X, Y, intensity, fire_cluster_id, time_stamp, means

    def _mahalanobis(self, X, mean, cov_matrix):
        """Calculate the Mahalanobis distance."""
        diff = X - mean
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        mahalanobis = np.sum(diff @ inv_cov_matrix * diff, axis=1)
        return mahalanobis


if __name__ == "__main__":
    world_size = 100
    fireAreas_Num = 1
    episodes = 500

    env_list = []
    data_size = 1

    # pool = Pool(multiprocessing.cpu_count()-10)

    start_time = time.time()

    for i in tqdm(range(data_size)):
        # print("Creating env")
        env = Fire(
            world_size=world_size, episodes=episodes, fireAreas_Num=fireAreas_Num
        )
        X, Y, intensity, fire_cluster_id, time_stamp, means = env.generate_fire_data(
            key="interp"
        )
        # time_env = time.time() - start_time
        # print("env created ", time_env)
        # start_time = time.time()
        # gaussian_map = create_gaussian_map(X, Y, intensity, world_size)
        # gp_time = time.time() - start_time
        # print("gaussian map created ", gp_time)

        # fig, ax = plt.subplots(1, 2, figsize=(30, 15), sharey=True, gridspec_kw={'width_ratios': [1, 1]})
        # fig.colorbar(ax[0].scatter(Y, X, c=intensity, cmap='viridis'), ax=ax[0])
        # fig.colorbar(ax[1].imshow(gaussian_map, cmap='viridis', origin='lower', extent=(0, world_size, 0, world_size)), ax=ax[1])
        # ax[0].set_aspect('equal', adjustable='box')
        # ax[1].set_aspect('equal', adjustable='box')
        # # plt.imshow(gaussian_map, cmap='viridis', origin='lower', extent=(0, world_size, 0, world_size))
        # plt.title('2D Gaussian Map (GPyTorch)')
        # plt.show()
