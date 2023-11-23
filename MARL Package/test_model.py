import numpy as np
import matplotlib.pyplot as plt
import torch
from temp_mdn import Fire
from tqdm import tqdm
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pytorch_tabular import TabularModel
import pandas as pd

loaded_model = TabularModel.load_from_checkpoint("imgs/lr4_3g_5f_100ws/mdn_model10")
test_size = 1

world_size = 100
fireAreas_Num = 20
num_gaussians = 4
episodes = 20

from scipy.special import softmax

for i in tqdm(range(test_size)):
	df_test = pd.DataFrame(columns=['x', 'y', 'time_stamp', 'target'])
	env = Fire(world_size=world_size, episodes=episodes, fireAreas_Num=fireAreas_Num)
	# env_list.append(env)
	x, y, intensity, fire_cluster_id, time_stamp, fire_cluster_means = env.generate_fire_data()
	df_test = pd.concat([df_test, pd.DataFrame({'x': x, 'y': y, 'time_stamp': time_stamp, 'target': intensity})], ignore_index=True)
	
	pred_df = loaded_model.predict(df_test, quantiles=[0.25,0.5,0.75], n_samples=100, ret_logits=True)
	
	# print(pred_df.keys())
	
	# print(sum_of_pis)
	# print(np.sum(sum_of_pis))

	# softmax pi
	# pred_df[['pi_0','pi_1', 'pi_2']] = softmax(pred_df[['pi_0','pi_1', 'pi_2']].values, axis=-1)
	pred_df[[f'pi_{i}' for i in range(num_gaussians)]] = softmax(pred_df[[f'pi_{i}' for i in range(num_gaussians)]].values, axis=-1)
	# pi = np.argmax(pred_df[[f'pi_{i}' for i in range(num_gaussians)]].values, axis=-1)

	# pi = np.where(pred_df['pi_0'] > pred_df['pi_1'], 0, 1)
	# print(pi)
	pi = np.argmax(pred_df[['pi_0','pi_1', 'pi_2']].values, axis=-1)
	print(pred_df[['pi_0','pi_1', 'pi_2']])

	# plot the variation of 5 pi in one plot
	fig, ax = plt.subplots(1, 1, figsize=(15, 15))
	x_test = pred_df['x']
	y_test = pred_df['y']
	intensity = pred_df['target']
	fig.colorbar(ax.scatter(x_test, y_test, c=pi, cmap='jet', s=10))
	plt.show()

	# mu corresponding to the max pi among the num_gaussians pi
	mu = np.where(pi == 0, pred_df['mu_0'], np.where(pi == 1, pred_df['mu_1'], pred_df['mu_2']))

	# sigma corresponding to the max pi among the num_gaussians pi
	sigma = np.where(pi == 0, pred_df['sigma_0'], np.where(pi == 1, pred_df['sigma_1'], pred_df['sigma_2']))
		
	

	# # # mu corresponding to the max of the 5 pi
	# mu = np.where(pi == 0, pred_df['mu_0'], np.where(pi == 1, pred_df['mu_1'], np.where(pi == 2, pred_df['mu_2'], np.where(pi == 3, pred_df['mu_3'], pred_df['mu_4']))))

	# # # sigma corresponding to the max of the 5 pi
	# sigma = np.where(pi == 0, pred_df['sigma_0'], np.where(pi == 1, pred_df['sigma_1'], np.where(pi == 2, pred_df['sigma_2'], np.where(pi == 3, pred_df['sigma_3'], pred_df['sigma_4']))))
	
	
	# mu corresponding to the max pi
	# mu = np.where(pi == pred_df['pi_0'], pred_df['mu_0'], pred_df['mu_1'])

	# # sigma corresponding to the max pi
	# sigma = np.where(pi == pred_df['pi_0'], pred_df['sigma_0'], pred_df['sigma_1'])

	# # plot
	fig, ax = plt.subplots(1, 3, figsize=(45, 15))
	x_test = pred_df['x']
	y_test = pred_df['y']
	intensity = pred_df['target']
	fig.colorbar(ax[0].scatter(x_test, y_test, c=pi, cmap='jet', s=10))
	fig.colorbar(ax[1].scatter(x_test, y_test, c=mu, cmap='jet', s=10))
	fig.colorbar(ax[2].scatter(x_test, y_test, c=intensity, cmap='jet', s=10))

	ax[0].set_title('confidence')
	ax[1].set_title('predicted mean')
	ax[2].set_title('ground truth intensity')

	plt.savefig(f'imgs/lr4_3g_5f_100ws/1h_testing_fire250_5_{i}.png')
	plt.close()