import numpy as np
import matplotlib.pyplot as plt
import torch
from gst_mdn import Fire, MDN
from tqdm import tqdm
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


# plot loss from file, each line is <iter loss>
def plot_loss(file_name, title):
	loss = []
	with open(file_name, 'r') as f:
		for line in f:
			# if float(line) < 7:
			loss.append(float(line))
	plt.plot(loss)
	plt.title(title)
	plt.xlabel('iteration')
	plt.ylabel('loss')
	# plt.savefig(file_name[:-4] + '_bias.png')
	plt.show()
file_path = 'results/train/1hidden_loss.txt'
# plot_loss(file_path, 'MDN loss')


# load model from resu;ts/train/mdn.pt
network = MDN(n_hidden=20, n_gaussians=5)
# network.load_state_dict(torch.load('results/train/1hidden_int_mdn_480.pt'))
network.load_state_dict(torch.load('results/train/1hidden_int_mdn_final.pt'))


# # test model
for i in tqdm(range(20)):
	env = Fire(world_size=250, episodes=20, fireAreas_Num=5)
	x, y, intensity, fire_cluster, fire_cluster_means = env.generate_fire_data()
	fire_means = []
	for cluster in fire_cluster_means:
		fire_means.append(np.mean(cluster, axis=0, dtype=np.int32))

	# print('cluster: ', fire_means)
	x_test = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), intensity.reshape(-1, 1)], axis=1)
	# x_test = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
	y_test = fire_cluster.reshape(-1, 1)
	# y_test = intensity.reshape(-1, 1)

	x_test = torch.tensor(x_test, dtype=torch.float32)
	y_test = torch.tensor(y_test, dtype=torch.float32, requires_grad=False)

	# network.mu_bias = torch.tensor(np.array(fire_means), dtype=torch.float32)
	pi, sigma, mu = network(x_test)
	# print('test loss: ', network.mdn_loss_fn(pi, sigma, mu, y_test).item())

	pi = pi.detach().numpy()
	sigma = sigma.detach().numpy()
	mu = mu.detach().numpy()

	pi1 = np.argmax(pi, axis=1)
	pi2 = np.max(pi, axis=1)

	mu1 = np.max(mu, axis=1)
	mu2 = np.argmax(mu, axis=1)

	# print(mu)

	sigma1 = sigma[np.indices(mu2.shape)[0], mu2]


	# # Assuming you have defined x, y, mu1, sigma1, intensity, and fire_means

	# fig, ax = plt.subplots(1, 3, figsize=(20, 10))

	# # Create a divider for the third subplot
	# divider = make_axes_locatable(ax[2])
	# cax = divider.append_axes('right', size='5%', pad=0.1)  # Adjust the size and pad as needed

	# # Plot the subplots
	# sc1 = ax[0].scatter(x, y, c=mu1, cmap='Oranges', s=10)
	# sc2 = ax[1].scatter(x, y, c=sigma1, cmap='jet', s=10)
	# sc3 = ax[2].scatter(x, y, c=intensity, cmap='Oranges', s=10)

	# # Create a common colorbar for the third subplot
	# cbar_common = fig.colorbar(sc3, cax=cax)

	# # Create a colorbar for the second subplot (variance)
	# divider2 = make_axes_locatable(ax[1])
	# cax2 = divider2.append_axes('right', size='5%', pad=0.1)  # Adjust the size and pad as needed
	# cbar2 = fig.colorbar(sc2, cax=cax2)

	# # Set titles for the subplots
	# ax[0].set_title('prediction')
	# ax[1].set_title('variance')
	# ax[2].set_title('ground truth intensity')

	# # Save the figure
	# plt.savefig(f'results/test/int_mdn/fire_mdn_{i}.png')




	# -----------------------
	fig, ax = plt.subplots(1, 3, figsize=(20, 10))
	# plot colorbar
	ax[0].scatter(x, y, c=pi2, cmap='jet', s=10)
	# fig.colorbar(ax[0].scatter(x, y, c=pi2, cmap='jet', s=10), ax=ax[0])
	fig.colorbar(ax[1].scatter(x, y, c=mu2, cmap='jet', s=10), ax=ax[1])
	# for j in range(len(fire_means)):
	# 	ax[2].scatter(fire_means[i][0], fire_means[i][1], marker='x', c='red', s=100)
	# fig.colorbar(ax[2].scatter(x, y, c=intensity, cmap=cm.Oranges, s=10), ax=ax[2])
	fig.colorbar(ax[2].scatter(x, y, c=fire_cluster, cmap='jet', s=10), ax=ax[2])
	ax[0].set_title('prediction')
	ax[1].set_title('variance')
	ax[2].set_title('ground truth intensity') 
	# plt.savefig(f'results/test/int_mdn/fire_mdn_{i}.png')
	plt.savefig(f'results/test/soft/mdn_{i}.png')


	# fig, ax = plt.subplots(1, 3, figsize=(20, 10))
	# ax[0].scatter(x, y, c=pi1, cmap='jet', s=10)
	# # ax[1].scatter(x, y, c=pi2, cmap='jet', s=10)
	# # plot colorbar
	# fig.colorbar(ax[1].scatter(x, y, c=pi2, cmap='jet', s=10), ax=ax[1])
	# # for j in range(len(fire_means)):
	# # 	ax[2].scatter(fire_means[i][0], fire_means[i][1], marker='x', c='red', s=100)
	# ax[2].scatter(x, y, c=fire_cluster, cmap='jet', s=10)
	# ax[0].set_title('argmax prediction')
	# ax[1].set_title('max prediction')
	# ax[2].set_title('ground truth')
	# plt.savefig(f'results/test/int_mdn/mdn_{i}.png')

