import numpy as np
import matplotlib.pyplot as plt
import torch
from gst_mdn import Fire, MDN
from tqdm import tqdm


# plot loss from file, each line is <iter loss>
def plot_loss(file_name, title):
	loss = []
	with open(file_name, 'r') as f:
		for line in f:
			if float(line) < 7:
				loss.append(float(line))
	plt.plot(loss)
	plt.title(title)
	plt.xlabel('iteration')
	plt.ylabel('loss')
	# plt.savefig(file_name[:-4] + '_bias.png')
	plt.show()
file_path = 'results/train/loss.txt'
# plot_loss(file_path, 'MDN loss')


# load model from resu;ts/train/mdn.pt
network = MDN(n_hidden=20, n_gaussians=5)
network.load_state_dict(torch.load('results/train/low_loss_mdn.pt'))




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

	fig, ax = plt.subplots(1, 3, figsize=(20, 10))
	ax[0].scatter(x, y, c=pi1, cmap='jet', s=10)
	# ax[1].scatter(x, y, c=pi2, cmap='jet', s=10)
	# plot colorbar
	fig.colorbar(ax[1].scatter(x, y, c=pi2, cmap='jet', s=10), ax=ax[1])
	# for j in range(len(fire_means)):
	# 	ax[2].scatter(fire_means[i][0], fire_means[i][1], marker='x', c='red', s=100)
	ax[2].scatter(x, y, c=fire_cluster, cmap='jet', s=10)
	ax[0].set_title('argmax prediction')
	ax[1].set_title('max prediction')
	ax[2].set_title('ground truth')
	plt.savefig(f'results/test/mdn_{i}.png')

