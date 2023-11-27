import numpy as np
import matplotlib.pyplot as plt
import os 

# fire_{i}.npy files in gp_fire_img/interp/data/
data = 'gp_fire_img/interp/data/'
files = os.listdir(data)

num_files = len(files)

# Load data
fire = []
for i in range(len(files)):
	fire.append(np.load(data + files[i]))
	
# each fire is 3D array of N x N x time	where N is field size

# Plot
t, h, w = fire[0].shape
print(h, w, t)

for i in range(num_files):
	
	for j in range(t):
		xx = []
		yy = []
		ii = []
		for m in range(fire[i][j].shape[0]):
			for n in range(fire[i][j].shape[1]):
				if fire[i][j][m][n] != 0:
					xx.append(m)
					yy.append(n)
					ii.append(fire[i][j][m][n])
		
		fig, ax = plt.subplots(1,1)
		fig.colorbar(
			ax.scatter(
				np.array(yy), np.array(xx), c=np.array(ii), cmap="viridis"
			),
			ax=ax,
			)
		ax.set_xlim(0, h)
		ax.set_ylim(0, w)

		plt.show()