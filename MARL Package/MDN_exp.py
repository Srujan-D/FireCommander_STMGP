import numpy as np
import matplotlib.pyplot as plt
import gpytorch
import torch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_spd_matrix

# Simulated spatiotemporally varying GP data
def generate_spatiotemporal_data(num_points, input_dim=3, noise=0.1):
    X = torch.tensor(np.random.rand(num_points, input_dim), dtype=torch.float32)
    true_means = torch.sin(X[:, 0] * 2 * np.pi) * torch.sin(X[:, 1] * 2 * np.pi) * torch.sin(X[:, 2] * 2 * np.pi)
    
    # Add noise
    true_means += noise * torch.randn(true_means.shape)
    
    return X, true_means

# Define the MDN + GP model for spatiotemporal data
class MDN_GP_Model_Spatiotemporal(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y):
        super(MDN_GP_Model_Spatiotemporal, self).__init__(train_x, train_y, gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Generate synthetic spatiotemporal data
num_points = 2000
X, y = generate_spatiotemporal_data(num_points)
train_x = X[:1500, :]
train_y = y[:1500]
test_x = X[1500:, :]

# Train MDN + GP model
model_mdn = MDN_GP_Model_Spatiotemporal(train_x, train_y)
model_mdn.likelihood.noise = 1e-3
model_mdn.train()
mll_mdn = gpytorch.mlls.ExactMarginalLogLikelihood(model_mdn.likelihood, model_mdn)
optimizer_mdn = torch.optim.Adam(model_mdn.parameters(), lr=0.1)

n_epochs = 1
for i in range(n_epochs):
    optimizer_mdn.zero_grad()
    output_mdn = model_mdn(train_x)
    loss_mdn = -mll_mdn(output_mdn, train_y)
    print(output_mdn, train_y.shape)
    loss_mdn.backward()
    optimizer_mdn.step()

# Initialize a Mixture of GPs model using EM
num_mixtures = 5
gmm = GaussianMixture(n_components=num_mixtures, covariance_type='full', random_state=0)
gmm.fit(train_x)
gp_models = []

# Create a GP model for each component
for i in range(num_mixtures):
    train_x_i = train_x[gmm.predict(train_x) == i]
    train_y_i = train_y[gmm.predict(train_x) == i]

    model_gp_i = MDN_GP_Model_Spatiotemporal(train_x_i, train_y_i)
    model_gp_i.likelihood.noise = 1e-3
    model_gp_i.train()
    mll_gp_i = gpytorch.mlls.ExactMarginalLogLikelihood(model_gp_i.likelihood, model_gp_i)
    optimizer_gp_i = torch.optim.Adam(model_gp_i.parameters(), lr=0.1)

    for epoch in range(n_epochs):
        optimizer_gp_i.zero_grad()
        output_gp_i = model_gp_i(train_x_i)
        loss_gp_i = -mll_gp_i(output_gp_i, train_y_i)

        loss_gp_i.backward()
        optimizer_gp_i.step()

    gp_models.append(model_gp_i)

# Test data
test_y = gmm.predict(test_x)
test_x = test_x.numpy()  # Convert to NumPy

# Convert test_x to PyTorch tensor
test_x = torch.tensor(test_x, dtype=torch.float32)

# Generate predictions for MDN + GP model
model_mdn.eval()
with torch.no_grad():
    predictions_mdn = model_mdn(test_x)

# Extract the means of MDN
mdn_means = predictions_mdn.mean.detach().numpy()

# Generate predictions for Mixture of GPs model
predictions_gp = np.zeros(test_y.shape)
for i in range(num_mixtures):
    test_indices_i = np.where(test_y == i)
    test_x_i = test_x[test_indices_i]

    model_gp_i = gp_models[i]
    model_gp_i.eval()
    with torch.no_grad():
        predictions_gp_i = model_gp_i(test_x_i)

    predictions_gp[test_indices_i] = predictions_gp_i.mean.detach().numpy()

# Calculate Mean Squared Error (MSE)
mse_mdn = mean_squared_error(test_y, mdn_means)
mse_gp = mean_squared_error(test_y, predictions_gp)

print(f"MSE for MDN + GP: {mse_mdn}")
print(f"MSE for Mixture of GPs: {mse_gp}")

# # Create a single figure with subplots
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# # Plot predictions for MDN + GP model
# sc1 = axes[0].scatter(test_x[:, 0], test_x[:, 1], c=mdn_means, cmap='viridis', marker='o', s=50)
# axes[0].set_title("MDN + GP Predictions")
# axes[0].set_xlabel('X-axis')
# axes[0].set_ylabel('Y-axis')

# # Plot predictions for Mixture of GPs model
# sc2 = axes[1].scatter(test_x[:, 0], test_x[:, 1], c=predictions_gp, cmap='viridis', marker='o', s=50)
# axes[1].set_title("Mixture of GPs Predictions")
# axes[1].set_xlabel('X-axis')
# axes[1].set_ylabel('Y-axis')

# # Add colorbars to both subplots
# cbar1 = fig.colorbar(sc1, ax=axes[0])
# cbar2 = fig.colorbar(sc2, ax=axes[1])

# # Show the plot
# plt.show()

# Calculate absolute differences between MDN + GP and Mixture of GPs predictions
differences = np.abs(mdn_means - predictions_gp)

# Create a single figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot predictions for MDN + GP model
sc1 = axes[0].scatter(test_x[:, 0], test_x[:, 1], c=mdn_means, cmap='viridis', marker='o', s=50)
axes[0].set_title("MDN + GP Predictions")
axes[0].set_xlabel('X-axis')
axes[0].set_ylabel('Y-axis')

# Plot predictions for Mixture of GPs model
sc2 = axes[1].scatter(test_x[:, 0], test_x[:, 1], c=predictions_gp, cmap='viridis', marker='o', s=50)
axes[1].set_title("Mixture of GPs Predictions")
axes[1].set_xlabel('X-axis')
axes[1].set_ylabel('Y-axis')

# Add colorbars to both subplots
cbar1 = fig.colorbar(sc1, ax=axes[0])
cbar2 = fig.colorbar(sc2, ax=axes[1])

# Highlight points with differences
highlighted_points = np.where(differences > 0.1)  # Adjust the threshold as needed
axes[0].scatter(test_x[highlighted_points, 0], test_x[highlighted_points, 1], facecolors='none', edgecolors='red', marker='o', s=60)
axes[1].scatter(test_x[highlighted_points, 0], test_x[highlighted_points, 1], facecolors='none', edgecolors='red', marker='o', s=60)

# Show the plot
plt.show()

