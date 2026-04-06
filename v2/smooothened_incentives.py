# Fit a Gaussian Process over n_K_points and v_scr_K_iter to find the optimal K with more precision
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import numpy as np
import matplotlib.pyplot as plt

v_scr_K_iter = pd.read_csv("df_v_scr_K_iter_limit_False.csv", index_col=0).values
n_K_points = v_scr_K_iter.shape[1]
n_random_trials = v_scr_K_iter.shape[0]
upper_K = 200

X = np.linspace(1, upper_K, n_K_points).reshape(-1, 1)
y = np.mean(np.stack(v_scr_K_iter), axis = 0)

# Normalize X and y
X_input = (X - X.min()) / (X.max() - X.min())
y_input = (y - y.min()) / (y.max() - y.min())

kernel = RBF(length_scale=10.0) + WhiteKernel(noise_level=1)
gp = GaussianProcessRegressor(kernel=kernel).fit(X_input, y_input)
X_pred = np.linspace(1, upper_K, 100).reshape(-1, 1)
X_pred_input = (X_pred - X.min()) / (X.max() - X.min())
y_pred, sigma = gp.predict(X_pred_input, return_std=True)
y_pred = y_pred * (y.max() - y.min()) + y.min()  # Denormalize
opt_val_smooth = np.argmax(y_pred)

# Plot

plt.figure(figsize=(6, 5))
# plot each scr  in a differnte color
plt.plot(X_pred, y_pred, label="GP Fit")
plt.fill_between(X_pred.flatten(), y_pred - sigma, y_pred + sigma, alpha=0.5, label="GP Uncertainty")
# Add best K in plot as a line and in legend
opt_val_loc = np.argmax(y_pred)
opt_K = np.linspace(1, upper_K, 100)[opt_val_loc]
plt.axvline(x=opt_K, color='r', alpha = 0.7, linestyle='--', label=f"Best K = {opt_K:.2f}")
plt.tight_layout()

plt.legend(loc='upper right')
plt.xlabel("K")
plt.ylabel(r"$\phi(I)$")
plt.title(f"Marginalized Incentive Over The Entire Population")
plt.savefig(f"outputs/smoothed_overall_limit_False_best_K_ARA.png", bbox_inches='tight')
plt.close()