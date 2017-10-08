import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_regression

# Set random seed (for reproducibility)
np.random.seed(1000)

nb_samples = 500
nb_features = 4

# Create our dataset
X, Y = make_regression(n_samples=nb_samples,
                       n_features=nb_features)

# Implement a Passive Aggressive Regression
C = 0.01
eps = 0.1
w = np.zeros((X.shape[1], 1))
errors = []

for i in range(X.shape[0]):
    xi = X[i].reshape((X.shape[1], 1))
    yi = np.dot(w.T, xi)
    
    loss = max(0, np.abs(yi - Y[i]) - eps)
    
    tau = loss / (np.power(np.linalg.norm(xi, ord=2), 2) + (1 / (2*C)))
    
    coeff = tau * np.sign(Y[i] - yi)
    errors.append(np.abs(Y[i] - yi)[0, 0])
    
    w += coeff * xi
    
# Show the error plot
fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(errors)
ax.set_xlabel('Time')
ax.set_ylabel('Error')
ax.set_title('Passive Aggressive Regression Absolute Error')
ax.grid()

plt.show()

# The quality of the regression (in particular, the length of the transient period when the error is high) can be controlled by picking better C and ε values. In particular, I suggest to check different range centers for C (100, 10, 1, 0.1, 0.01), in order to determine whether a higher aggressiveness is preferable.