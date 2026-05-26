## Stochastic Gradiant Decent
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)
n_samples = 100
X = 2 * np.random.rand(n_samples, 1) ## generate a random num [0,1]
y = 2.0 * X + 5.0 + np.random.randn(n_samples, 1) * 0.5 ## relationship plus error term
X.shape[0]
X
## define a function 
def sgd (X, y, lr = 0.01, batch_size = 10, n_epochs = 20):

