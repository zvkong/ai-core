import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
n_samples = 100
## Define generate data function
def generate_data(n_samples = 100):
    np.random.seed(42)
    X = np.random.rand(n_samples, 1)
    noise = np.random.rand(n_samples, 1)*0.5
    y = 3 * X + 2 + noise
    return X, y

## SGD
## MSE loss
def mse_loss(X_batch, y_batch, w, b):
    m = X_batch.shape[0]
    y_pred = w * X_batch + b 
    loss = np.sum((y_pred - y_batch)**2)/m
    return loss

## Gradiant compute
def gradiants(X_batch, y_batch, w, b):
    m = X_batch.shape[0]
    y_pred =  w * X_batch + b
    error = y_pred - y_batch
    dw = 2/m * np.sum(X_batch * error)
    db = 2/m * np.sum(error)
    return dw, db

## SDG
def sdg(X, y, lr = 0.1, batch_size = 10, max_epochs = 20):
    n_samples = X.shape[0]

    w = np.random.randn()
    b = np.random.randn()
    loss_history = []

    for epoch in range(max_epochs):
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start_idx in range(0, n_samples, batch_size):
            end_idx = start_idx + batch_size
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            dw, db = gradiants(X_batch, y_batch, w, b)

            w -= lr*dw
            b -= lr*db

        epoch_loss = mse_loss(X, y, w, b)
        loss_history.append(epoch_loss)  

        print(f"Epoch {epoch+1}/{max_epochs}, Loss={epoch_loss:.4f}, w={w:.3f}, b={b:.3f}")

    return w, b, loss_history

if __name__ == "__main__":
    ## generate data
    X, y = generate_data(n_samples=1000)

    # Hyper parameters 
    lr = 0.01
    batch_size = 8
    epochs = 30

    # 
    w_final, b_final, losses = sdg(X, y, 
                                                     lr=lr, 
                                                     batch_size=batch_size, 
                                                     max_epochs=epochs)

    print("\nEnd:")
    print(f"  Learned w = {w_final:.3f}")
    print(f"  Learned b = {b_final:.3f}")


## Plot the loss history
plt.figure(figsize=(6,4))
plt.plot(losses, marker='o', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()

## SGD wt Momentum
def sdg_m(X, y, lr = 0.1, alpha = 0.1, batch_size = 10, max_epochs = 20):
    n_samples = X.shape[0]

    w = np.random.randn()
    b = np.random.randn()
    vw = np.random.randn()
    vb = np.random.randn()
    loss_history = []

    for epoch in range(max_epochs):
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start_idx in range(0, n_samples, batch_size):
            end_idx = start_idx + batch_size
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            dw, db = gradiants(X_batch, y_batch, w, b)
            vw = alpha * vw - lr*dw
            vb = alpha * vb - lr*db

            w += vw
            b += vb

        epoch_loss = mse_loss(X, y, w, b)
        loss_history.append(epoch_loss)  

        print(f"Epoch {epoch+1}/{max_epochs}, Loss={epoch_loss:.4f}, w={w:.3f}, b={b:.3f}")

    return w, b, loss_history

if __name__ == "__main__":
    ## generate data
    X, y = generate_data(n_samples=1000)

    # Hyper parameters 
    lr = 0.01
    alpha = 0.1
    batch_size = 8
    epochs = 30

    # 
    w_final_m, b_final_m, losses_m = sdg_m(X, y, 
                                                     lr=lr, 
                                                     alpha = alpha,
                                                     batch_size=batch_size, 
                                                     max_epochs=epochs)

    print("\nEnd:")
    print(f"  Learned w = {w_final:.3f}")
    print(f"  Learned b = {b_final:.3f}")
losses_m
w_final_m
b_final_m
plt.plot(losses_m, marker='o', label='Training Loss')
plt.show()


## Evaluating 
def f(x):
    return (1/2)*x**2

def finDiff(f, x, h, trueValue):
    return abs((f(x + h) - f(x))/h - trueValue)

err = [finDiff(f, x = 3, h = 10**(-exp), trueValue=3)\
        for exp in range(17)]
plt.plot(range(17), err, marker='o')
plt.yscale('log')
plt.show()

import sympy as sym