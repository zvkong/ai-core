
## Day 1 Summary

### Objective
Build the minimum working Python/PyTorch engineering setup, complete the first STAT-9340 optimization reading, verify conceptual understanding, and finish the LeetCode Two Sum algorithm practice.

### Materials Used
* **STAT-9340:** `Optimization_Jan31.pdf` (Intro, Section 1), `Homework_1_Sp23.pdf`
* **Engineering Setup:** `ai-core` repo, `torch-env` Conda environment, `uv pip`, Python package layout
* **Algorithm Practice:** LeetCode Problem 1 (Two Sum)

### Completed Tasks
* Created Git repo, Conda environment, and installed core packages.
* Completed readings and oral concept checks on optimization strategies.
* Implemented reproducibility utilities (`repro.py`, `test_repro.py`) and updated `Makefile`.
* Solved Two Sum using a one-pass hash map with $O(n)$ time and space complexity.
* Completed repository cleanup and made the daily commit.

### Core Concepts Learned
* **Loss Function:** Converts learning into an optimization problem: $$\min_\theta Q(\theta)$$
* **GD vs SGD:** The difference lies in the gradient estimator, not the parameters updated.
* **Positive Definite Matrix:** If $A$ is positive definite, the update direction is guaranteed to be a descent direction: $$g^\top d = -g^\top A g < 0$$
* **Optimizers:** AdaGrad uses squared gradients but can become too conservative. RMSProp uses an exponential moving average (EMA). Adam tracks first/second moments and applies bias correction.
* **Statistics:** Minimizing negative log-likelihood is equivalent to maximum likelihood estimation.
* **Engineering:** `repro.py` controls randomness and centralizes device selection. `__pycache__` is normal and should be git-ignored. 

### Deliverables
* **Files:** `src/utils/repro.py`, `tests/test_repro.py`, `Makefile`, `src/leetcode/day01_two_sum.py`, `docs/daily_log.md`, `.gitignore`
* **Commit:** `day01: init repo + reproducibility utilities (stat9340)`

---

## Day 2 Summary

### Objective
Transition from reverse-mode Automatic Differentiation (AD) theory to implementing a minimal scalar autograd engine by hand, verifying gradients with tests.

### Materials Used
* **Video:** Karpathy's `micrograd` introduction (scalar graphs, local derivatives, reverse-mode AD)
* **STAT-9340:** `Optimization_Jan31.pdf` (Section 2, Automatic Differentiation)
* **Algorithm Practice:** LeetCode Problem 217 (Contains Duplicate)

### Completed Tasks
* Studied the core concepts of reverse-mode AD and computation graphs.
* Implemented a minimal `Value` class storing data, gradients, parent nodes, and backward rules.
* Hand-coded local backward logic for standard operations (`+`, `*`, `-`, `**`, `tanh`, `exp`, `/`).
* Passed manual sanity checks and wrote finite-difference gradient tests.
* Built a Jupyter notebook demo showcasing `tanh` saturation behavior.
* Solved Contains Duplicate using a hash set.

### Core Concepts Learned
* **AD Theory:** AD is exact via the chain rule, not a finite-difference approximation. Reverse-mode AD is backpropagation.
* **Gradient Accumulation:** Gradients must be accumulated with `+=` because one node may influence the output through multiple paths.
* **Backward Execution:** Execution must follow reverse topological order, heavily relying on `out.grad` for local rules.

### Deliverables
* **Files:** `src/z2h/micrograd_pytorch/value.py`, `tests/test_micrograd.py`, `notebooks/day02_micrograd.ipynb`, `src/leetcode/day02_contains_duplicate.py`
* **Commit:** `day02: micrograd scalar autograd + tests`

---

## Day 3 Summary

### Objective
Compare evaluation traces, forward-mode AD, reverse-mode AD, and PyTorch autograd on a single concrete function to connect manual differentiation with software autodiff.

### Materials Used
* **Video:** Karpathy's `micrograd` (whole-graph backward, node reuse bugs, MLP abstraction)
* **STAT-9340:** `Homework_1_Sp23.pdf` (Problem 3, 2D Rosenbrock function)
* **Algorithm Practice:** LeetCode Problem 242 (Valid Anagram)

### Completed Tasks
* Clarified the necessity of gradient accumulation and the decomposition of complex operations like `tanh`.
* Built an evaluation trace for the Rosenbrock function: $$f(x_1, x_2)=100(x_2-x_1^2)^2+(1-x_1)^2$$ evaluated at $(0,0)$.
* Performed manual forward-mode and reverse-mode AD, confirming both yield $\frac{\partial y}{\partial x_1}=-2$ and $\frac{\partial y}{\partial x_2}=0$.
* Verified manual results against PyTorch autograd in a Jupyter notebook.
* Solved Valid Anagram using a single frequency dictionary.

### Core Concepts Learned
* **AD Consistency:** Forward-mode and reverse-mode can be cleanly compared on the same evaluation trace and will yield the same results.
* **Micrograd vs PyTorch:** PyTorch autograd is a larger-scale engineering implementation of the exact same mechanism used in `micrograd`.
* **Abstraction:** Moving from a scalar `Value` to an MLP adds structural abstraction (neurons, layers, models), not new calculus.

### Deliverables
* **Files:** `notebooks/day03_ad_traces.ipynb`, `src/leetcode/day03_valid_anagram.py`, `docs/daily_log.md`
* **Commit:** `day03: forward+reverse mode ad trace + verification`

---

## Day 4 Summary

### Objective
Implement a nonlinear model using pure SGD and SGD with momentum by hand, deriving gradients manually, and comparing their behavior on synthetic and real datasets.

### Materials Used
* **Video:** Karpathy's `makemore` (probabilistic models, NLL loss, smoothing)
* **STAT-9340:** `Homework_1_Sp23.pdf` (Problem 1a–1c), `HW1_Problem1_Data.csv`
* **Algorithm Practice:** LeetCode Problem 20 (Valid Parentheses)

### Completed Tasks
* Loaded dataset and built the code skeleton separating data, model definition, and loss computation.
* Manually derived gradients for the model: $$y_i=\theta_1(\sin(x_i)+\cos(\theta_2 x_i))+\epsilon_i$$
* Implemented mini-batch SGD and Momentum SGD in pure Python/NumPy.
* Verified optimizers on synthetic data, then ran on real data.
* Generated loss curves and data-fitting comparison plots.
* Solved Valid Parentheses using a stack dictionary.

### Core Concepts Learned
* **NLL:** Negative log-likelihood neatly turns probability assignments into optimization objectives.
* **Optimizer Dynamics:** Momentum helps early descent but can introduce severe oscillation if poorly tuned.
* **Fit Quality:** Optimizer choice heavily influenced convergence speed, but both methods reached nearly identical final fits under current settings.
* **Visualization:** Inputs must be sorted before plotting nonlinear fitted curves to avoid zig-zag lines.

### Deliverables
* **Files:** `src/stat9340/hw1_problem1.py`, `src/leetcode/day04_valid_parentheses.py`, `docs/daily_log.md`
* **Artifacts:** Day 4 plots.
* **Commit:** `day04: hw1 p1 sgd+momentum on real data`

---

## Day 5 Summary

### Objective
Extend the optimizer comparison by implementing Adam and Particle Swarm Optimization (PSO) from scratch, creating a unified performance comparison across four optimizers.

### Materials Used
* **Video:** Karpathy's `makemore` (one-hot encoding, vectorized loss)
* **STAT-9340:** PSO slides, `Homework_1_Sp23.pdf` (Problem 1d–1f)
* **Algorithm Practice:** LeetCode Problem 121 (Best Time to Buy and Sell Stock)

### Completed Tasks
* Studied PSO mechanics (inertia, cognitive, and social terms).
* Hand-coded Adam, fixing a bug requiring simultaneous parameter state updates before computing gradients.
* Hand-coded PSO, resolving neighborhood logic and adding boundary clipping.
* Ran a unified evaluation comparing SGD, Momentum, Adam, and PSO on the real regression dataset.
* Generated 4-way comparison plots (loss curves and fitted data).
* Solved Best Time to Buy and Sell Stock using a greedy one-pass approach.

### Core Concepts Learned
* **Adam Consistency:** Both parameter gradients must be computed from the exact same state before applying updates.
* **PSO Dynamics:** PSO searches directly in parameter space using particle dynamics, completely bypassing gradient calculations. A particle represents a parameter vector, not a data point.
* **Vectorized Implementations:** One-hot encoding effectively acts as row selection in a weight matrix. Vectorized loss mathematically mirrors batch loss.

### Deliverables
* **Files:** Optimizer comparison code, `artifacts/day05/hw1_p1_optimizer_compare.csv`, `src/leetcode/day05_best_time_buy_sell_stock.py`, `docs/daily_log.md`
* **Artifacts:** Comparison plots.
* **Commit:** `day05: hw1 p1 adam+pso + comparison harness`

---

## Day 6 Summary

### Objective
Rewrite HW1 Problem 2a–2c in PyTorch to construct 2D contour plots, gradient vectors, and optimizer trajectories on the Rosenbrock function.

### Materials Used
* **STAT-9340:** `Optimization_Jan31.pdf` (Section 1 & 2), `Homework_1_Sp23.pdf` (Problem 2a–2c)
* **Algorithm Practice:** LeetCode Problem 53 (Maximum Subarray)

### Completed Tasks
* Implemented the PyTorch Rosenbrock function: $$f(x_1, x_2)=100(x_2-x_1^2)^2+(1-x_1)^2$$
* Verified manual gradient formulas against autograd.
* Generated contour surfaces with mapped gradient vectors.
* Overlaid SGD, Momentum, and Adam trajectories starting from $(-0.1, 2.5)$.

### Core Concepts Learned
* **Rosenbrock Difficulty:** The challenge is a narrow, curved valley with shifting curvature, not a high volume of local minima.
* **Optimizer Trajectories:** SGD zig-zags wildly in narrow valleys, Momentum smooths out the travel path, and Adam enters the valley floor extremely fast.

### Deliverables
* **Files:** `notebooks/day06_rosenbrock_paths.ipynb`
* **Artifacts:** Contour + gradient plots, contour + trajectory plots, brief comparison notes.
* **Commit:** `day06: hw1 p2 rosenbrock + optimizer trajectories (pytorch)`

---

## Day 8 Summary

### Objective
Implement a 1-hidden-layer MLP regression model with manual backpropagation and an L2 penalty, and compare its performance against a multiple linear regression baseline.

### Materials Used
* **STAT-9340:** Shallow Neural Network notes
* **Homework:** `Homework_2.pdf` (Problem 1a)

### Completed Tasks
* Clarified the task is continuous regression on `quality`, not classification.
* Implemented the core manual backprop loop, incorporating output bias and an L2 penalty on non-bias weights.
* Fit a standard multiple linear regression baseline.
* Evaluated held-out test MSE for both models.
* Debugged extensive matrix shape, activation function, and minibatch indexing issues.

### Core Concepts Learned
* **Layer Mechanics:** A hidden unit acts as a linear predictor fed through a nonlinearity: $$z_j = f\left(\sum_i w_{ji}x_i\right)$$. 
* **Output Functions:** The output function is task-dependent (identity for regression, sigmoid for binary, softmax for multi-class).
* **Matrix Calculus:** Gradient shapes must perfectly mirror parameter matrix dimensions: $$\delta_h = (y_{\text{pred}}-y)\beta^\top \circ h \circ (1-h)$$ and $$g_W = X^\top \delta_h$$
* **Current Performance:** The linear baseline (0.5539 MSE) outperformed the manual MLP (0.7592 MSE) under current hyperparameter tuning.

### Deliverables
* **Files:** HW2 1a manual backprop model, baseline comparison script.

---

## Day 9 Summary

### Objective
Implement a 1-hidden-layer binary classifier for wine `type` using manual gradients, cross-entropy loss, and sigmoid activations without PyTorch autograd.

### Materials Used
* **Homework:** `Homework_2.pdf` (Problem 1b)
* **Algorithm Practice:** LeetCode Problem 35 (Search Insert Position)

### Completed Tasks
* Encoded wine `type` into binary labels and built the forward pass with sigmoid activations.
* Successfully derived and implemented the backward pass for hidden weights, hidden bias, output weights, and output bias.
* Fixed loss calculation logic to use elementwise mean Binary Cross-Entropy (BCE).
* Achieved strong validation results (Loss: 0.074, Accuracy: 0.98).
* Solved Search Insert Position using a standard `lower_bound` binary search pattern.

### Core Concepts Learned
* **Classification Mechanics:** BCE behaves exactly as Bernoulli negative log-likelihood. The network acts as a logistic regression layered on top of learned nonlinear features.
* **Metrics:** Accuracy measures absolute correctness; calibration measures how well predicted probabilities match empirical frequencies.
* **Bias Gradients:** Hidden layer biases possess non-zero gradients and must be systematically updated component-wise.

### Deliverables
* **Files:** `notebooks/day09_manual_ce_gradients.ipynb`
* **Artifacts:** Validation metrics, calibration placeholder, LeetCode 35 solution.

---

## Day 10 Summary

### Objective
Transition from manual derivations to a minimal, reusable PyTorch MLP training framework incorporating standard deep learning components.

### Materials Used
* **Framework:** PyTorch MLP Trainer Requirements

### Completed Tasks
* Narrowed project scope to avoid premature over-engineering (e.g., stripping out complex dataclasses and generic JSON parsers).
* Implemented a cohesive PyTorch script containing:
  * An `MLP` architecture class with `BatchNorm` and `Dropout`.
  * Data train/validation splitting.
  * A single-epoch training function utilizing `loss.backward()`.
  * A structured evaluation function.
  * A full training loop with CSV logging and early stopping based on validation loss.

### Core Concepts Learned
* **Framework Design:** A minimal, focused script is vastly more effective for early learning stages than a sprawling, hyper-abstracted training framework. 

### Deliverables
* **Files:** PyTorch trainer script.

---

## Day 11 Summary

### Objective
Study activation scaling, gradient flow, and BatchNorm, then implement a PyTorch multitask neural network handling simultaneous regression and classification.

### Materials Used
* **Video:** Karpathy's `makemore` Part 3
* **Homework:** `Homework_2.pdf` (Problem 2)

### Completed Tasks
* Finished video study on internal network dynamics.
* Implemented a multitask network with a shared feature extractor diverging into a regression head (`quality`) and a classification head (`type`).

### Core Concepts Learned
* **Architecture Routing:** Use `nn.Sequential` for simple stacks, but custom `nn.Module` classes are required for divergent multi-head architectures.
* **Logits vs Probabilities:** Use raw logits alongside `BCEWithLogitsLoss` for numerical stability. Keep the final `sigmoid` activation out of the model definition, applying it only during inference.
* **Loss Tracking:** Regression and classification losses must be logged separately, as their scales are fundamentally different.

### Deliverables
* **Files:** PyTorch multitask neural network model.

---

## Day 15 Summary

### Objective
Reproduce the WaveNet architecture, study CNN convolution theory, and implement manual $3 \times 3$ convolution filters using pure NumPy.

### Materials Used
* **Video:** Z2H WaveNet
* **STAT-9340:** CNN Section 1 (Convolutions), HW3 Problem 1a
* **Algorithm Practice:** LeetCode Pandas, DeepML (NumPy)

### Completed Tasks
* Successfully reproduced the toy WaveNet model.
* Studied core convolution theory.
* Implemented manual NumPy filters for MNIST images (shift-left, shift-up, and Laplacian edge-detection).
* Plotted side-by-side comparisons of original images and convolved outputs.
* Completed targeted Pandas and NumPy coding exercises.

### Core Concepts Learned
* **Filter Mechanics:** Convolutions systematically apply a localized filter over neighboring spatial regions. Padding explicitly preserves spatial dimensions.
* **Fixed Filters:** Before any deep learning occurs, fixed matrices can perform highly interpretable image transformations (e.g., Laplacian filters highlighting high-frequency edge structures).

### Deliverables
* **Files:** `notebooks/day15_z2h_wavenet.ipynb`, `notebooks/day15_mnist_manual_convs_numpy.ipynb`
* **Artifacts:** `artifacts/day15/mnist_manual_convs.png`
* **Commit:** `day15: wavenet toy + hw3 conv filters`

---

## Day 16 Summary

### Objective
Transition from manual filters to formal CNN layer formulations, focusing heavily on tensor shape reasoning, pooling, and parameter sharing.

### Materials Used
* **STAT-9340:** `CNNs_Feb23.pdf` (Section 2)
* **Algorithm Practice:** LeetCode Pandas (2880, 2881, 2882), DeepML (2, 3, 4)

### Completed Tasks
* Mastered standard convolution spatial formulas.
* Studied the operational flow of feature maps and filter depths.
* Executed data manipulation and matrix reshaping exercises across Pandas and DeepML.

### Core Concepts Learned
* **Dimensionality:** Filter depth strictly matches input channel depth. A single filter generates a single 2D feature map.
* **Pooling:** Pooling operations are applied independently to each individual feature map to reduce spatial resolution.
* **Parameter Sharing:** Applying identical filters across the entire spatial field drastically reduces model parameters but requires gradient accumulation during backpropagation.
* **Shape Formula:** The fundamental output size formula: $$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2P - K}{S} \right\rfloor + 1$$

### Deliverables
* **Files:** LeetCode and DeepML completed tasks.
* ---

## Day 17 Summary

---

## Day 17 Summary

### Objective
Complete the main CNN implementation and interpretation tasks for HW3 Problem 1 by training a PyTorch MNIST CNN classifier, inspecting learned convolutional behavior, analyzing layer activations and tensor dimensions, and strengthening matrix/grid reasoning through LeetCode and DeepML practice.

### Materials Used
* **STAT-9340:** CNN implementation and architecture slides
* **Homework:** HW3 Problem 1b–1e
* **Dataset:** MNIST
* **Algorithm Practice:** LeetCode 74, 48, 73, 200
* **DeepML Practice:** Confusion Matrix, Covariance Matrix, Feature Scaling, Matrix Multiplication

### Completed Tasks
* Finished the CNN slides focused on implementation and specific CNN architectures.
* Built and trained a PyTorch CNN baseline for MNIST 10-class classification.
* Reached the required accuracy target for HW3 Problem 1b.
* Extracted learned kernels from the first convolutional layer.
* Applied selected learned kernels to input images and inspected their visual effects.
* Passed example images through the trained CNN and recorded the output dimensions at each layer.
* Justified the observed tensor dimensions using convolution, padding, stride, pooling, and flattening rules.
* Visualized activated layers for example digits 0–9.
* Completed matrix/grid-oriented LeetCode practice:
  * 74. Search a 2D Matrix
  * 48. Rotate Image
  * 73. Set Matrix Zeroes
  * 200. Number of Islands
* Completed DeepML practice related to evaluation, preprocessing, and linear algebra:
  * Generate a Confusion Matrix for Binary Classification
  * Calculate Covariance Matrix
  * Feature Scaling Implementation
  * Matrix times Matrix

### Core Concepts Learned
* **CNN Tensor Shapes:** CNN implementation requires explicit shape tracking. Each convolution and pooling operation changes spatial dimensions according to kernel size, padding, stride, and pooling window size.
* **Feature Maps:** One convolutional filter produces one feature map. Multiple filters in the same layer produce multiple output channels.
* **Kernel Interpretation:** Learned first-layer filters often respond to local image structures such as strokes, edges, local contrast, and digit-specific spatial patterns.
* **Activation Inspection:** Layer activations show how the network transforms raw pixels into progressively more class-discriminative features.
* **Pooling:** Pooling reduces spatial resolution while preserving locally strong responses. This reduces computation and gives limited robustness to small spatial shifts.
* **Flattening:** Flattening converts spatial feature maps into a vector representation so that the classifier head can perform final class prediction.
* **Evaluation:** Accuracy gives an overall correctness rate, while a confusion matrix exposes class-specific errors.
* **Feature Scaling:** Standardization and min-max scaling are preprocessing tools that control the numerical scale of model inputs. Even when MNIST mainly uses pixel normalization, the same principle applies to stable gradient-based learning.
* **Covariance Matrix:** Covariance measures how features vary together. It is useful for understanding feature dependence and activation statistics.
* **Matrix Multiplication:** Matrix multiplication is repeated inner-product computation. Convolution can be interpreted as repeated local inner products between image patches and kernels.
* **2D Matrix Indexing:** Flattened indexing connects directly to image tensor reasoning. A 2D position can be recovered from a 1D index using integer division and modulo.
* **LeetCode 48 Rotate Image:** The transpose step should iterate with the inner loop starting from `range(i, n)` instead of `range(n)`. This processes the upper triangle including the diagonal and avoids swapping symmetric entries twice. After transposition, reversing each row completes the 90-degree clockwise rotation.
* **Grid Traversal:** Number of Islands reinforces DFS/BFS traversal over a 2D grid, which is conceptually close to connected-component reasoning in image analysis.
* **In-place Matrix Updates:** Set Matrix Zeroes emphasizes careful marker usage so row/column updates do not corrupt information needed later.

### Deliverables
* **CNN Work:** MNIST CNN baseline completed, model evaluation completed, kernel inspection completed, layer-dimension analysis completed, activation visualization completed.
* **LeetCode:** 74, 48, 73, and 200 completed.
* **DeepML:** Confusion Matrix, Covariance Matrix, Feature Scaling, and Matrix Multiplication completed.
* **Commit:** `day17: mnist cnn baseline and interpretability practice`

### Materials Used
* **STAT-9340:** CNN implementation and architecture slides
* **Homework:** HW3 Problem 1b–1e
* **Dataset:** MNIST
* **Algorithm Practice:** LeetCode 74, 48, 73, 200
* **DeepML Practice:** Confusion Matrix, Covariance Matrix, Feature Scaling, Matrix Multiplication

### Completed Tasks
* Finished the CNN slides focused on implementation and specific CNN architectures.
* Built and trained a PyTorch CNN baseline for MNIST 10-class classification.
* Reached the required accuracy target for HW3 Problem 1b.
* Extracted learned kernels from the first convolutional layer.
* Applied selected learned kernels to input images and inspected their visual effects.
* Passed example images through the trained CNN and recorded the output dimensions at each layer.
* Justified the observed tensor dimensions using convolution, padding, stride, pooling, and flattening rules.
* Visualized activated layers for example digits 0–9.
* Completed matrix/grid-oriented LeetCode practice:
  * 74. Search a 2D Matrix
  * 48. Rotate Image
  * 73. Set Matrix Zeroes
  * 200. Number of Islands
* Completed DeepML practice related to evaluation, preprocessing, and linear algebra:
  * Generate a Confusion Matrix for Binary Classification
  * Calculate Covariance Matrix
  * Feature Scaling Implementation
  * Matrix times Matrix

### Core Concepts Learned
* **CNN Tensor Shapes:** CNN implementation requires explicit shape tracking. Each convolution and pooling operation changes spatial dimensions according to kernel size, padding, stride, and pooling window size.
* **Feature Maps:** One convolutional filter produces one feature map. Multiple filters in the same layer produce multiple output channels.
* **Kernel Interpretation:** Learned first-layer filters often respond to local image structures such as strokes, edges, local contrast, and digit-specific spatial patterns.
* **Activation Inspection:** Layer activations show how the network transforms raw pixels into progressively more class-discriminative features.
* **Pooling:** Pooling reduces spatial resolution while preserving locally strong responses. This reduces computation and gives limited robustness to small spatial shifts.
* **Flattening:** Flattening converts spatial feature maps into a vector representation so that the classifier head can perform final class prediction.
* **Evaluation:** Accuracy gives an overall correctness rate, while a confusion matrix exposes class-specific errors.
* **Feature Scaling:** Standardization and min-max scaling are preprocessing tools that control the numerical scale of model inputs. Even when MNIST mainly uses pixel normalization, the same principle applies to stable gradient-based learning.
* **Covariance Matrix:** Covariance measures how features vary together. It is useful for understanding feature dependence and activation statistics.
* **Matrix Multiplication:** Matrix multiplication is repeated inner-product computation. Convolution can be interpreted as repeated local inner products between image patches and kernels.
* **2D Matrix Indexing:** Flattened indexing connects directly to image tensor reasoning. A 2D position can be recovered from a 1D index using integer division and modulo.
* **LeetCode 48 Rotate Image:** The transpose step should iterate with the inner loop starting from `range(i, n)` instead of `range(n)`. This processes the upper triangle including the diagonal and avoids swapping symmetric entries twice. After transposition, reversing each row completes the 90-degree clockwise rotation.
* **Grid Traversal:** Number of Islands reinforces DFS/BFS traversal over a 2D grid, which is conceptually close to connected-component reasoning in image analysis.
* **In-place Matrix Updates:** Set Matrix Zeroes emphasizes careful marker usage so row/column updates do not corrupt information needed later.

### Deliverables
* **CNN Work:** MNIST CNN baseline completed, model evaluation completed, kernel inspection completed, layer-dimension analysis completed, activation visualization completed.
* **LeetCode:** 74, 48, 73, and 200 completed.
* **DeepML:** Confusion Matrix, Covariance Matrix, Feature Scaling, and Matrix Multiplication completed.
* **Commit:** `day17: mnist cnn baseline and interpretability practice`
---

# Day 18 Summary — Deep Learning Intensive Plan

## Objective for Today

The main goal today was to finish the remaining STAT-9340 Homework 3 components and consolidate the stochastic neural model material, especially Gaussian Processes and Deep Gaussian Processes.

By the end of Day 18, Homework 3 was considered complete. The main completed work included the DGP extension from the stochastic neural model notes, the HW3 CNN/Bayesian-layer requirement in the current implementation path, and the final comparison/write-up needed for HW3.

A PyTorch-native implementation of Bayesian layers was not completed today. This is intentionally deferred as an optional future improvement and is not blocking HW3 completion.

---

## Learning Resources Used Today

**STAT-9340: Stochastic Neural Models**  
File: `StochasticNeuralModels_Mar17.pdf`

Main reading:
* Gaussian Process regression
* GP posterior prediction
* covariance functions
* hierarchical GP vs Deep GP
* DGP setup from the single-layer `sin(x)` example
* Bayesian Neural Network section as conceptual background

Main topics:
* latent process model
* Gaussian process as a distribution over functions
* posterior GP prediction by conditional Gaussian formulas
* covariance kernels and prediction uncertainty
* DGP as composition of GP mappings
* Bayesian neural networks as uncertainty-aware neural models

**STAT-9340 Homework 3**  
File: `Homework_3_Sp23.pdf`

Main reading:
* Problem 1a–1f: MNIST CNN, convolution filters, CNN fitting, kernel inspection, layer activations, Bayesian-layer uncertainty
* Problem 2: extend the single-layer `sin(x)` example from stochastic neural model slides 29–33 into a two-layer DGP and compare the fit

**Practice**
* LeetCode 11. Container With Most Water
* DeepML 2x2 Matrix Inverse / linear algebra practice

---

## What Was Completed Today

### 1. Completed the remaining Homework 3 work

Homework 3 is now complete.

Completed HW3 scope:
* Problem 1a: manual 3x3 convolution filters on MNIST
  * shift image left
  * shift image up
  * Laplacian edge filter
* Problem 1b: MNIST CNN classifier
  * trained CNN for 10-class digit classification
  * evaluated performance on the test set
* Problem 1c: first convolutional layer kernel extraction
  * extracted kernel weights
  * applied learned filters to input images
  * interpreted what the filters were detecting
* Problem 1d: layer output shape inspection
  * passed an image through the model
  * inspected activated layer outputs
  * justified tensor dimensions layer by layer
* Problem 1e: activation visualization
  * generated activation plots for example digits 0–9
  * compared how features evolve through the network
* Problem 1f: Bayesian-layer / uncertainty component
  * completed the HW3 uncertainty requirement in the current implementation path
  * generated repeated predictions / uncertainty-style outputs and interpreted the behavior
* Problem 2: DGP
  * extended the single-layer `sin(x)` GP example into a two-layer DGP comparison
  * compared the fit against the single-layer baseline
  * wrote up the answer to “How much did the fit change?”

### 2. Reviewed the Gaussian Process data model

The GP regression setup was reviewed as:

\[
z_i = Y_i + \epsilon_i,
\]

where \(Y_i\) is the latent process value and \(\epsilon_i\) is observation noise.

The latent process is modeled as:

\[
Y(x)=f(x),
\qquad
f(x)\sim GP(m(x), c(x,x')).
\]

Key interpretation:
* \(m(x)\) controls the prior mean function.
* \(c(x,x')\) controls covariance between any two input locations.
* Prediction at unobserved \(x_0\) is possible because the GP gives a joint Gaussian distribution over observed and unobserved function values.

### 3. Reviewed GP posterior prediction

The conditional Gaussian prediction formula was reviewed:

\[
Y(x_0)\mid z
\sim
Gau\left(
\mu_0+c_0^\top C_z^{-1}(z-\mu),
\;
c_{0,0}-c_0^\top C_z^{-1}c_0
\right).
\]

The posterior mean is:

\[
\hat{Y}(x_0)
=
\mu_0+c_0^\top C_z^{-1}(z-\mu).
\]

The posterior variance is:

\[
\sigma^2_{Y_0}
=
c_{0,0}-c_0^\top C_z^{-1}c_0.
\]

Main implementation point:
* avoid explicitly computing \(C_z^{-1}\)
* use a linear solve such as `np.linalg.solve(...)` or `torch.linalg.solve(...)`

### 4. Compared covariance functions

Reviewed the role of covariance functions in GP regression.

Main points:
* the covariance function must be positive semi-definite
* the covariance function controls smoothness and uncertainty
* nearby inputs are typically assumed to be more strongly dependent
* common choices include Gaussian/RBF, exponential, Matérn, and related kernels

Important distinction:
* RBF / Gaussian kernels tend to imply smoother functions
* exponential kernels allow rougher sample paths
* adding observation noise prevents exact interpolation and improves numerical stability

### 5. Completed the Deep Gaussian Process component

Completed the HW3 DGP task by extending the single-layer `sin(x)` GP example to a two-layer DGP.

Key conceptual distinction:
* A hierarchical GP can stack GP priors while still operating directly on the original input \(x\).
* A true DGP composes mappings, where the output representation from one GP layer becomes the input to the next layer.

The DGP composition can be summarized as:

\[
f(x)
=
f_L \circ f_{L-1} \circ \cdots \circ f_1(x).
\]

Main learning:
* DGPs are more flexible than ordinary single-layer GPs.
* They can learn more complex nonlinear transformations.
* They are also harder to fit because latent outputs become inputs to deeper GP layers.
* Variational methods and inducing points are commonly used to make DGP training computationally feasible.

### 6. Clarified Bayesian-layer status

The HW3 Bayesian-layer requirement is treated as complete in the current HW3 implementation path.

However, the following task is deferred:

* PyTorch-native Bayesian layer implementation for CNNs

This deferred task is optional and can be revisited later as a separate implementation exercise. It is not a blocker for moving forward to Day 19.

Possible future file:
* `notebooks/optional_torch_bayesian_cnn.ipynb`
* `src/models/bayesian_layers.py`

Possible future objective:
* implement BayesianLinear or BayesianConv2d in PyTorch
* learn \(\mu\) and \(\rho\) for weight distributions
* sample stochastic weights during forward passes
* compare class-probability histograms against the current HW3 uncertainty result

---

## Practice Completed Today

### LeetCode 11 — Container With Most Water

Link: https://leetcode.com/problems/container-with-most-water/

Solved using the two-pointer method.

Core idea:
* The area is determined by width times the shorter height.
* Since width always decreases as pointers move inward, the only possible way to improve area is to move the pointer at the shorter height.
* Moving the taller side cannot help because the shorter side remains the bottleneck.

Complexity:
\[
O(n)
\]
time and
\[
O(1)
\]
space.

### DeepML — 2x2 Matrix Inverse / Linear Algebra Practice

Reviewed the closed-form inverse of a \(2\times 2\) matrix:

\[
A=
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix},
\qquad
A^{-1}
=
\frac{1}{ad-bc}
\begin{pmatrix}
d & -b \\
-c & a
\end{pmatrix}.
\]

Key point:
* The inverse exists only when
  \[
  ad-bc\neq 0.
  \]
* This connects directly to GP implementation because covariance matrices must be numerically invertible or stabilized with jitter/noise.
* In real implementations, solving a linear system is preferred over explicitly computing a matrix inverse.

---

## Main Files Created or Updated Today

* `notebooks/day18_dgp_single_two_layer_tf.ipynb`
* `notebooks/day18_gp_posterior_from_scratch.ipynb`
* `reports/hw3_dgp_fit_comparison.md`
* `reports/day18_gp_to_dgp_notes.md`
* `artifacts/day18/gp_posterior_rbf.png`
* `artifacts/day18/gp_posterior_exponential.png`
* `artifacts/day18/gp_metrics.json`
* `src/leetcode/day18_container_with_most_water.py`
* `src/deepml/day18_inverse_2x2.py`
* `docs/daily_log.md`

If some file names differ locally, the important requirement is that the Day 18 folder contains:
* a runnable DGP / GP notebook
* a DGP comparison report
* saved plots or metrics
* practice solutions
* this daily log entry

---

## Core Concepts Learned

### GP vs Gaussian Distribution

A Gaussian distribution is a distribution over a finite-dimensional vector:

\[
Y\sim Gau(\mu,\Sigma).
\]

A Gaussian process is a distribution over functions:

\[
f\sim GP(m,c).
\]

In practice, the infinite-dimensional GP becomes finite-dimensional because we only compute over observed locations and prediction locations.

### GP Prediction

GP prediction is conditional Gaussian inference. Once the joint Gaussian distribution of observed values and prediction values is specified, the posterior mean and variance follow from standard multivariate normal conditioning.

### Kernel Controls Function Behavior

The kernel determines:
* smoothness
* local dependence
* extrapolation behavior
* posterior uncertainty
* numerical conditioning

### DGP as Composition

A DGP is not just a stack of GPs with the same input. The crucial idea is that one layer transforms the representation used by the next layer.

A simplified view:

\[
x
\rightarrow
f_1(x)
\rightarrow
f_2(f_1(x))
\rightarrow
y.
\]

This makes DGPs more expressive than ordinary GPs but also harder to train.

### Bayesian Layers

Bayesian layers replace point-estimated weights with distributions over weights. Instead of learning only a weight value, the model learns parameters of a weight distribution, often a mean and variance-related parameter.

This enables repeated stochastic forward passes and uncertainty summaries, such as histograms of predicted class probabilities.

---

## Current Status

The Day 18 objective is complete.

Completed:
* HW3 Problem 1a–1f
* HW3 Problem 2
* GP regression review
* DGP comparison
* Day18 practice tasks
* daily log
* commit-ready artifacts

Deferred optional improvement:
* PyTorch-native Bayesian layer implementation for CNN uncertainty modeling

This deferred task should not delay Day 19.

---

## Recommended Commit Message

**`day18: complete hw3 dgp and stochastic neural models`**