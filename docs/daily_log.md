
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