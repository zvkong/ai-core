# Day 1 Summary — Deep Learning Intensive Plan

## Date
Day 1

## Overall Goal of Day 1
Build the minimum working Python/PyTorch engineering setup for the project, complete the first optimization reading from STAT-9340, verify conceptual understanding through oral questioning, and finish the first algorithm practice task.

---

## Materials Used

### 1. STAT-9340 course materials
- `Optimization_Jan31.pdf`
  - Completed:
    - Introduction
    - Section 1: **Optimization Strategies**

- `Homework_1_Sp23.pdf`
  - Read completely
  - Extracted scope of:
    - Problem 1
    - Problem 2
    - Problem 3

### 2. Engineering setup materials
- Local repo structure for `ai-core`
- Conda environment `torch-env`
- `uv pip` package installation workflow
- Minimal Python package layout:
  - `src/`
  - `src/utils/`
  - `tests/`
  - `docs/`
  - `Makefile`

### 3. Algorithm practice
- LeetCode Problem 1: **Two Sum**
  - Python version
  - One-pass hash map solution
  - Complexity target:
    - Time: `O(n)`
    - Space: `O(n)`

---

## What Was Completed

### 1. Project and environment setup
Completed:
- Created the Git repo named `ai-core`
- Created the Conda environment `torch-env`
- Installed core packages with `uv pip`
  - `torch`
  - `numpy`
  - `pandas`
- Created the required project folders and files

### 2. Reading and learning
Completed:
- Read `Optimization_Jan31.pdf` introduction and Section 1
- Read `Homework_1_Sp23.pdf`
- Completed oral review and concept check on:
  - loss functions
  - gradient descent vs SGD
  - positive definite scaling matrix
  - Momentum
  - AdaGrad
  - RMSProp
  - Adam
  - bias correction
  - negative log-likelihood and MLE connection

### 3. Engineering implementation
Completed:
- Implemented `src/utils/repro.py`
- Implemented `tests/test_repro.py`
- Updated `Makefile`
- Successfully ran:
  ```bash
  make test
  ```

### 4. Algorithm practice

Completed:

* Implemented Python solution for **Two Sum**
* Understood the one-pass hash map logic
* Clarified the difference between the LeetCode C++ template and the Python solution template

### 5. Project cleanup

Completed:

* Updated `.gitignore`
* Wrote `docs/daily_log.md`
* Made the Day 1 commit with the required message:

  ```text
  day01: init repo + reproducibility utilities (stat9340)
  ```

---

## Files Completed on Day 1

### Core utility

* `src/utils/repro.py`

### Tests

* `tests/test_repro.py`

### Build/test entry

* `Makefile`

### Algorithm practice

* `src/leetcode/day01_two_sum.py`

### Log and housekeeping

* `docs/daily_log.md`
* `.gitignore`

---

## Core Concepts Learned

### 1. Role of the loss function

The loss function is not only a measure of prediction error. It converts the learning problem into an optimization problem:
[
\min_\theta Q(\theta)
]

### 2. GD vs SGD

The difference is in the **gradient estimator**:

* GD uses the full dataset gradient
* SGD uses a stochastic estimate from one sample or a minibatch

This is **not** a difference in which parameters get updated. Both update the full parameter vector.

### 3. Why positive definite (A) matters

If the update direction is
[
d = -A g
]
and (A) is positive definite, then
[
g^\top d = -g^\top A g < 0
]
for any nonzero gradient (g), so the direction is a descent direction.

### 4. Why AdaGrad uses squared gradients

Squared gradients provide a coordinate-wise scale estimate. They are closely related to the diagonal of the gradient outer product:
[
g_t \odot g_t \equiv \mathrm{diag}(g_t g_t^\top)
]
This gives each parameter its own effective learning rate.

### 5. Why AdaGrad can become too conservative

AdaGrad accumulates squared gradients:
[
r_t = \sum_{\tau=1}^t g_\tau^2
]
As (r_t) keeps growing, the effective learning rate
[
\frac{\eta}{\sqrt{r_t} + \epsilon}
]
shrinks toward zero.

### 6. Why RMSProp helps

RMSProp replaces the full accumulation with an exponential moving average:
[
r_t = \rho r_{t-1} + (1-\rho) g_t^2
]
This prevents the effective learning rate from decaying too aggressively.

### 7. Adam moments

Adam tracks:

* first moment: EMA of gradients
* second moment: EMA of squared gradients

### 8. Bias correction in Adam

Because the moving averages start at zero, the early estimates are biased toward zero. Adam corrects them using:
[
\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad
\hat v_t = \frac{v_t}{1-\beta_2^t}
]

### 9. Statistical meaning of negative log-likelihood training

If the loss is negative log-likelihood, then minimizing the loss is equivalent to maximum likelihood estimation:
[
L(\theta) = -\sum_{i=1}^n \log p(y_i \mid x_i, \theta)
]

---

## Engineering Concepts Learned

### 1. What `repro.py` is for

`repro.py` is a small reproducibility utility module. Its purpose is to:

* control randomness
* centralize device selection
* make run configuration explicit
* improve debugging and fair experiment comparison

### 2. Main parts of `repro.py`

* `set_seed(...)`
* `get_device()`
* `report_config(...)`

### 3. Seed vs deterministic

* `seed` controls where randomness starts
* `deterministic` controls whether computation behavior is repeatable under the same setup

### 4. Why `set_seed()` usually returns nothing

`set_seed()` is a configuration function with side effects. It usually returns `None` and does not print anything. Its correctness is checked by its effect on random number generation.

### 5. Why `Makefile` matters

`Makefile` gives the project a unified command entry point.
For Day 1, the key target was:

```makefile
.PHONY: test

test:
  PYTHONPATH=. pytest -v tests/
```

This allowed testing via:

```bash
make test
```

### 6. Why `__pycache__` appears

`__pycache__` is a Python bytecode cache directory created automatically when modules are imported. It is normal, should not be edited manually, and should be ignored in Git.

Suggested `.gitignore` entries:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
```

---

## Debugging and Workflow Issues Solved

### 1. `pytest` collected 0 items

Cause:

* tests were not being discovered correctly

Resolution:

* ensured file name matched `test_*.py`
* ensured test functions started with `test_`
* saved the file correctly under `tests/`

### 2. `ModuleNotFoundError: No module named 'src'`

Cause:

* Python could not find the `src` package during test execution

Resolution:

* added empty `__init__.py` files:

  * `src/__init__.py`
  * `src/utils/__init__.py`
* updated the `Makefile` to include:

  ```makefile
  PYTHONPATH=. pytest -v tests/
  ```

### 3. Clarification on `Makefile`

Mistake:

* tried running:

  ```bash
  make tests/test_repro.py
  ```

Correct command:

```bash
make test
```

### 4. Clarification on LeetCode templates

* The displayed class template using `vector<int>` was **C++**
* Day 1 implementation used **Python**, not C++

---

## Final Review Questions and Correct Answers

### Q1. What is the difference between gradient descent and stochastic gradient descent at the level of gradient estimation?

**Answer:**
GD uses the full dataset to compute the gradient of the objective, while SGD uses a stochastic estimate based on one sample or a minibatch. The key difference is the gradient estimator, not which parameters are updated.

### Q2. Why does a positive definite matrix (A) guarantee that (-Ag) is a descent direction?

**Answer:**
If (A) is positive definite, then for any nonzero gradient (g),
[
g^\top A g > 0
]
If the update direction is
[
d = -Ag
]
then
[
g^\top d = -g^\top A g < 0
]
so (d) is a descent direction.

### Q3. Why can AdaGrad become too conservative later in training?

**Answer:**
Because the accumulated squared gradient keeps increasing:
[
r_t = \sum_{\tau=1}^t g_\tau^2
]
This makes the denominator grow, so the effective learning rate
[
\frac{\eta}{\sqrt{r_t}+\epsilon}
]
shrinks toward zero.

### Q4. What exactly does bias correction fix in Adam?

**Answer:**
It fixes the early-stage underestimation of the first and second moment estimates caused by zero initialization:
[
m_t \text{ and } v_t \text{ are biased toward } 0
]
So Adam uses:
[
\hat m_t=\frac{m_t}{1-\beta_1^t}, \qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}
]

### Q5. If the loss is negative log-likelihood, what is the statistical meaning of training?

**Answer:**
Minimizing negative log-likelihood is equivalent to maximum likelihood estimation. Training is therefore an optimization-based form of statistical estimation.

---

## Important Clarification Questions Asked During Day 1

### 1. What is an exponential moving average?

An EMA is a recursively weighted average:
[
s_t = \rho s_{t-1} + (1-\rho)x_t
]
It gives more weight to recent values and exponentially decaying weight to older values.

### 2. Is the early-step issue in Adam just that the step size becomes (1-\rho)?

Not exactly. The correct statement is that the moving-average moment estimates are biased toward zero early on because they start from zero. Bias correction removes that early underestimation.

### 3. What does deterministic mean?

Deterministic means that under the same code, same input, and same environment, the computation produces the same output each time.
Seed fixes the random source; deterministic controls computation behavior.

### 4. Why should `set_seed()` not print anything?

Because it is a configuration function, not a reporting function. It should modify the environment and return `None`. The visible reporting should be handled by `report_config()`.

### 5. Why does `__pycache__` appear?

Because Python caches bytecode when importing modules. It is normal and should be ignored in Git.

---

## Two Sum — Final Algorithm Learned

### Problem

Given an integer array `nums` and an integer `target`, return the indices of the two numbers such that they add up to `target`.

### Final Python idea

Use a one-pass hash map:

* maintain a dictionary `seen`
* for each number, compute its complement
* if the complement is already in `seen`, return the two indices
* otherwise store the current number and index

### Final logic

```python
class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        seen = {}

        for i, num in enumerate(nums):
            complement = target - num

            if complement in seen:
                return [seen[complement], i]

            seen[num] = i

        return []
```

### Complexity

* Time: `O(n)`
* Space: `O(n)`

### Important implementation detail

Check first, then store:

```python
if complement in seen:
    return [seen[complement], i]

seen[num] = i
```

This prevents reusing the same element twice.

---

## Day 1 Final Status

**Completed**

### Main deliverables finished

* `src/utils/repro.py`
* `tests/test_repro.py`
* `Makefile`
* `src/leetcode/day01_two_sum.py`
* `docs/daily_log.md`
* `.gitignore`
* Day 1 git commit

### Main learning outcome

Day 1 successfully moved from:

* reading optimization concepts
  to
* building a minimal Python/PyTorch engineering workflow with testing, imports, reproducibility control, and a correct first algorithm implementation.


# Day 2 Summary — Deep Learning Intensive Plan
## Objective for Today
The main goal today was to move from “understanding reverse-mode AD” to “implementing a minimal scalar autograd engine by hand,” and then verifying its gradients with tests.

## Learning Resources Used Today

**Karpathy, The spelled-out intro to neural networks and backpropagation: building micrograd**
Main focus:
* scalar computational graph
* local derivatives
* reverse-mode autodiff
* topological order for backward pass

**Optimization_Jan31.pdf**
Main reading: Section 2, Automatic Differentiation
Main topics:
* symbolic differentiation
* numerical differentiation
* automatic differentiation
* forward mode vs reverse mode
* computation graph / evaluation trace
* adjoints and the relation to backprop

**LeetCode**
* 217. Contains Duplicate
* Link: https://leetcode.com/problems/contains-duplicate/

---

## What Was Completed Today

### 1. Learned the core concepts of Automatic Differentiation
Completed the conceptual study of reverse-mode AD, including:
* what a computation graph is
* what a local derivative is
* why backprop is essentially reverse-mode AD
* why forward mode is inefficient for large-parameter neural networks
* why reverse mode is efficient for scalar loss
* what an adjoint / sensitivity means

Key conceptual clarifications today:
* automatic differentiation is not a finite-difference approximation; it propagates derivatives exactly through the chain rule (up to machine precision)
* backprop propagates gradients backward from the final loss to the parameters
* if one node influences the output through multiple paths, gradient contributions must be accumulated, not overwritten

### 2. Implemented the minimal Value class
Implemented the basic structure of a minimal scalar autograd engine in `src/z2h/micrograd_pytorch/value.py`.
The current `Value` class includes:
* `data`
* `grad`
* `_prev`
* `_backward`
* `_op`
* `label`
* `__repr__`

This means each scalar node in the computation graph now stores:
* its current value
* its gradient
* its parent nodes
* the local backward rule for sending gradients to parents

### 3. Implemented local operations and backward propagation
Implemented and corrected the following operations and their local backward logic:
* `__add__`
* `__mul__`
* `__neg__`
* `__sub__`
* `__pow__` (scalar power)
* `tanh()`
* `exp()`
* `__rmul__`
* `__truediv__` (basic usable version for the current stage)

Also implemented:
* `backward()`
* topological traversal of the computation graph
* gradient propagation over reverse topological order

The most important engineering insight today was:
* `out.grad` is the starting point for local backward propagation
* every local backward rule must multiply by `out.grad`
* gradients must be accumulated with `+=`, because one node may receive contributions from multiple paths

### 4. Ran manual sanity checks
Completed and passed manual sanity checks for:

* **Addition:**
  `c = a + b`
  verified:
  `a.grad = 1.0`
  `b.grad = 1.0`
* **Multiplication:**
  `c = a * b`
  verified:
  `a.grad = b.data`
  `b.grad = a.data`
* **Negation:**
  `b = -a`
  verified:
  `a.grad = -1.0`
* **Subtraction:**
  `c = a - b`
  verified:
  `a.grad = 1.0`
  `b.grad = -1.0`
* **Power:**
  `b = a ** 2`
  verified:
  `a.grad = 2a`
* **tanh():**
  verified:
  `a.grad = 1 - tanh(a)^2`

These checks confirmed that the minimal backward mechanism is behaving correctly.

### 5. Wrote finite-difference gradient checks
Implemented finite-difference helper functions in `tests/test_micrograd.py`:
* `numerical_grad_1d`
* `numerical_grad_2d_x`
* `numerical_grad_2d_y`

Used them to validate the gradients from the autograd engine.
Completed tests:
* `test_backward_populates_grad`
* `test_grad_quadratic` (f(x)=x^2+2x+1)
* `test_grad_bilinear` (f(x,y)=xy+x+y)
* `test_grad_tanh_composite` (f(x)=\tanh(x^2+3x))

Test result:
* all tests in `test_micrograd.py` passed
* together with Day 1 `test_repro.py`, total: 7 passed

This shows:
* the current micrograd gradients agree with finite differences
* Day 2 work did not break the Day 1 project structure

### 6. Built a notebook demo
Completed a minimal demo in `notebooks/day02_micrograd.ipynb`:
* built a small graph:
  * `a`
  * `b`
  * `c = a * b`
  * `d = c + a`
  * `e = d.tanh()`
* printed forward data
* called `backward()`
* printed the gradients of the nodes

A very important phenomenon observed in the notebook:
* when `d = 8`, `tanh(8)` is extremely close to 1
* therefore `1 - tanh^2(8)` is extremely close to 0
* so the gradients become very small

This means:
* the output is reasonable
* this is not a bug, but the saturation behavior of tanh

### 7. Completed LeetCode
Completed today’s LeetCode problem:
* 217. Contains Duplicate

Method used:
* hash set

Complexity:
* Time: O(n)
* Space: O(n)

The current solution is correct.

---

## Most Important Understandings Today
* AD is not a finite-difference approximation
* reverse-mode AD is the abstract form of backprop
* gradients must be accumulated
* `out.grad` is the key quantity in local backward rules
* backward execution must follow reverse topological order
* finite difference is a validation tool, not the actual differentiation mechanism used in training
* tanh can saturate and produce very small gradients at large inputs

## Main Files Created or Updated Today
* `src/z2h/micrograd_pytorch/value.py`
* `tests/test_micrograd.py`
* `notebooks/day02_micrograd.ipynb`
* `src/leetcode/day02_contains_duplicate.py`

## Current Status
The core Day 2 learning and implementation tasks are complete, including:
* AD theory understanding
* minimal autograd engine
* backward mechanism
* finite-difference tests
* notebook demo
* LeetCode

If the final cleanup is still pending, the only remaining items are:
* update `docs/daily_log.md`
* make the git commit

Recommended commit message:
`day02: micrograd scalar autograd + tests`

# Day 3 Summary — Deep Learning Intensive Plan

## Objective for Today
The main goal today was to use one concrete function to compare evaluation trace, forward-mode AD, reverse-mode AD, and PyTorch autograd, so that manual differentiation and software autodiff could be connected clearly.

## Learning Resources Used Today

**Karpathy, The spelled-out intro to neural networks and backpropagation: building micrograd**  
Main focus:
* whole-expression-graph backward
* bug caused by node reuse
* why gradients must accumulate
* decomposing complex ops like `tanh`
* correspondence between micrograd and PyTorch autograd
* how autodiff connects to loss, parameter collection, and gradient descent
* how scalar-node autodiff grows into an MLP abstraction

**Homework_1_Sp23.pdf**  
Main reading: Problem 3  
Main topics:
* 2-dimensional Rosenbrock function
* evaluation trace
* manual forward-mode AD
* manual reverse-mode AD
* comparing manual derivatives with PyTorch autograd

Main function studied:
\[
f(x_1, x_2)=100(x_2-x_1^2)^2+(1-x_1)^2
\]
evaluated at
\[
(x_1,x_2)=(0,0)
\]

**LeetCode**
* 242. Valid Anagram
* Link: https://leetcode.com/problems/valid-anagram/

---

## What Was Completed Today

### 1. Recalibrated the true focus of the later Z2H micrograd segment
Instead of repeating Day 2 topics, I identified the actual new emphasis in the later part of the video:
* whole-graph backward rather than only local node rules
* the concrete bug that appears when one node is reused multiple times
* why gradient accumulation is required
* how a complex operation like `tanh` can still be decomposed into elementary steps
* how micrograd corresponds conceptually to PyTorch autograd
* how autodiff connects to training through loss, parameter collection, and gradient descent
* how scalar-node logic scales into neuron, layer, and MLP abstractions

### 2. Answered conceptual questions from the video
Clarified:
* the difference between one node’s `_backward()` and full-graph `backward()`
* why reused nodes force gradient accumulation
* why a complex function can still fit into the same autodiff framework
* how `Value.data`, `Value.grad`, `_prev`, and `_backward` correspond to PyTorch ideas
* why loss function, parameter collection, and gradient descent form the bridge from differentiation to training
* why moving from `Value` to MLP mainly adds software abstraction and parameter organization, not new calculus

### 3. Worked through Homework 1 Problem 3
For the Rosenbrock function at \((0,0)\), determined:
* function value:
  \[
  f(0,0)=1
  \]
* target gradients:
  \[
  \frac{\partial f}{\partial x_1}(0,0)=-2,\qquad
  \frac{\partial f}{\partial x_2}(0,0)=0
  \]

### 4. Built the evaluation trace
Decomposed the function into intermediate variables suitable for manual AD tracing:
\[
v_1=x_1,\quad
v_2=x_2,\quad
v_3=v_1^2,\quad
v_4=1-v_1,\quad
v_5=v_4^2,
\]
\[
v_6=v_2-v_3,\quad
v_7=v_6^2,\quad
v_8=100v_7,\quad
v_9=v_8+v_5
\]

with \(v_9=f(x_1,x_2)\).

This made it possible to run both forward-mode and reverse-mode on the same trace.

### 5. Performed manual forward-mode AD
Manually propagated derivative traces for both directions:
* with respect to \(x_1\)
* with respect to \(x_2\)

Final result:
\[
\frac{\partial y}{\partial x_1}=-2,\qquad
\frac{\partial y}{\partial x_2}=0
\]

This confirmed that the forward derivative trace was set up correctly.

### 6. Performed manual reverse-mode AD
Manually propagated adjoints backward through the same evaluation trace.

Final result:
\[
\frac{\partial y}{\partial x_1}=-2,\qquad
\frac{\partial y}{\partial x_2}=0
\]

This matched the forward-mode result exactly, showing that both manual AD approaches were consistent on the same traced function.

### 7. Verified the result using PyTorch autograd
In the notebook, defined the same Rosenbrock function in PyTorch at \((0,0)\), called `.backward()`, and verified:
* function value = `1.0`
* `x1.grad = -2.0`
* `x2.grad = 0.0`

This confirmed that:
* manual forward-mode AD
* manual reverse-mode AD
* PyTorch autograd

all agreed exactly.

### 8. Built the Day 3 notebook
Completed the main body of:
* `notebooks/day03_ad_traces.ipynb`

The notebook now includes:
* the Rosenbrock evaluation trace
* manual forward-mode AD
* manual reverse-mode AD
* PyTorch verification

This means the Day 3 notebook is no longer only conceptual; it now contains a full numerical comparison pipeline.

### 9. Completed LeetCode
Completed today’s LeetCode problem:
* 242. Valid Anagram

Final approach used:
* one dictionary
* increment counts for `s`
* decrement counts for `t`
* verify all counts return to zero

Complexity:
* Time: `O(n)`
* Space: `O(k)` or, more conservatively, `O(n)`

The current solution is correct and cleaner than the original two-dictionary version.

---

## Most Important Understandings Today
* forward-mode and reverse-mode can be compared cleanly on the same evaluation trace
* reverse-mode AD is closer to backprop because it starts from a scalar output and propagates sensitivities backward
* gradient accumulation is not a coding trick; it is required because one node may influence the output through multiple downstream paths
* PyTorch autograd is not conceptually different from micrograd; it is a larger-scale engineering implementation of the same mechanism
* moving from `Value` to MLP mainly adds abstraction layers:
  * neuron
  * layer
  * model
  * parameter collection
  * training loop
* one concrete traced function is enough to make the relationship between symbolic structure, manual AD, and software autodiff much clearer

## Main Files Created or Updated Today
* `notebooks/day03_ad_traces.ipynb`
* `src/leetcode/day03_valid_anagram.py`
* `docs/daily_log.md`

## Current Status
The core Day 3 learning and implementation tasks are complete, including:
* video study
* Homework 1 Problem 3
* evaluation trace
* manual forward-mode AD
* manual reverse-mode AD
* PyTorch verification
* LeetCode 242

If final cleanup is still pending, the only remaining items are:
* confirm `docs/daily_log.md` is saved
* make the git commit

Recommended commit message:
`day03: forward+reverse mode ad trace + verification`

# Day 4 Summary — Deep Learning Intensive Plan

## Objective for Today
The main goal today was to connect the optimization material from STAT-9340 Homework 1 Problem 1 with actual implementation by hand: loading the dataset, defining the nonlinear model, deriving gradients manually, implementing SGD and SGD with momentum, and comparing their behavior on both synthetic and real data.

## Learning Resources Used Today

**Karpathy, The spelled-out intro to language modeling: building makemore**  
Link: https://youtu.be/PaCmpygFfXo

Main focus today:
* `00:00:00–00:24:02`
  * intro
  * reading and exploring the dataset
  * exploring the bigrams in the dataset
  * counting bigrams in a python dictionary
  * counting bigrams in a 2D torch tensor
  * visualizing the bigram tensor
  * token cleanup and sampling intuition
* `00:36:17–01:00:50`
  * vectorized normalization
  * tensor broadcasting
  * negative log likelihood loss
  * model smoothing with fake counts

Main topics:
* bigram counting as a probabilistic model
* normalization from counts to probabilities
* negative log likelihood as an optimization objective
* smoothing and why zero counts are a problem

**Homework_1_Sp23.pdf**  
Link: [Homework_1_Sp23.pdf](sandbox:/mnt/data/Homework_1_Sp23.pdf)

Main reading: Problem 1a–1c

Main topics:
* nonlinear regression model
* squared error objective
* hand-coded SGD
* SGD with momentum
* effect of initialization, learning rate, and batch size

Model studied:
\[
y_i=\theta_1(\sin(x_i)+\cos(\theta_2 x_i))+\epsilon_i,\qquad \epsilon_i\sim N(0,\sigma^2)
\]

**Dataset**
* [HW1_Problem1_Data.csv](sandbox:/mnt/data/HW1_Problem1_Data.csv)

**LeetCode**
* 20. Valid Parentheses
* Link: https://leetcode.com/problems/valid-parentheses/

---

## What Was Completed Today

### 1. Reviewed the learning objective of Homework 1 Problem 1
Clarified the structure of the problem:
* 1a: load the data and make a scatter plot
* 1b: implement SGD by hand to estimate \(\theta_1,\theta_2\)
* 1c: add momentum and compare with basic SGD

Also clarified that the optimization target is the squared error loss on noisy observed data.

### 2. Built the Day 4 problem skeleton
Implemented the core problem structure:
* `load_hw1_problem1_data(...)`
* `generate_hw1_problem1_synth(...)`
* `model_fn(x, theta1, theta2)`
* `squared_error_loss(y_true, y_pred)`

This separated:
* data loading
* model definition
* loss computation

which made the later optimizer implementation cleaner.

### 3. Completed Problem 1a on the real dataset
Loaded `HW1_Problem1_Data.csv` successfully and plotted the scatter plot of the observed data.

This confirmed:
* the file path and columns were read correctly
* the observed pattern was nonlinear and consistent with the model structure
* the project was ready for optimization experiments

### 4. Derived gradients by hand for Problem 1b
For the model
\[
\hat y_i=\theta_1(\sin(x_i)+\cos(\theta_2 x_i)),
\]
and mean squared error
\[
L=\frac{1}{m}\sum_{i=1}^m (y_i-\hat y_i)^2,
\]
manually derived the gradients with respect to:
* \(\theta_1\)
* \(\theta_2\)

This was the key theoretical step before implementing SGD without autograd.

### 5. Implemented hand-coded SGD
Implemented a mini-batch SGD optimizer in pure Python / NumPy.

Main features:
* random mini-batch sampling
* hand-coded gradient updates
* loss tracking by iteration
* tunable initialization, learning rate, batch size, and max iterations

This satisfied the core requirement of Homework 1b.

### 6. Implemented SGD with momentum
Implemented momentum-based SGD for Homework 1c.

Main features:
* velocity variables for both parameters
* momentum coefficient \(\alpha\)
* same hand-coded gradients as basic SGD
* loss logging for comparison with SGD

This made it possible to compare:
* optimizer stability
* early descent speed
* final fit quality

### 7. Tested first on synthetic data
Before running on the real dataset, tested both optimizers on synthetic data generated from the same model form.

This was useful because it verified:
* the gradient formulas were correct
* the update rules were pointing in the right direction
* the implementation was numerically stable enough to proceed

An important observation:
* the first momentum configuration was too aggressive and caused oscillation
* after reducing the learning rate, the momentum version became much more stable

### 8. Ran both optimizers on the real dataset
Ran SGD and SGD with momentum on `HW1_Problem1_Data.csv`.

Observed:
* both methods reduced the loss substantially
* momentum showed faster initial descent
* both methods converged to very similar final loss values
* the final fitted curves were nearly identical

This suggests:
* optimizer choice affected optimization speed more than final fit quality under the current tuning
* momentum helped early optimization but did not materially improve the final fit over basic SGD

### 9. Built comparison plots
Generated the two key plots required for interpretation:

* **loss curve comparison**
  * SGD
  * SGD with momentum

* **data + fitted curve comparison**
  * original scatter data
  * SGD fitted curve
  * SGD with momentum fitted curve

Important implementation note:
* sorted `x` before plotting fitted curves so the line plot was meaningful and not zig-zagged by unsorted inputs

### 10. Completed LeetCode
Completed today’s LeetCode problem:
* 20. Valid Parentheses

Method used:
* stack
* matching dictionary for brackets

Complexity:
* Time: `O(n)`
* Space: `O(n)`

The current solution is correct.

---

## Most Important Understandings Today
* a probability model becomes trainable once it is paired with a loss function
* negative log likelihood is a clean way to turn probability assignment into an optimization objective
* hand-coded gradients can be implemented cleanly once model, residual, and loss are separated
* optimizer behavior should be judged by both stability and speed, not only final loss
* momentum can help early descent, but poor tuning can introduce severe oscillation
* on this problem, optimizer choice influenced convergence speed more than final fitted shape under the current settings
* for plotting fitted nonlinear curves, sorting the input is necessary before drawing the prediction line

## Main Files Created or Updated Today
* `src/stat9340/hw1_problem1.py` or corresponding Day 4 notebook version
* `src/leetcode/day04_valid_parentheses.py`
* `docs/daily_log.md`
* plots saved under `artifacts/day04/`

## Current Status
The core Day 4 learning and implementation tasks are complete, including:
* video study
* Homework 1 Problem 1a–1c
* real data loading and plotting
* hand-derived gradients
* SGD implementation
* SGD with momentum implementation
* synthetic-data verification
* real-data comparison
* LeetCode 20

If final cleanup is still pending, the only remaining items are:
* confirm `docs/daily_log.md` is saved
* save the final plots to `artifacts/day04/`
* make the git commit

Recommended commit message:
`day04: hw1 p1 sgd+momentum on real data`

# Day 5 Summary — Deep Learning Intensive Plan

## Objective for Today
The main goal today was to extend the optimizer comparison from Homework 1 Problem 1 beyond SGD and momentum by adding Adam and PSO, then building a unified comparison across all optimizers on the same nonlinear regression task.

## Learning Resources Used Today

**Karpathy, The spelled-out intro to language modeling: building makemore**  
Link: https://youtu.be/PaCmpygFfXo

Main focus today:
* `01:00:50–01:26:17`
  * model smoothing with fake counts
  * the neural network approach
  * creating the bigram dataset for the neural net
  * one-hot encodings
  * one linear layer implemented with matrix multiplication
  * softmax probabilities
* `01:35:49–01:57:45`
  * vectorized loss
  * backward and update in PyTorch
  * putting everything together
  * one-hot as row selection in the weight matrix
  * smoothing as regularization
  * sampling from the neural net

Main topics:
* one-hot encoding as row selection
* vectorized loss as a tensor-parallel version of per-sample loss
* connecting model, loss, gradient, and update into a single training loop

**Optimization_Jan31.pdf**  
Main reading:
* PSO-related slides
  * Particle Swarm Optimization
  * Velocity Updates
  * Algorithm Details

Main topics:
* particle as candidate solution
* inertia / cognitive / social terms
* personal best and global or neighborhood best
* position and velocity update order

**Homework_1_Sp23.pdf**  
Link: [Homework_1_Sp23.pdf](sandbox:/mnt/data/Homework_1_Sp23.pdf)

Main reading: Problem 1d–1f

Main topics:
* hand-coded Adam
* hand-coded PSO
* optimizer comparison
* good / bad fit comparison

Model studied:
\[
y_i=\theta_1(\sin(x_i)+\cos(\theta_2 x_i))+\epsilon_i,\qquad \epsilon_i\sim N(0,\sigma^2)
\]

**Dataset**
* [HW1_Problem1_Data.csv](sandbox:/mnt/data/HW1_Problem1_Data.csv)

**LeetCode**
* 121. Best Time to Buy and Sell Stock
* Link: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/

---

## What Was Completed Today

### 1. Clarified the Day 5 extension of Homework 1 Problem 1
Identified the new scope of Problem 1d–1f:
* 1d: implement Adam by hand
* 1e: implement PSO by hand
* 1f: compare optimizer behavior and fitted results

This made Day 5 a continuation of the same nonlinear regression problem, but with a broader optimizer comparison framework.

### 2. Refined the conceptual understanding of one-hot encoding and vectorized loss
Clarified two important ideas from the makemore video:

* **one-hot encoding**
  * transforms a discrete token index into a sparse indicator vector
  * when multiplied by a weight matrix, it effectively selects one row of the matrix

* **vectorized loss**
  * has the same mathematical objective as batch loss
  * differs only in implementation style
  * replaces explicit sample-by-sample loops with parallel tensor operations

These ideas were useful conceptually even though the Homework 1 implementation remained in pure Python / NumPy.

### 3. Studied PSO before implementation
Inserted a bridge step before coding PSO and reviewed the PSO section from the optimization slides.

Key concepts clarified:
* one particle represents one candidate parameter vector, such as \((\theta_1,\theta_2)\)
* velocity update contains:
  * inertia term
  * cognitive term
  * social term
* \(p_i\) is the personal best of particle \(i\)
* \(g_i\) is the best location from the social reference set, such as a neighborhood or global best
* the standard iteration order is:
  * evaluate current positions
  * update personal best
  * update global/neighborhood best
  * update velocity
  * update position

### 4. Implemented Adam by hand
Built a hand-coded Adam optimizer for the same Homework 1 regression model.

Main components implemented:
* first moment estimates
* second moment estimates
* bias correction
* parameter updates for \(\theta_1\) and \(\theta_2\)
* loss logging

Important debugging insight:
* both gradients for \(\theta_1\) and \(\theta_2\) must be computed from the same parameter state within each iteration
* updating one parameter before computing the other gradient would contaminate the update

After correction, Adam ran stably and converged properly on the real dataset.

### 5. Compared Adam with SGD and SGD with momentum
Ran Adam on the same real dataset and compared it with the previous Day 4 optimizers.

Observed:
* Adam reduced the loss substantially
* Adam was stable
* SGD with momentum still showed the fastest early descent under the current tuning
* all three methods converged to very similar final loss regions and nearly identical fitted curves

This showed that, for this problem, optimizer differences were more visible in the training path than in the final fit.

### 6. Implemented PSO by hand
Built a PSO optimizer for the same squared error objective.

Core components implemented:
* particle positions
* particle velocities
* personal best positions and losses
* neighborhood-best or social reference selection
* iterative velocity and position updates
* loss history tracking

Important corrections made during debugging:
* excluded each particle from its own neighborhood search
* corrected the iteration count logic
* added boundary handling through clipping
* aligned the return format with the other optimizers
* interpreted the tracked PSO loss as a best-so-far history

### 7. Improved PSO until it reached comparison quality
The first PSO run was not competitive with SGD / momentum / Adam.  
After debugging and adjustment, PSO improved substantially.

Final behavior:
* PSO reduced the loss to roughly the same region as the other optimizers
* it no longer stalled at a clearly worse level
* its fitted curve became nearly indistinguishable from the others

This was the main engineering success of Day 5:
* PSO went from “running” to “comparable”

### 8. Built the four-optimizer comparison
Collected results across:
* SGD
* SGD with momentum
* Adam
* PSO

For each optimizer, organized:
* final loss
* final parameter estimates
* optimizer label
* hyperparameter setting
* maximum iteration count

This formed the basis for the Day 5 comparison harness.

### 9. Produced final comparison plots
Generated the two main plots needed for interpretation:

* **loss curve comparison**
  * SGD
  * SGD with momentum
  * Adam
  * PSO

* **data + fitted curve comparison**
  * original data points
  * fitted curve from each optimizer

Final interpretation from these plots:
* all four optimizers substantially reduced the loss
* momentum often had the fastest early descent
* Adam was stable
* PSO became competitive after debugging and tuning
* all four methods ended with very similar fitted curves

### 10. Completed LeetCode
Completed today’s LeetCode problem:
* 121. Best Time to Buy and Sell Stock

Main idea:
* one-pass scan
* track the minimum price seen so far
* update the maximum profit greedily

Complexity:
* Time: `O(n)`
* Space: `O(1)`

---

## Most Important Understandings Today
* optimizer comparison should consider both early descent behavior and final fit quality
* Adam requires consistent gradient evaluation, first/second moments, and bias correction
* PSO does not use gradients; it searches directly in parameter space using particle dynamics
* a particle is a candidate parameter vector, not a data point
* PSO performance depends heavily on update structure, neighborhood logic, and parameter tuning
* one-hot encoding is best understood as row selection in a weight matrix
* vectorized loss and batch loss optimize the same objective; they differ mainly in implementation form
* under the current tuning, all four optimizers reached similar final fitted solutions on this problem

## Main Files Created or Updated Today
* optimizer comparison code in the Day 5 notebook or corresponding source file
* `artifacts/day05/hw1_p1_optimizer_compare.csv`
* comparison plots under `artifacts/day05/`
* `src/leetcode/day05_best_time_buy_sell_stock.py`
* `docs/daily_log.md`

## Current Status
The core Day 5 learning and implementation tasks are complete, including:
* video study
* PSO slide study
* Homework 1 Problem 1d–1f
* hand-coded Adam
* hand-coded PSO
* optimizer comparison across four methods
* final comparison plots
* LeetCode 121

If final cleanup is still pending, the only remaining items are:
* confirm `artifacts/day05/hw1_p1_optimizer_compare.csv` is saved
* confirm final plots are saved under `artifacts/day05/`
* update `docs/daily_log.md`
* make the git commit

Recommended commit message:
`day05: hw1 p1 adam+pso + comparison harness`

# Day 6

## Daily Blueprint

- **Reading**
  - `Optimization_Jan31.pdf`
    - Section 1 recap: loss definitions
    - Section 2 recap: gradients
  - `Homework_1_Sp23.pdf`
    - Problem 2a–2c

- **Specific Practice**
  - Rewrite Homework 1 Problem 2a–2c in **PyTorch**
  - Build:
    - Rosenbrock contour plot
    - gradient vectors
    - optimizer trajectories for:
      - SGD
      - SGD with momentum
      - Adam
  - Starting point: `(-0.1, 2.5)`
  - Store intermediate optimizer states

- **LeetCode**
  - 53. Maximum Subarray
  - https://leetcode.com/problems/maximum-subarray/

- **Expected Outcome**
  - `notebooks/day06_rosenbrock_paths.ipynb`
  - commit:
    - `day06: hw1 p2 rosenbrock + optimizer trajectories (pytorch)`

---

## Core Formula

The 2D Rosenbrock function is

\[
f(x_1, x_2)=100(x_2-x_1^2)^2+(1-x_1)^2
\]

Manual gradient:

\[
\frac{\partial f}{\partial x_1}=-400x_1(x_2-x_1^2)-2(1-x_1)
\]

\[
\frac{\partial f}{\partial x_2}=200(x_2-x_1^2)
\]

Global minimum:

\[
(x_1, x_2)=(1,1)
\]

---

## What to Understand

### 1. Why Rosenbrock is hard
It is not mainly difficult because of many local minima.  
The main issue is a **narrow, curved valley** with very different curvature in different directions.

### 2. What each plot means
- **Contour plot**: the geometry of the objective surface
- **Gradient vectors**: the local steepest direction
- **Optimizer trajectory**: how a specific update rule actually moves across the surface

### 3. Why compare SGD, momentum, and Adam
- **SGD** uses only the current gradient, so it often zig-zags in narrow valleys
- **Momentum** carries historical direction, so it usually moves more smoothly
- **Adam** adds coordinate-wise adaptive scaling, so it often enters the valley faster

---

## Engineering Checklist

### Phase 1
- [ ] Implement `rosenbrock(x)` in PyTorch
- [ ] Implement manual gradient
- [ ] Verify manual gradient against autograd

### Phase 2
- [ ] Draw contour plot
- [ ] Draw gradient vectors on the contour
- [ ] Run SGD from `(-0.1, 2.5)`
- [ ] Run SGD with momentum from `(-0.1, 2.5)`
- [ ] Run Adam from `(-0.1, 2.5)`
- [ ] Save intermediate states for each optimizer
- [ ] Overlay trajectories on the contour

### Phase 3
- [ ] Compare the three optimizers in words
- [ ] Save notebook
- [ ] Write short notes in `docs/daily_log.md`
- [ ] Make the required Git commit

---

## Minimal Comparison Template

Use these four lines only:

- entry into valley
- oscillation across valley walls
- progress along valley floor
- final distance to `(1, 1)`

---

## Self-Check Questions

1. Why is minibatch SGD noisier than full gradient descent, but still useful?

2. Why can momentum both speed up movement along a valley and reduce oscillation?

3. Why is Rosenbrock difficult because of valley geometry rather than many local minima?

4. If SGD zig-zags, momentum is smoother, and Adam enters the valley quickly then slows, what does that tell you about the update rules?

---

## Final Deliverables

- `notebooks/day06_rosenbrock_paths.ipynb`
- one contour + gradient plot
- one contour + trajectory overlay plot
- one short comparison note
- commit:
  - `day06: hw1 p2 rosenbrock + optimizer trajectories (pytorch)`

# Day 8 Summary / Day 8 学习总结

## English

### 1. Day objective

Today’s target was Day 8 of the study plan: use the shallow neural network material plus Homework 2 Problem 1a to build a **1-hidden-layer MLP regression model with manual backpropagation and L2 penalty**, and compare it against a baseline. The plan explicitly specifies manual gradients and treats this as the Day 8 engineering task. :contentReference[oaicite:0]{index=0}  
Homework 2 Problem 1a specifically asks to **predict `quality`**, include **L2 shrinkage**, use a **held-out test set**, and compare performance to **multiple linear regression**. :contentReference[oaicite:1]{index=1}

### 2. What was completed today

- Read and discussed the shallow neural network formulation:
  - hidden unit definition
  - activation function role
  - output function role
  - single-output regression vs binary / multi-class classification
- Clarified that **HW2 1a is regression**, not classification, because `quality` is treated as a numeric response in the homework. :contentReference[oaicite:2]{index=2}
- Implemented the core **manual backpropagation** training loop for a one-hidden-layer neural network.
- Added **output bias** through the hidden-bias construction.
- Added **L2 penalty** into the gradient update.
- Added **held-out test MSE** evaluation.
- Fit a **multiple linear regression baseline** and compared test MSEs.
- Verified that training loss decreases steadily across epochs.

### 3. Key concepts understood today

#### 3.1 Hidden activation vs output activation

A hidden layer unit is
\[
z_j = f\left(\sum_i w_{ji}x_i\right),
\]
so the hidden layer consists of a linear predictor followed by a nonlinear activation. The notes explicitly state that the nonlinearity of \(f\) is what makes the model more than “a big linear regression.” :contentReference[oaicite:3]{index=3}

The output function \(g\) is task-dependent:

- regression: identity
- binary classification: sigmoid
- multi-class classification: softmax :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

#### 3.2 Why HW2 1a is regression

The homework defines `quality` as a score and asks to compare to multiple linear regression using test MSE. That means the problem is treated as **single-output regression**, even if the raw response is integer-valued. :contentReference[oaicite:6]{index=6}

#### 3.3 Matrix shape logic

Under row-wise Python notation:

- \(X \in \mathbb{R}^{m \times p}\)
- \(W \in \mathbb{R}^{p \times J}\)
- \(H \in \mathbb{R}^{m \times J}\)
- \(\beta \in \mathbb{R}^{J \times 1}\) or \((J+1)\times 1\) with output bias
- \(y_{\text{pred}} \in \mathbb{R}^{m \times 1}\)

The gradient must have the same shape as the corresponding parameter matrix.

#### 3.4 Input standardization vs regularization

The lecture notes list these separately:

- **input standardization**: make inputs mean zero and standard deviation one
- **regularization**: weight decay / L2 penalty in the objective function :contentReference[oaicite:7]{index=7}

These are related but not the same thing.

### 4. Errors made today

This was the most important part of the session.

#### 4.1 Misidentified HW2 1a as a classification problem

At first, the task was treated as if `quality` were a class label. That was incorrect.  
Correction: HW2 1a is a **regression** task on `quality`. :contentReference[oaicite:8]{index=8}

#### 4.2 Mixed up hidden-layer output and pre-activation

An incorrect expression appeared:
\[
y_{\text{pred}} = z @ \beta
\]

Correction:
\[
y_{\text{pred}} = h @ \beta
\]
because \(\beta\) connects the **activated hidden units** to the output, not the raw pre-activation.

#### 4.3 Confused true label and model output in classification notation

There was confusion between:

- \(\tilde y\): observed target
- \(y\): network output probability

Correction: in classification, \(\tilde y\) is the label, while the network output is a continuous probability in \((0,1)\) or a softmax vector. :contentReference[oaicite:9]{index=9}

#### 4.4 Confused dimensions of \(\beta\) and \(\partial Q / \partial \beta\)

There was uncertainty about whether \(\beta\) should be \(J \times K\) or \(K \times J\).  
Resolution: the dimension depends on matrix convention, but the gradient must always have the **same shape** as \(\beta\).

#### 4.5 Wrote the wrong matrix expression for \(g_W\)

An early gradient form for `gw` attempted to chain matrix multiplication directly in a way that ignored the required elementwise hidden-layer derivative.  
Correction:
\[
\delta_h = (y_{\text{pred}}-y)\beta^\top \circ h \circ (1-h), \qquad
g_W = X^\top \delta_h
\]

This fixed both the logic and the shape.

#### 4.6 Added a bias column to \(y\)

That was incorrect.  
Correction: bias belongs in the input and hidden feature matrices, not in the response vector.

#### 4.7 Used the wrong minibatch indexing logic

Earlier versions had:
- batching based on the wrong dimension
- shuffled arrays created but not actually used

Correction:
- shuffle over sample indices
- slice `x_shuffle` and `y_shuffle`
- divide gradients by the actual current minibatch size

#### 4.8 Inconsistent regularization on output bias

The loss excluded output bias from the L2 penalty, but the gradient originally penalized the full `beta`.  
Correction: the penalty on `beta` was adjusted so that the output bias row is not regularized.

#### 4.9 Stale variable in the loss calculation

`beta_nonbias` used in the epoch loss was initially taken from an earlier state rather than the updated `beta`.  
Correction: recompute `beta_nonbias = beta[1:]` immediately before calculating the loss.

#### 4.10 Train/test feature mismatch and notebook-state confusion

A matrix multiplication error appeared because train/test feature dimensions looked inconsistent after adding bias.  
The final diagnosis was that:
- the split logic itself was fine
- the more likely issue was notebook state / stale function version / inconsistent inputs during earlier runs

#### 4.11 Forgot the homework requirement to remove `type`

Homework 2 Problem 1a requires prediction using all inputs **except `type`**. This needed to be explicitly checked in the final pipeline. :contentReference[oaicite:10]{index=10}

#### 4.12 Mixed array/DataFrame types in the linear regression baseline

`fit()` was called with `.values`, while `predict()` was called on a DataFrame.  
This caused a warning but not a wrong result.  
Correction: use arrays on both sides or DataFrames on both sides.

### 5. Current implementation status

#### Completed
- manual 1-hidden-layer backprop core
- L2 penalty
- output bias handling
- held-out test MSE
- multiple linear regression baseline
- test MSE comparison

#### Still worth confirming in the final saved version
- predictor matrix excludes `type`
- train/test standardization uses training-set mean/std
- final notebook/script is cleaned and reproducible

### 6. Current numerical result

From the latest run:

- Neural network test MSE: approximately **0.7592**
- Linear regression test MSE: approximately **0.5539**

So under the current tuning configuration:

> the linear regression baseline performed better than the manual neural network.

This is a valid result, not a failure. It means the current network configuration still needs tuning if the goal is to outperform the linear baseline.

### 7. Practical interpretation

A reasonable interpretation paragraph for the homework is:

> The manually implemented one-hidden-layer neural network achieved a test MSE of about 0.7592, while the multiple linear regression baseline achieved a test MSE of about 0.5539. Under the current tuning configuration, the linear regression model performed better on the held-out test set. This suggests that the neural network may require further tuning of hidden size, learning rate, regularization strength, or input preprocessing to achieve competitive performance.

### 8. Final status for Day 8

Recommended status label:

```md
[✓] Day 8 core objective completed
[✓] HW2 1a manual backprop model implemented
[✓] Baseline comparison completed
[ ] Final cleanup and packaging