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