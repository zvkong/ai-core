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