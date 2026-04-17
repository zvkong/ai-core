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