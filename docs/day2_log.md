### 09:00–10:15
**Z2H: “The spelled-out intro to neural networks and backpropagation: building micrograd”**
* **Watch** 0:00–1:15:00
* **URL:** [https://youtu.be/VMj-3S1tku0](https://youtu.be/VMj-3S1tku0)

**Your focus while watching:**
* scalar computational graph
* local derivatives
* topological order for backward pass
* why reverse-mode autodiff is efficient for scalar loss
* how Value.data and Value.grad should be separated conceptually

### 10:30–11:30
**STAT-9340: Optimization — Overview of Methods**
* **Read** Section 2: “Automatic Differentiation” from `Optimization_Jan31.pdf`

**Your extraction target:**
* forward mode vs reverse mode
* computational graph viewpoint
* why reverse mode matches neural network training
* chain rule as graph propagation, not just symbolic differentiation

### 11:30–12:00
**Write a short handwritten or markdown note:**
* “What is the difference between symbolic differentiation, numerical differentiation, and automatic differentiation?”
* “Why is reverse-mode AD the right abstraction for deep learning?”

---

## [Specific Practice]

### 13:00–15:30
**Implement:**
`src/z2h/micrograd_pytorch/`

**Minimum required files:**
* `src/z2h/micrograd_pytorch/value.py`
* `tests/test_micrograd.py`
* `notebooks/day02_micrograd.ipynb`

Today’s implementation scope is only the minimal scalar autograd engine.
**Your Value object should at minimum support:**
* scalar data storage
* gradient storage
* parent tracking
* local backward function
* operator overloads for:
  * addition
  * multiplication
  * negation
  * subtraction
  * power by scalar
* activation:
  * tanh() at minimum

**You must also implement:**
* topological sort over the computation graph
* backward() that traverses the graph in reverse topological order

> Do not use PyTorch autograd inside this engine.
> PyTorch is only allowed later for checking and comparison.

### 15:30–17:00
**Write unit tests in:**
`tests/test_micrograd.py`

You need 3 gradient-correctness tests against finite differences.

**Use three small scalar functions, for example:**
* f(x)=x2+2x+1
* f(x)=x 2 +2x+1
* f(x,y)=xy+x+y
* f(x,y)=xy+x+y
* f(x)=tanh⁡(x2+3x)
* f(x)=tanh(x 2 +3x)

**For each one:**
* compute gradient from your engine
* compute numerical gradient by finite differences
* check they are close within a tolerance

**Also add one graph-structure sanity test:**
* calling backward() on a scalar output produces non-None gradients on upstream nodes

### 17:00–18:00
**LeetCode: 217. Contains Duplicate**
* **URL:** [https://leetcode.com/problems/contains-duplicate/](https://leetcode.com/problems/contains-duplicate/)

**Required standard:**
* solve it in Python
* use the hash-set approach
* write time and space complexity
* save it to something like: `src/leetcode/day02_contains_duplicate.py`

### 18:00–18:30
**Update:**
`docs/daily_log.md`

**Record:**
* what micrograd components you built
* where reverse-mode AD felt unclear
* which gradient test passed first
* which bug took the longest to fix

---

## [Expected Outcome]

**By the end of Day 2, you should have:**
* `src/z2h/micrograd_pytorch/value.py` with a working scalar autograd engine
* `tests/test_micrograd.py` with finite-difference checks on 3 functions
* `notebooks/day02_micrograd.ipynb` showing at least one worked demo graph and one sanity-check gradient
* `src/leetcode/day02_contains_duplicate.py`
* A Day 2 commit with exactly: `day02: micrograd scalar autograd + tests`

**Minimal acceptance standard for Day 2:**
* your custom Value graph can propagate gradients correctly
* tests pass
* you can explain why reverse-mode AD is more natural than forward-mode AD for scalar-loss neural network training

*Note: some earlier uploaded files have expired on my side. I can still follow the Day 2 master plan exactly, but if you want me to inspect or quote a previous PDF directly again, re-upload that file.*