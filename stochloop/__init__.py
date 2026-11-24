"""stochloop: learned stochastic computation in looped transformers.

Two sources of stochasticity are combined inside one weight-tied (looped)
transformer of fixed size:

* depth stochasticity  -- *when* to stop looping (stochastic / learned halting),
* path stochasticity   -- *what* to compute, via a random seed encoding that is
                          injected into every loop iteration.

Both are trained under an explicit average-compute constraint.
"""

__all__ = ["tasks", "model", "objectives", "evaluation", "coloring"]
