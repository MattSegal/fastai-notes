## Gradient Descent

### Linear Example

Say we have some function that we want to learn: `y = 2x + 30`, and a 100 (x, y) pairs sampled from the function.

We guess that the function is linear so we assume that it is parameterized by (`a`, `b`), ie `y = ax + b`.

How do we learn the correct values of `a` and `b`?

We can start by

- guessing an initial value (`a`, `b`) = (1, 1)
- making a prediction on a sample (`x`, `y`) = (14, 58), giving us `ŷ = 1 * 14 + 1 = 15`
- calculate a prediction error `e = (ŷ - y)^2 = (15 - 58)^2 = 1849`

If we know the gradient of the error with respect to each parameter, ie `de/da` and `de/db`, then we can update each parameter to reduce the error, using some fixedstep size `s = 0.0001`:

- `a' = a - s * de/da`
- `b' = b - s * de/db`

So how do we calculate the gradient? We could use:

- Finite difference method
- Analytical solution

In the finite difference method, we perturb each variable independently by some small amount, and use the difference between the output and perturbed output to get the gradient. This is great because we don't need to know the analytical solution to the gradient. The problem is that the finite difference method suffers the curse of dimensionality - a high dimensional space will have a huge number of parameters and it'll take too long to perturb each one individually.

So, we use the analytical solution to get the gradient instead:

- `de/da = 2x(ax + b - y)`
- `de/db = 2(ax + b - y)`

This limits our choices for models (they must be differentiable wrt all parameters), but it's computationally much faster.

L5 - 1:50
