## Gradient Descent

We use gradient descent to minimize the loss function of a parameterized model (ie. neural network) over some dataset. We use gradient descent to do this optimization because it's a nice balance to converge and cheap to compute:

> ...gradient descent is a relatively efficient optimization method, since the computation of first-order partial derivatives w.r.t. all the parameters is of the same computational complexity as just evaluating the function ([Source](https://arxiv.org/pdf/1412.6980.pdf))

There are oher optimization methods which could be used as well, including:

- random guess-and-check (bad idea)
- [genetic algorithms](https://eng.uber.com/deep-neuroevolution/)
- other gradient-based approaches, like Newton's method (computationally expensive)

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

### Batch Size

How much data should we use in a single update? For each gradient update we could use:

- all the training data available ("batch gradient descent")
- a single sample of training data ("online gradient descent")
- a random subset of the training data ("stochastic gradient descent")

Batch gradient descent is:

- the least noisy alternative
- the most computationally expensive per update

Online gradient descent is:

- the noisiest alternative
- computationally the least expensive per update

Stochastic gradient descent (SGD) allows us to fine-tune this trade off, by selecting a batch size between 1 and the size of the training set. With a larger batch size we may see more redundant data in the batch, which is wasteful. A smaller batch size will be noisier than a larger batch, because of the increased weight given to any outliers. A small amount of gradient noise isn't necessarily a bad thing, as it may help escape local minimal or plateaus.

Although batch gradient descent has the most expensive computation per _update_, it provides the fastest computation per _epoch_, because it'll typically chew through the training set more quickly, given sufficient hardware. A very large batch size may not fit in-memory on a GPU.

OpenAI in "[How AI Training Scales](https://blog.openai.com/science-of-ai/)", posit that a metric called "gradient noise scale" can be used to assit in choosing the batch size.

Note that when doing SGD you should shuffle the data after every epoch.

### Momentum Method

We can make an assumption about our loss-surface to help speed up training. Let's assume that our past gradient updates are a good predictor of future graident updates. This is like saying

> If you were going down a hill the last timestep, you're probably still going down a hill now.

This assumption is true enough to be useful. We can incorporate our gradient descent into a variable called _velocity_. The velocity at this timestep depends on the velocity from the last timestep, plus the gradient. We use a decay coefficient `b` in [0, 1] to incrementally forget past gradients. `b = 0.9` seems quite popular. This means that 90% of the gradient update is based on past gradients, and 10% is based on this gradient.

Given:

- velocities `v_t` and `v_{t-1}`
- decay factor `b`
- loss-function gradient `dL`
- learning rate `a`
- weights `w_t`, `w_{t-1}`

We do our velocity updates as follows:

```python
# Calculate this timestep's velocity
v_t = b * v_{t-1} + (1 - b) * dL
# Update weights with velocity
w_t = w_{t-1} - a * v_t
```

This is a nice method because we can just drop it into our previous gradient descent calculaitons, and all we have to do is keep track of one extra variable per parameter. This method also:

- de-noises our gradient updates
- helps us power over plateus and through small bumps in the loss-surface
- allows us to avoid oscillating between the walls of long, skinny 'valleys'

N.B This model of velocity is not physically realistic and the term "velocity" is just used as a metaphor. Mathematically this is an exponential moving average.

### Adam (Adaptive Moment Estimation)

[Adam](https://arxiv.org/abs/1412.6980) is an extension of the momentum method. We can think of _velocity_ from the momentum method as being an estimate of the mean gradient - the "first statistical [moment](https://en.wikipedia.org/wiki/Moment_(mathematics))". In the Adam method we also incorporate an estimate of the variance of the gradient - the "second statistical moment".

Just like in the momentum method, we keep track of the exponential moving average of the gradient in `m`, and the gradient squared in `v`. The gradient and squared gradient each have a separate decay term, `b1` and `b2`.

Given:

- timestep `t`, initialized at 0
- epsilon `e`, set to some small number, like 1e-8
- gradient mean `m_t` and `m_{t-1}`
- gradient variance `v_t` and `v_{t-1}`
- decay factors `b1` and `b2`
- loss-function gradient `dL`
- learning rate `a`
- weights `w_t`, `w_{t-1}`

We do our Adam updates as follows:

```python
# Advance timestep
t += 1

# Calculate this timestep's gradient's mean
m_t = b1 * m_{t-1} + (1 - b2) * dL
# Correct initialization bias
m_t = m_t / (1 - b1**t)

# Calculate this timestep's gradient's uncentered variance
v_t = b2 * v_{t-1} + (1 - b2) * dL**2
# Correct initialization bias
v_t = v_t / (1 - b2**t)

# Update weights with our gradient mean / variance estimates
w_t = w_{t-1} - a * m_t / (sqrt(v_t) + e)
```

The Adam paper claims that the following hyperparameter settings make a good default:

- alpha 1e-3
- beta1 0.9
- beta2 0.999
- epsilon 1e−8

The desired outcome from Adam is:

- when the gradient history is high variance, slow down
- when the gradient history is uniform, speed up

Consider:

- when the gradient is constant, `m_t` and `v_t` cancel out, giving us the learning rate
- when the gradient is constantly changing, `m_t` will be small, whereas `v_t` will remain large due to the square operation, giving us a small step size

A nice property of Adam is that the size of the effective step-size will be approximate bounded by the chosen step-size alpha / `a`.

An issue with Adam is that our mean and variance terms will be initalized to zero, which biases our `m` and `v` values to zero. We can counteract this bias by dividing by `(1 - b**t)`, which boosts the gradient when `t` is small.

### Adam with Weight Decay

In classic gradient descent, applying L2 regularization to the loss function is equivalent to decaying the weights as follows:

```python
delta = learning_rate * gradient
decay = learning_rate * decay * weight
weight = weight - delta - decay
```

That's not the case with ADAM.  TLDR - applying L2 regularization to the loss function isn't a good idea, just use the weight decay formulation directly.

Sources:

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [fast.ai - Adam weight decay](https://www.fast.ai/2018/07/02/adam-weight-decay/)
- [ruder.io](http://ruder.io/optimizing-gradient-descent/)
