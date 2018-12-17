# Regularization

Regularization is a method we can use to avoid overfitting. In regularization, we penalize overly "complex" models.

### L2 Regularization

In L2 regularization we add an extra term to our loss function, which penalizes models with large weights.

The intuition behind this approach is that large weights suggest that we are boosting the signal of one feature in particular, which is could be a case of overfitting.

We can add L2 regularization by extending our loss function:

```python
decay = 1e-4
L2 = sum([w**2 for w in weights]) / 2
loss = loss + decay * L2
```

Our choice of coefficient `decay` determines how aggresive we want to be in penalizing models with large weights.

L2 regularization is also called _weight decay_, because applying it is equivalent to reducing the weight a little bit in proportion to the size of the weight every timestep, when doing classical SGD updates. That is:

```python
delta = learning_rate * gradient
decay = learning_rate * decay * weight
weight = weight - delta - decay
```

### L1 Regularization

It's the same as L2 but we don't square the weights.
