### Sigmoid Activation

Sigmoid activation is useful because the output is bounded between (0, 1). But there are lots of possible functions with those bounds, so what's up with sigmoid specfically?

For and input scalar `x`, the sigmoid function is:

```
sigmoid(x) = e^x / ( 1 +  e^x )
```

Where does the sigmoid come from? It's derived from the study of odds.

If we have an event with probability p, then the odds of that event are `odds(p) = p / (1 - p)`.

As p varies [0, 1] the odds vary [0, +inf]

For example if p = 0.2, odds are (0.2 / 0.8) = 4, which gives us 1:4 odds
For example if p = 0.1, odds are (0.1 / 0.9) = 9, which gives us 1:9 odds

The "logit" function is the logarithm of the odds `logit(p) = log(odds(p)) = log(p / (1 - p))`.
It is useful, in part, because it maps a limited range [0, 1] onto all the real numbers.
As p varies [0, 1] the odds varies [-inf, +inf], with most sensible values of in [0.001, 0.999] mapping to about [-5, 5]

The sigmoid is the  inverse of the logit function.

```
logit(p)        = log(p / (1 - p))
x               = log(p / (1 - p))
e^x             = p / (1 - p)
e^x - p*e^x     = p
e^x             = p + p*e^x
e^x             = p*(1 + e^x)
p               = e^x / ( 1 +  e^x )
sigmoid(x)      = e^x / ( 1 +  e^x )
```

#### Why Sigmoid?

So... why use this as an activation function?

TODO

In general our activation function for classfication needs to map some input value x from [-inf, inf] to [0, 1]


#### Why Not Sigmoid?

Why not use sigmoid? Sigmoid activations can suffer from a "vanishing gradient" problem. The sigmoid gradient is greatest around an activation of 0.5, but becomes flatter as it tends towards 0 or 1. The gradient for very high or low ouput activations (eg, 0.0001, 0.999) becomes small, making it harder for us to make meaningful updates to our weights when learning, since the gradient is an input to our gradient descent updates. It also means that at extreme output values, large changes to the input will not change the output by much.
