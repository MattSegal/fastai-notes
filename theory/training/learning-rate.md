### Learning Rate

A steeper gradient suggests that we're further from the minimum.

If

- alpha is too small, slow convergence
- alpha is too large, divergence

Alpha is the hyperparameter with the biggest impact on training for the fastai library, other hyperparameters (regularization?) are abstracted away by `fastai`.

How do we choose alpha? Can use "guess and check", but "Cyclical Learning Rates for Training Neural Networks" demonstrates an algorithmic approach. The fastai learning rate finder implements this paper. The general idea is to

- pick a tiny learning rate
- exponentially increase it
- graph the loss vs learning rate
- pick the learning rate that maximizes fall in loss (why not minimum loss?)
