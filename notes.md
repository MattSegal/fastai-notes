## Lesson 1

Use 3 data sets

- train, used for updates
- validation, used to test for overfitting online
- test, used for test after training (offline)

After training you can check

- correctly classified examples
- incorrectly classified examples
- most correctly classified examples
- most incorrectly classified examples
- most incorrect by class
- most correct by class

interesting online training info

- epoch
- loss on training set
- loss on validation set
- classification accuracy

### Learning Rate

A steeper gradient suggests that we're further from the minimum.

If

- alpha is too small, slow convergence
- alpha is too large, divergence

How do we choose alpha? Can be "guess and check", but "Cyclical Learning Rates for Training Neural Networks" demonstrates an algorithmic approach.

Alpha is the hyperparameter with the biggest impact on training for the fastai library, other hyperparameters are abstracted away.

### Overfitting

Model is too closely aligned with the training data, loses generalization. This is a risk when the model degrees of freedom is >= the amount of training data.

How do we prevent overfitting?

- Dumb down model
- Get more training data
- Augmentation current training data



