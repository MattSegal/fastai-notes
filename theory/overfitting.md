# Over and under fitting

When we're training our model it's good to keep track of the:

- epoch
- loss on training set
- loss on validation set
- classification metric (eg. accuracy)

The training loss is being used for model updates, whereas the validation loss is not. Instead, the validation loss is used to judge whether the model is over or underfitting.

Overfitting is when our model agrees closely with training examples, but does not perform well on other data from the same underlying distribution. We can think of this as being a high variance" type of error.

Underfitting is when our model is too simple or untrained to express the necessary complexity to perform well on either the training set or the test set. We can of this as being a "high bias" type of error.


### Overfitting

This is where the model is too "fit" to the training data, and the issue is that it loses generalization to new examples. This is a risk when the model degrees of freedom is >= the amount of training data. If your model is performing "much better" on the training set than on the validation set, then that suggests that you model is overfitting and will not generalize.

An extreme example of overfitting is if the model has just rote memorized the training data, in that case, it will experience zero training loss, but will not be able to make any meaningful predictions on the validation or test set.

How do we prevent overfitting?

- Dumb down model (eg. fewer DOF)
- Get more training data
- Augmentation current training data
- Use dropout
- Add regularizaton (TODO)
- Use weight decay (TODO)

Possible image augmentations:

- zoom
- rotate
- flip
- change colors
- translate
- adjust brightness
- adjust contrast

We want to generate new examples from our current data that preserve the meaning of what we want to learn. For example in letter recognition, zooming in or translating the letter preserves the meaning, where as flipping it in the vertical axis may destroy it (eg a backwards 'F'). As such, different types of image will have augmentation transforms that are appropriate and inappropriate.

We can also do "test time augmentation", which is where we classify the image, plus several randomly augmented versions of the image. We use the average result for our answer. This prevents cases where we, for example, accidentally crop out an important part of the image. May improve accuracy.

We can also think about the shape of our local minima wrt overfitting. If we descend into a steep, sharp "crevice"-like minima, then small changes to the weights in any direction will climb the walls of the loss function and our results will suck. Conversely, small changes to the underlying dataset will cause the loss function to shift and we'll also be climbing the walls of the trench.

If our minima was more of a broad valley, then our loss will be robust to small pertubations in the weights/data and will generalize better. So it's in our interest to prefer broad, smooth minima to skinny, spiky minima. _Stochastic gradient descent with restarts (SGDR)_ is a technique that helps to escape these spiky trenches. In SGDR we "anneal" the learnign rate (decrease over time), but periodically spike it up, in order to escape spiky minima. If we spike the learning rate in a smooth valley, we won't escape it. (decrease learning rate according to a cosine curve every mini-batch, reset it to initial value after `cycle_rate` epochs)

Another method to prevent getting stuck in spiky minima is to run "ensemble" models, where we randomly initialize many models and train them, hoping that one of them will end up in a robust local minima, vs an overfit crevice. OP reckons that SGDR works better than ensemble models.

Another idea is to "snapshot" the weights every time we spike the learning rate, which will generate a different kind of ensemble.

### Underfitting

If our validation loss < training loss then we may be underfitting, in which case we may need to:

- train for longer
- decrease step size near minima
- tweak model architecture
