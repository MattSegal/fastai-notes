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

Also plot a _confusion matrix_, which answers "how did my label go about classifying false positives / negatives?"

Eg. in the below case the classifier:

- misclassified A into (B, C) rarely and randomly
- never misclassified B
- misclassified C often into A and rarely into B

```

true label

    +------+------+------+
 A  |  98  | 1    | 1    |
    +------+------+------+
 B  |  0   | 100  | 0    |
    +------+------+------+
 C  | 20   | 1    | 79   |
    +------+------+------+
       A      B      C

        predicted label

```

interesting online training info

- epoch
- loss on training set
- loss on validation set
- classification accuracy

if our training loss < validation loss we may be overfitting - learning some feature which doesn't generalize. We may need to:

- get more training data
- decrease DOF of the model
- add regularization
- augment what data we have

if our validation loss < training loss then we may be underfitting, in which case we may need to:

- train for longer
- decrease step size near minima
- tweak model architecture

### Learning Rate

A steeper gradient suggests that we're further from the minimum.

If

- alpha is too small, slow convergence
- alpha is too large, divergence

Alpha is the hyperparameter with the biggest impact on training for the fastai library, other hyperparameters (regularization?) are abstracted away.

How do we choose alpha? Can use "guess and check", but "Cyclical Learning Rates for Training Neural Networks" demonstrates an algorithmic approach. The fastai learning rate finder implements this paper. The general idea is to

- pick a tiny learning rate
- exponentially increase it
- graph the loss vs learning rate
- pick the learning rate that maximizes fall in loss (why not minimum loss?)

### Overfitting

Model is too fit to the training data, loses generalization to new examples. This is a risk when the model degrees of freedom is >= the amount of training data. If your model is performing "much better" on the training set than on the validatio set, then that suggests that you model is overfitting and will not generalize.

How do we prevent overfitting?

- Dumb down model (eg. fewer DOF)
- Get more training data
- Augmentation current training data

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

### Precomputing

Precomputing is where we train our CNN a dataset and learn the weights for all layers. The earlier layers typically learn quite general abstractions that can be re-used later, eg:

- Layer 1: edges, gradients
- Layer 2: corners, curves, circles
- Layer 3: shapes, patterns
- Layer 4: legs, concentric circles,
- Layer 5: human faces, bird legs, dog faces, lizard eyes

When pre-computing, we "freeze" the weights on all but the last layer, and then re-train the network on some new data. This allows us to make diverse use of a single trained network. If we decide to augment the training data, we may need to unfreeze the earlier layers because the low level structure of the images may have changed.

When we train the unfrozen layers, we may want to apply different learning rates to different layers. The idea is that earlier layers (1, 2, 3) have learned more generalizable low level concepts and probably do not need lots of updating, so we choose lower learning rates to earlier layers and higher learning rates for deeper layers (4, 5). FastAI allows us to pass in an array of learning rates to `learn.fit`. This is not commonly named, but can be thought of as _differential learning rates_.


### Lesson 1 Summary

How to train a real good image classifier from a pre-trained CNN using `fastai`:

- Enable data augmentation, set `precompute=True`
- Use `lr_find()` to find highest learning rate where loss is still clearly improving
- Train last layer from precomputed activations for 1-2 epochs
- Train last layer with data augmentation (i.e. precompute=False) for 2-3 epochs with cycle_len=1
- Unfreeze all layers
- Set earlier layers to 3x-10x lower learning rate than next higher layer (depending on similarity to pre-trained dataset)
- Use `lr_find()` again
- Train full network with cycle_mult=2 until over-fitting or you get bored

### Example - dog preeds

Train with 224 x 224 images then use 299 x 299 size images:

- train on small images for a few epochs
- switch to bigger images and continue training

Apparently it works with "fully conolutional" architectures which can handle arbitrarily sized input images. This apparently helps prevent overfiting... somehow.


Some pre-trained networks:
    - resnet34
    - resnext50 - trains longer and consumes more memory

### AWS

There is a fastai pt1 v2 AMI on AWS - p2.xlarge
