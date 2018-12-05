General unstructured notes.

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

