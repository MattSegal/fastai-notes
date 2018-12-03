# Homework Notes

Notes on homework I'm doing for the FastAI DL for coders course

### Cats vs Dogs Redux

Kaggle comp [here](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition). Doing this to get comfortable using the fastai library on basic image classification tasks.

- Randomly split training images into a training and validation set.
- Used the resnet34 model, trained on ImageNet with a randomly initialized fully connected final layer
- Images are mostly 500x400px
- Used a learning rate finder, picked batch size of 64 via guessing
- Trained at 224x224 using side on transforms, got to 99% accuracy in several epochs
- Resized to 299x299 because it apparently helps avoid overfitting, used a learning rate cycle multiplier to try get a better fit, accuracy still hovers around 99%, so it doesn't seem like further training really helped.
- Didn't bother unfreezing the convolutional layers, since this dataset seems really close to imagenet.
- Training loss is slightly above validation loss, which suggests that we're not overfitting.
- Finally realised how `precompute=True` works - it caches activations from the _frozen_ layers. Definitely worth trying next time!

Results:

- Ranked 250 - 320 / 1300
- Training loss of 0.03
- Validation loss of 0.03
- Test loss of 0.08

### Understanding the Amazon from Space

Kaggle comp [here](https://www.kaggle.com/c/
planet-understanding-the-amazon-from-space)

Multi label classification task, where each image may have many labels.

Had trouble training on 250x250 images (original width). I found good convergence training the final layer ontop of ResNet, but I found little-no improvement when I unfroze the ResNet layers and tried to train it.

The learning rate finder produce a learning rate / loss curve that was very choppy and divergent. Maybe it's because I had trained my model into some sort of sharp local minima, but I'm not really sure.

Downsizing to 64x64 images and training on those seems to work much better when I unfroze the ResNet layers.

I'm also not sure why Jeremy likes changing image sizes for training in general. It seems kind of magical and there isn't a rigorous explanation in the course. I don't like the magic.

The technique was:

- train on 64x64 images (last layer only) (-1 loss)
- train on 64x64 images (entire network) (-0.1 loss)
- train on 128x128 images (last layer only) (-0.01 loss)
- train on 128x128 images (entire network) (-0.01 loss)
- train on 256x256 images (last layer only) (-0.01 loss)
- train on 256x256 images (entire network) (-0.01 loss)

Kaggle wouldn't accept my submission because I'm missing 20k images in my test set (are those the included tiff files?), but I plotted some results and it looks good to me. Ship it!
