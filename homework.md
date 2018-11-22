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

### Ships in Satellite Imagery

Kaggle comp [here](https://www.kaggle.com/rhammell/ships-in-satellite-imagery/home)
