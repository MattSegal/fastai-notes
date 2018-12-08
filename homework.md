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

Kaggle comp [here](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space)

Multi label classification task, where each image may have many labels. The fastai library does all the "one hot encoding"

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

### Grocery Sales Forecasting (Rossman)

Kaggle comp [here](https://www.kaggle.com/c/rossmann-store-sales). Rossmann is challenging you to predict 6 weeks of daily sales for 1,115 stores located across Germany.

Submissions are evaluated on the Root Mean Square Percentage Error (RMSPE). The submission files should be csv that maps a store ID to a sale amount in dollars:

```
Id,Sales
1,0
2,0
...
```

The files are:

- train.csv: historical data including Sales
- test.csv: historical data excluding Sales
- sample_submission.csv: a sample submission file in the correct format
- store.csv: supplemental information about the stores

Overall, the experience was mixed. The good:

- I decided not to incorporate Google trends or weather data into my dataset
- I saw a lot of pandas tricks by replicating the data cleaning:
    - merging data in different CSVs using dataframes
    - saving dataframes to feather files
- I saw a few cool data prep tricks:
    - extracting features from timestamps
    - filling in null values
    - "rephrasing" date columns, eg: "open since date" => "num days open"
    - removing data that should never appear (eg. < 0)
    - transforming discrete events into continuous "days before" and "days after" fields
    - adding rolling sums for multi day events

The bad:

- 99% of the time was spent wrangling data
- My model achieved way worse performance than the demo model and I don't know why, and I can't really be bothered finding out for this one.
