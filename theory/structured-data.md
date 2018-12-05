### Structured Data

Unstructured (all the things in the data are the same kind of thing)
    - image
    - audio
    - natural language text

Structured data
    - profit / loss statement
    - survey results

We have different type of data:
    - continuous quantities (distance, k.m)
    - discrete categories (type of fruit, brand of soap)
    - ambiguous (year, day of week)

Quantities are easily represented as a float. We use quantities when the difference or ratio between different values are meaningful to us. Otherwise we use categories. Categories must be encoded somehow. One way to do it is to encode each category as a one-hot vector.

We can also reserve a null category "unknown", for when we see a category that we haven't seen before.

At a high level, using `fastai`, we want to
    - identify our categorical and continuous variables
    - whack it all in a dataframe
    - pick validation rows
    - create a ModelData instance
    - define category embedding sizes
    - get a `learner` instance
    - train the learner

Jeremy isn't sure how to do data augmentation on this data, but he thinks it's possible in theory.

### Preparing Continuous Data

It's a good idea to normalize continuous input data. This prevents our weights from having to be very large or small, it prevents one feature from dominating activations with initial random weights. One method for normalization is to assume a normal distribution, subtract the mean and divide by the standard deviation. This would work well for human height:

[176cm, 120cm, 190cm, ...] ==> [0.0, -7.5, 1.9, ...]

This may not work as well for features that have a different underlying distribution, like a power law.
