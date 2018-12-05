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
- Use dropout
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

There is a fastai pt1 v2 AMI on AWS - use a p2.xlarge

### Convnets

A CNN has
    - conv layers: input matrix passed through a kernel/filter
    - pooling layers: input matrix is shrunk down
    - fully connected layers: classical neural net, connecting each input activation to each neuron in the next layer

**Convolutions**

A convolution, in this context, maps a grid of pixels to a single scalar value, which can be thought of as another pixel in a product image.

The "kernel" is a matrix, usually 3x3, which we use to do a dot product on the input matrix to produce the output scalar. The kernel values are learned by the network during training.

After applying the kernel, we then pass the convolution output through a nonlinearity, like a relu. Example

Input - vertical and horizontal edge
```
 0  1  0  0  0
 0  1  1  1  1
 0  1  0  0  0
 0  1  0  0  0
```

kernel - 3x3 (will detect vertical edges)
```
-1  0  1
-1  0  1
-1  0  1
```

apply kernel (out of bounds counts as 0)
```
 2  0  0  0
 2  1  0 -1
 2  0  0  0
 2  0  0  0
```

apply nonlinearity: ReLU (max(0, activation))
```
 2  0  0  0
 2  1  0  0
 2  0  0  0
 2  0  0  0
```

Let's try a different kernel
```
 1  1  1
 0  0  0
-1 -1 -1
```
apply kernel (out of bounds counts as 0)
```
-1 -2 -3 -2
 0  0  0  0
 0  1  2  2
 0  1  0  0
```

apply nonlinearity: ReLU (max(0, activation))
we can see that this has activated for vertical edges
```
 0  0  0  0
 0  0  0  0
 0  1  2  2
 0  1  0  0
```

Convolutional layers can also reduce the size of the input matrix with a "stride" (???)

**Pooling Layers**
Now we have 2 4x4 matrices describing our input image

product of horizontal edge kernel

```
 2  0  0  0
 2  1  0  0
 2  0  0  0
 2  0  0  0
```

product of vertical edge kernel

```
 0  0  0  0
 0  0  0  0
 0  1  2  2
 0  1  0  0
```

We think there's some redundant information in here, so we want to map each these matrices to a 2x2. We use "pooling" to do this. In "max pooling", we break each matrix up into 2x2 grids and take the max of each, producing a matrix that is half as big.

product of horizontal edge kernel (2x2 max pooling)
```
 2  0
 2  0
```

product of vertical edge kernel  (2x2 max pooling)
```
 0  0
 1  2
```

**Fully Connected Layers**

Once we've filtered and pooled out input data a whole lot, we may want to use the result as an input to a fully connected neural network for classification.
Jeremy reckons fully connected layers can be "slow" and prone to overfitting because they have lots of weights. Modern architectures may do something different?

N.B we don't have a ReLU in the last layer. We might use tanh or sigmoid or softmax or some other non linearity for the last layer.

Softmax is nice because it always produces a set of numbers [0, 1] with a sum of 1, which is easy to interpret as a probability distribution over possible answers.

For the softmax of element i of array x with length n
```
softmax_i = e^x_i / ( sum_{k=1}^n e^k )
```

Because the e^x blows up small differences in the inputs, softmaxes tend to produce one big result, with the others relatively small.

### Multi Label Classification

Some images have single labels (multiple labels), but some have _many_ labels, eg the kaggle planet dataset "clear day, river, forest", and we want to be able to predict all these things.

Softmax is not good at this! It wants to pick _one_ thing.

With softmax we'd receive a class index for our dataset (eg 3 of 5 classes) and the label passed to the network would be a "one hot encoded" vector, ie [0, 0, 1, 0, 0]

For multi label encoding we could receive a list of class indexes (3, 5) and turn it into a vector  [0, 0, 1, 0, 1]

We can use sigmoid instead of softmax, because we get an output in the range (0, 1), which we can interpret as a probability. Each sigmoid output is independent of the others, so we can have multiple predictions for one input.

### Dropout

Dropout is a technique where we set a given activation to 0 with some probabiility `p`.

For example a dropout layer with `p=0.5` would, on average, cause half the activations that pass through that layer to be set to 0. This is done randomly for each minibatch. You may need to signal boost the non-deleted activations (eg if `p=0.5`, then boost remaining activations by 2).

This is used to prevent overfitting. If the network relies some activation has learned to represent some _exact_ example, and that activation is 'dropped out', then the network will need to find how to use its other activations to represent that example. This technique aims to make the network more robust.

This technique is used by the `fastai` library, and it explains why you can sometimes see validation loss that is _better_ than training loss. The training loss is calculated using dropout, but the validation loss is calculated without dropout.

### Structured Data

Unstructured (all the things in the dataare the same kind of thing)
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

### Embedding Categorical Data

When we have nothing but continuous variables as an input to our network, then the network input is simple. Eg for a regression problem measuring $ spent on fast food p.a, the inputs:

- salary ($/pa)
- weight (kg)
- age (years)

could be normalized and inserted into an input vector:

```

 feature    value
 -------    -----
 salary     +0.2
 age        -1.0  => 3x100  => ...  => output in $
 weight     +2.3     matrix

```

What if we add a categorical variable describing their favourite AFL team? We would have to one-hot encode it into a 19x1 vector to pass it into a network. If we do this naievely we end up with a lot of sparse features:

```

 feature    value
 -------    -----
 salary     +0.2
 age        -1.0  => 21x100 => ...  => output in $
 weight     +2.3     matrix
 crows      +0.0
 lions      +0.0
 blues      +0.0
 magpies    +0.0
 bombers    +0.0
 dockers    +0.0
 cats       +1.0
 giants     +0.0
 suns       +0.0
 hawks      +0.0
 kangaroos  +0.0
 demons     +0.0
 power      +0.0
 tigers     +0.0
 saints     +0.0
 swans      +0.0
 eagles     +0.0
 bulldogs   +0.0

```
This is not ideal because we've moved from having 300 weights in the first layer to 2100, which is more computationally expensive. This gets even worse for 1000s of categories,

There's also another issue where we've decoupled all the classes in the category from each other. For example some of these teams are from Melbourne, whereas some are inter-state. It would be good to be able to represent that fact, and other common features of football teams somehow. On one hand, if the feature is important, the 21x100 matrix and other downstream weights will learn a distributed representation of the feature, but if we can anticipate that there will be common features in a categorical input, we might as well tweak the network architecture to learn those abstractions early, so that we can use downstream weights to learn higher level abstractions.

An alternative to appending categorical one-hot-encodings is to add an _embedding_ for the AFL club category to the input vector. Let's guess that there are about 4 features that describe each AFL club. What we can do is create a 19x4 lookup table which "embeds" our one-hot encoded category vector into these 4 features. Now our network looks like this:

```

                                feature    value
                                -------    -----
                                salary     +0.2
 club       active              age        -1.0  => 7x100 => ...  => output in $
 ----       ------              weight     +2.3     matrix
 crows      +0.0    => 19x4 =>  club_1     +0.5
 lions      +0.0       lookup   club_2     -1.1
 blues      +0.0                club_3     +0.2
 magpies    +0.0                club_4     +0.8
 bombers    +0.0
 dockers    +0.0
 cats       +1.0
 giants     +0.0
 suns       +0.0
 hawks      +0.0
 kangaroos  +0.0
 demons     +0.0
 power      +0.0
 tigers     +0.0
 saints     +0.0
 swans      +0.0
 eagles     +0.0
 bulldogs   +0.0

```
Now we are using 700 weights + a 76 weight lookup table instead of 2100 thanks to our embedding matrix, and we are also representing each item as a concept in 4 dimensional space, rather than a boolean flag. This is even more userful when we have categories with thousands of values, like zip codes. So how big do we make our embedding matrix? Jeremy has a rule of thumb: "cardinality divided by 2, no greater than 50".

### Embedding Time Data

We can use our domain knowledge to create hard-coded embeddings. For example in `fastai` there is an function to extract information from a date. From a single date we can extract:

- Year
- Month
- Day
- Day of week
- Day of year
- Is month end
- Is month start
- Is quater end
- Is quater start
- Is school holiday
- Is national holiday

### Natural Language Processing

Many subproblems in NLP. One is "language modelling": predict the next word given previous words. Kapathy's CharRNN does this character-by-character, wheras we're going to do it word-by-word.

Jeremy suggests training a language model on some corpus, and then using that model as a pre-trained model which feeds into a classification model. He thinks that fine tuning a pre-trained model is a powerful technique.

We can use a dataset of 50k IMDB movie reviews which are scored as +ve or -ve. At first we will ignore the labels and create a language model.

Text has to be turned into _tokens_ (ie words). There are some differences though, eg wasn't -> ['was', 'n\'t']

The spacy tokenizer is pretty good apparently. We don't use stemming or limitizing, but with no strong opinions on this.

Some words may appear only once or twice in the entire corpus. We don't want to try to learn these because we don't have enough data to do anything meaningful with them. As such, we define a _minimum frequency_ cutoff, where words below the minimum frequency are marked as 'unknown'.

_Vocab_ is the list of unique words that appear in the text, where each word has an index. Eg:

```
vocab = {
    0: '<unknown>',
    1: '<pad>',
    2: 'the',
    3: 'and',
    ...
}
```

So we can map tokens <--> integers.

We have two parameters instead of just "batch size" _batch size_ and _backprop through time_.

In our language model we concatenate all text into one big fuckoff list of tokens. We split that list up into batches.

Saw we had 64e6 tokens, and a batch size of 64. We would partition this list into 64 sections and arrange the data into a 1e6x64 matrix.

For this text: "Confronting the question most commonly asked of the growing number of Americans who support replacing America's uniquely inefficient and immoral for-profit healthcare system with Medicare for All—"How do we pay for it?"—a new paper released Friday by researchers at the Political Economy Research Institute (PERI) shows that financing a single-payer system would actually be quite simple, given that it would cost significantly less than the status quo."

We create this array of tokens:

```
tokens = [
    'Confronting', 'the', 'question', 'most', 'commonly', 'asked', 'of',
    'the', 'growing', 'number', 'of', 'Americans', 'who', 'support', 'replacing',
    'America\'s', 'uniquely', 'inefficient', 'and', 'immoral', 'for', 'profit',
    'healthcare', 'system', 'with', 'Medicare', 'for', 'All', '—', '"', 'How', 'do',
    'we', 'pay', 'for', 'it', '?', '"', 'a', 'new', 'paper', 'released', 'Friday',
    'by', 'researchers', 'at', 'the', 'Political', 'Economy', 'Research', 'Institute',
    '(', 'PERI', ')', 'shows', 'that', 'financing', 'a', 'single', 'payer', 'system',
    'would', 'actually', 'be', 'quite', 'simple', 'given', 'that', 'it', 'would',
    'cost', 'significantly', 'less', 'than', 'the', 'status', 'quo', '.'
]
```
And a batch size of '3', we would partition the tokens into 3 as follows:

```
batches = [
    ['Confronting', 'the', 'question', 'most', 'commonly', 'asked', 'of', 'the', 'growing', 'number', 'of', 'Americans', 'who', 'support', 'replacing', "America's", 'uniquely', 'inefficient', 'and', 'immoral', 'for', 'profit', 'healthcare', 'system', 'with', 'Medicare'],
    ['for', 'All', '—', '"', 'How', 'do', 'we', 'pay', 'for', 'it', '?', '"', 'a', 'new', 'paper', 'released', 'Friday', 'by', 'researchers', 'at', 'the', 'Political', 'Economy', 'Research', 'Institute', '('],
    ['PERI', ')', 'shows', 'that', 'financing', 'a', 'single', 'payer', 'system', 'would', 'actually', 'be', 'quite', 'simple', 'given', 'that', 'it', 'would', 'cost', 'significantly', 'less', 'than', 'the', 'status', 'quo', '.']
]
```

Then with a _backprop through time_ parameter of 4, we would grab a chunk of length 4 from each batch and feed that into the GPU for processing. `bptt` is permuted a little each batch by PyTorch so that we get a little bit of data augmentation.

```
batch = [
    ['Confronting', 'the', 'question', 'most'],
    ['for', 'All', '—', '"'],
    ['PERI', ')', 'shows', 'that'],
]
```

Say we have 35k unique tokens in our vocab. These are categorical labels and we can embed them into an embedding matrix of size 35000xY where each words gets an embedding vector of length Y. Jeremy chooses Y=200, he reckons 50-600 is reasonable.

The underlying architecture is the AWD LSTM language model.

Why aren't we using a pretrained word embedding like word2vec or GloVe? Jeremy things pretrained language models are "more powerful" than using pretrained word embeddings, but that you could incorporate a pretrained word embedding into your language model.

L4 2:05:14
