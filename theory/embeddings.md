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

### Interpreting Embeddings

Intepreting embeddings seems to be mostly informed guesswork. You can rank or cluster your data based on their embedding values and impose your own interpretation on them, hopefully with some domain knowledge.

The bias term in the embedding is maybe the easiest parameter to interpret. It captures features about the entitiy that are not well-explained by the input features. For example, the bias of a movie embedding in a movie rating predictor would capture whether a movie is widely liked by everyone, or widely reviled, regardless of who is watching it. In the movie rating case, we could rank all the movies by their embedding bias and interpret the movies with the highest bias as the most widely liked. It's not clear to me whether this method has an advantage over just ranking movies by their average rating.

You can also use principal component analysis (PCA) on all the embedding parameters to produce a small number of variables which explain most of the variance between different entities. This is done by taking some linear combination of the embedding parameters (I think?).

This is handy because it's hard to visualize a 50 dimensional embedding vector, but if we project that vector onto a 2D space then we can look at it on a graph. We may be able to inspect the clusters of embeddings and interpret the values of the low-dimensional representation.
