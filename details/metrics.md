# Metrics

We have been using "accuracy" as the metric that we want to optimize.

We can thing of accuracy as `correct / total`

We can think of this score as a function of the confusion matrix.

There are other metrics that we could use as well, eg "f2", which is the "f-beta" function with beta=2.
`fbeta_score` lives in SciPy.

We can pass an array of metrics to `ConvLearner`

What the fuck is this f metric, you may ask? Good question! Is it magic? No it's not!

The "F score" is a measure of a test's accuracy. It considers both the "precision" and "recall" of the test. These can be explained in terms of a truth / prediction table:

```
          what's
what      predicted
happened  +ve           -ve
        +------------+------------+
 +ve    | true +ve   |  false -ve |
        +------------+------------+
 -ve    | false +ve  |  true -ve  |
        +------------+------------+

```

and then

```

 precision  = true positives / all positive predictions
            = true positives / (true positives + false positives)

 recall     = true positive / all positive outcomes
            = true positives / (true positives + false negatives)

```

When precision is high, you can be confident that _if_ you made a positive prediction, then it's likely to be a positive outcome.

When recall is high, then you can be confident that _if_ you see a positive outcome, then it's likely to have a positive prediction associated with it.

If both recall and precision are high, then you expect to see mostly good predictions and who gives a shit then? You're laughing all the way to the bank.

If precision is high and recall is low, then you can be confident that your positive predictions are correct, but not so sure that your negative predictions aren't in fact positive. We will avoid false positives, but we may experience false negatives.

> When it's yes, it's definitely yes. When it's no, we're not so sure.

If recall is high and precision is low, then you're confident that your negative predictions are correct, but your not so sure that your positive predictions aren't in fact negative. We will avoid false negatives, but may experience false positives.

> When it's no, it's definitely no. When it's yes, we're not so sure.

In general, the f-beta metric is a function of beta, precision and recall, where increasing beta weighs recall more than precision. The f2 metric weighs recall higher than precision - meaning we're more worried about avoiding false negatives than avoiding false positives.

> We'd prefer to classify a shadow as a cat, rather than miss out on the chance to classify a cat as a cat.
