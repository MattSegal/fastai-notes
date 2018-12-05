# Metrics

TODO:
    - The Area Under an ROC Curve (classification)
    - mean absolute error (MAE) (regression)
    - mean squared error (MSE) (regression)
    - R squared (regression)

Metrics are used to score our predictions. We do not use them directly for our loss function because they may not provide an ideal error signal, but they are useful for evaluating how a model is performing.

Some metrics are based on the _confusion matrix_, which answers "how did my label go about classifying false positives / negatives?"

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

The confusion matrix is also nice to plot and eyeball after training. After training, you may also want to manually check some:

- correctly classified examples
- incorrectly classified examples
- most correctly classified examples
- most incorrectly classified examples
- most incorrect by class
- most correct by class

### Accuracy

Accuracy the simplest metric we can use to score models.

We can thing of accuracy as `correct predictions / total predictions`

This score is a function of the confusion matrix.

### F-Beta

F-beta is a different function of the confusion matrix.

It considers both the "precision" and "recall" of the test. These can be explained in terms of a truth / prediction table (aka confusion matrix):

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

> When it's yes, it's definitely yes. When it's no, it's definitely no.

If precision is high and recall is low, then you can be confident that your positive predictions are correct, but not so sure that your negative predictions aren't in fact positive. We will avoid false positives, but we may experience false negatives.

> When it's yes, it's definitely yes. When it's no, we're not so sure.

If recall is high and precision is low, then you're confident that your negative predictions are correct, but you're not so sure that your positive predictions aren't in fact negative. We will avoid false negatives, but may experience false positives.

> When it's no, it's definitely no. When it's yes, we're not so sure.

In general, the f-beta metric is a function of beta, precision and recall, where increasing beta weighs recall more than precision. The f2 metric weighs recall higher than precision - meaning we're more worried about avoiding false negatives than avoiding false positives.

> We'd prefer to classify a shadow as a cat, rather than miss out on the chance to classify a cat as a cat.
