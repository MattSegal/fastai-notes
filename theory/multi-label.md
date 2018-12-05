### Multi Label Classification

Some images have single labels (multiple labels), but some have _many_ labels, eg the kaggle planet dataset "clear day, river, forest", and we want to be able to predict all these things.

Softmax is not good at this! It wants to pick _one_ thing.

With softmax we'd receive a class index for our dataset (eg 3 of 5 classes) and the label passed to the network would be a "one hot encoded" vector, ie [0, 0, 1, 0, 0]

For multi label encoding we could receive a list of class indexes (3, 5) and turn it into a vector  [0, 0, 1, 0, 1]

We can use sigmoid instead of softmax, because we get an output in the range (0, 1), which we can interpret as a probability. Each sigmoid output is independent of the others, so we can have multiple predictions for one input.
