### Dropout

Dropout is a technique where we set a given activation to 0 with some probabiility `p`.

For example a dropout layer with `p=0.5` would, on average, cause half the activations that pass through that layer to be set to 0. This is done randomly for each minibatch. You may need to signal boost the non-deleted activations (eg if `p=0.5`, then boost remaining activations by 2).

This is used to prevent overfitting. If the network relies some activation has learned to represent some _exact_ example, and that activation is 'dropped out', then the network will need to find how to use its other activations to represent that example. This technique aims to make the network more robust.

This technique is used by the `fastai` library, and it explains why you can sometimes see validation loss that is _better_ than training loss. The training loss is calculated using dropout, but the validation loss is calculated without dropout.
