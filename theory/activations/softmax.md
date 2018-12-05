### Softmax Activation

Softmax is nice because it always produces a set of numbers [0, 1] with a sum of 1, which is easy to interpret as a probability distribution over possible answers.

For the softmax of element i of array x with length n
```
softmax_i = e^x_i / ( sum_{k=1}^n e^k )
```

Because the e^x blows up small differences in the inputs, softmaxes tend to produce one big result, with the others relatively small.
