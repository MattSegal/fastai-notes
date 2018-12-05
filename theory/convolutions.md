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

