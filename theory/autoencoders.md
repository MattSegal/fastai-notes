## Autoencoders

Start with some input vector. Run this vector through a neural net, which has to output the input vector at the end. In at least one intermediate layer, have fewer activations than there were inputs.

Eg. 100 variable input is fed into a neural net with a length 10 activation vectorin the middle somewhere, which then eventually outputs a 100 variable output, which should be as close as possible to the input.

We can then use the middle vector as a "summary" of the input vectors, which could be used as a search index, or the input into another neural net, or as an embedding.

This is a dumb approach that seems to work.
