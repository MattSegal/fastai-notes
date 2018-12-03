# Cross Entry Loss

#### Surprisal

If I see a result that I thought was unlikely, then I would be surprised. On the other hand, if I see a result that I thought was likely, then I would not be surprised.

We can formalise this as _surprisal_ - the degree to which you are surprised to see the result.

For the ith outcome in a sequence, with a probability y_i, we can quantify our surprisal at that outcome.

If y_i = 0, then we will experience infinite surprise when it happens.
If y_i = 1, then we will experience no surprise when it happens.
If y_i is somewhere in betweem and it happens, we can measure our surpise as follows:

s = ln(1 / y_i)

This surprisal value corresponds to how many bits we should use in a message describing a sequence of outcomes.
Outcomes that are more common, and less surprising should use fewer bits, whereas rarer, more surprising outcomes should use more bits.
We can also think of surprisal as a measure of the information gained upon learning the outcome.

For example imagine we draw fruit from a barrel and we need to encode our results as efficiently as possible in binary. The underlying distribution is:

- apple 60%
- orange 20%
- mango 10%
- pear 5%
- kiwi 2%
- durian 1%

What's the most efficient way to record our observations? Eg (apple, apple, apple, pear, orange, apple, orange, apple, ...)

For each outcome we can calculate our surprisal:

- `s_apple  = ln(1 / 0.6)  = 0.5`
- `s_orange = ln(1 / 0.2)  = 1.6`
- `s_mango  = ln(1 / 0.1)  = 2.3`
- `s_pear   = ln(1 / 0.05) = 3.0` 1
- `s_kiwi   = ln(1 / 0.02) = 3.9`
- `s_durian = ln(1 / 0.01) = 4.6`

We then want to generate a uniquely-decipherable [prefix code](https://en.wikipedia.org/wiki/Prefix_code) that describes what we've seen. We can use [huffman coding] to do this:

- `apple   1    `
- `orange  01   `
- `mango   001  `
- `pear    0001 `
- `kiwi    00001`
- `durian  00000`

So "apple apple mango kiwi" would be "1100100001", or the more likely "apple apple orange apple" would be "11011".

The difference in the code length is approximately the same as the difference in our surpisal values. For example the durian + kiwi codes are 4 bits longer than the apple codes, which is approximate the difference between `s_durian` and `s_apple`.

#### Entropy

So say we draw a bunch of fruit from the barrel and generate a sequence of outcomes. How much surprise do we _expect_ per fruit? We can calculate the average expected surprise of a an observation, drawn from an underlying distribution with outcomes `x_1, x_2, ..., x_n` and probabilities `p_1, p2, ..., p_n` as:

```
H   = p_1 * s_1         + p_2 * s_2         + ... + p_n * s_n

    = p_1 * ln(1 / p_1) + p_2 * ln(1 / p_2) + ... + p_n * ln(1 / p_n)

    = -(p_1 * ln(p_1)   + p_2 * ln(p_2)     + ... + p_n * ln(p_n))

    = -1 * sum_i (p_i * ln(p_i))
```

The expected surprisal per observation `H` is the entropy of the stochastic sequnce. It is, on average, how many bits of information we expect to learn per observed outcome. In a sequence with high entropy, we are often quite surprised by what we see and learn a lot from each observation. In a sequence with low entropy, we are rarely surprised much and learn little per outcome.

YOU ARE CONFUSING THE SEQUENCE WITH THE ENCODING OF THE SEQUENNCE YA DUCE


If we achieve an optimal encoding in our barrel of fruit example, then we will


Sources:

- [rdipietro.github.io](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/)
- [plus.maths.org](https://plus.maths.org/content/information-surprise)
