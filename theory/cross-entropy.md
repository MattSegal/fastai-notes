# Cross Entry Loss

NB: We're using a log of base 2 here, but for training neural networks it doesn't really matter what base we use, since we can extract it as a scaling factor (`log_x(z) = ln(z) / ln(x)`).

### Surprisal

If I see a result that I thought was unlikely, then I would be surprised. On the other hand, if I see a result that I thought was likely, then I would not be surprised.

We can formalise this as _surprisal_ - the degree to which you are surprised to see the result.

For some outcome of a stochastic process, which has a probability p, we can quantify our surprisal at that outcome.

If p = 0, then we will experience infinite surprise when the outcome happens.
If p = 1, then we will experience no surprise when the outcome happens.
If p is somewhere in betweem and it happens, we can quantify our surpise as follows:

s = log(1 / p)

We can think of surprisal as a measure of the information gained upon learning the outcome. This surprisal value, measured in bits, corresponds to how much our uncertainty has been reduced. 1 bit of surprisal reduces our uncertainty by 50%. 2 bits by 75%, 3 by 88% etc.

This value is also useful in encoding observations in messages. Assume we want to describe a sequence of observations using the shortest binary message possible. We can use the surprisal of each outcome to determine how many bits we should use to describe each outcome. Outcomes that are more common, and less surprising should use fewer bits, whereas rarer, more surprising outcomes should use more bits.

For example imagine we draw a sequence of fruit from a barrel, and we need to encode our results as efficiently as possible in binary. The underlying distribution is:

- apple 60%
- orange 20%
- mango 10%
- pear 5%
- kiwi 2%
- durian 1%

What's the most efficient way to record our observations? Eg (apple, apple, apple, pear, orange, apple, orange, apple, ...)

For each outcome we can calculate our surprisal:

- `s_apple  = log(1 / 0.6)  = 0.5`
- `s_orange = log(1 / 0.2)  = 1.6`
- `s_mango  = log(1 / 0.1)  = 2.3`
- `s_pear   = log(1 / 0.05) = 3.0`
- `s_kiwi   = log(1 / 0.02) = 3.9`
- `s_durian = log(1 / 0.01) = 4.6`

We then want to generate a uniquely-decipherable [prefix code](https://en.wikipedia.org/wiki/Prefix_code) that describes what we've seen. We can use the [Huffman coding](https://planetcalc.com/2481/) algorithm to do this:

- `apple   1    `
- `orange  01   `
- `mango   001  `
- `pear    0001 `
- `kiwi    00001`
- `durian  00000`

So the unlikely sequence "apple apple mango kiwi" would be "1100100001", or the more likely "apple apple orange apple" would be "11011".

The difference in the code length for each possible outcome is approximately the same as the difference in our surpisal values. For example the durian + kiwi codes are 4 bits longer than the apple codes, which is approximately the difference between `s_durian` and `s_apple`.

### Entropy

So say we draw a bunch of fruit from the barrel and generate a sequence of outcomes. How much surprise do we _expect_ for each fruit? We can calculate the average expected surprise of an observation. Given each observation is drawn from an underlying distribution of outcomes `x_1, x_2, ..., x_n` and probabilities `p_1, p2, ..., p_n`, we calculate the expected surprisal using a weighted average:

```
H   = p_1 * s_1         + p_2 * s_2         + ... + p_n * s_n

    = p_1 * log(1 / p_1) + p_2 * log(1 / p_2) + ... + p_n * log(1 / p_n)

    = -(p_1 * log(p_1)   + p_2 * log(p_2)     + ... + p_n * log(p_n))

    = -1 * sum_i (p_i * log(p_i))
```

The expected surprisal per observation `H` is the "entropy" of the stochastic sequence. It is, on average, how many bits of information we expect to learn per observed outcome. In a sequence with high entropy, we are often quite surprised by what we see and learn a lot from each observation. In a sequence with low entropy, we are rarely surprised much and learn little per outcome.

For a coin flip we can calculate entropy as

```python
from math import log

def entropy(probs):
    return -1 * sum([p * log(p, 2) for p in probs])

entropy([0.5, 0.5])
# 1
```

So on average we expect 1 bit of surpisal per coin toss. This corresponds with a 50% reduction in uncertainty.

For a biased coin toss where P(heads) = 0.8

```python
entropy([0.8, 0.2])
# 0.7
```

Here we are learning less per coin toss, because we were initially less uncertain about the outcome - we were pretty sure the outcome was going to be heads.

The entropy of a stochastic sequence also gives us a lower bound for the number of bits per outcome we can expect to send in a message. For example, in our fruit sample sequence, the absolute fewest bits we can send per fruit in a lossless message are:

```python
entropy([0.6, 0.2, 0.1, 0.05, 0.02, 0.01])
# 1.6
```

For any encoding scheme, we must encode, on average, _at least_ 1.6 bits per observation. It's impossible to losslessly communicate the sequence of fruit with less than, on average, 1.6 bits per observed fruit.

### Cross Entropy

Entropy tells us the most efficient encoding possible for a given distribution of outcomes. Cross entropy tells us



For our huffman-generated prefix code from above, we can figure out how many bits of surprisal we expect per character:

```python
codes = (
    # name, code, probability
    ('apple', '1', 0.6),
    ('orange', '01', 0.2),
    ('mango', '001', 0.1),
    ('pear', '0001', 0.05),
    ('kiwi', '00001', 0.02),
    ('durian', '00000', 0.01),
)


```



If we achieve an optimal encoding in our barrel of fruit example, then we will


Sources:

- [rdipietro.github.io](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/)
- [plus.maths.org](https://plus.maths.org/content/information-surprise)
