# Cross Entry Loss

### Surprisal

If I see a result that I thought was unlikely, then I would be surprised. On the other hand, if I see a result that I thought was likely, then I would not be surprised.

We can formalise this as _surprisal_ - the degree to which you are surprised to see the result.

For some outcome of a stochastic process, which has a probability p, we can quantify our surprisal at that outcome.

If p = 0, then we will experience infinite surprise when the outcome happens.
If p = 1, then we will experience no surprise when the outcome happens.
If p is somewhere in between and the outcome happens, we can quantify our surpise as follows:

```
s = log(1 / p) = -log(p)
```

N.B: log is log base 2

We can think of surprisal as a measure of the information gained upon learning the outcome. This surprisal value, measured in bits, corresponds to how much our uncertainty has been reduced. 1 bit of surprisal reduces our uncertainty to by 50%. 2 bits by 75%, 3 by 88% etc.

This value is also useful in encoding observations in efficient messages. Assume we want to describe a sequence of observations using the shortest binary message possible. We can use the surprisal of each outcome to determine how many bits we should use to describe that outcome. Outcomes that are more common, thus less surprising, should use fewer bits, whereas rarer, more surprising, outcomes should use more bits.

For example imagine we draw a sequence of fruit from a barrel, and we need to encode our results as efficiently as possible in binary. The underlying distribution is:

- apple 60%
- orange 20%
- mango 10%
- pear 5%
- kiwi 2%
- durian 1%

What's the most space-efficient way to record our observations? Eg (apple, apple, apple, pear, orange, apple, orange, apple, ...). We could use a fixed-length encoding like this:

- `apple   000`
- `orange  001`
- `mango   010`
- `pear    011`
- `kiwi    100`
- `durian  101`

So the unlikely sequence "apple apple mango kiwi" would be "000000010100", or the more likely "apple apple orange apple" would be "000000001000".

We can do better though! For each outcome we can calculate our surprisal:

- `s_apple  = -log(0.6)  = 0.7`
- `s_orange = -log(0.2)  = 2.3`
- `s_mango  = -log(0.1)  = 3.3`
- `s_pear   = -log(0.05) = 4.3`
- `s_kiwi   = -log(0.03) = 5.1`
- `s_durian = -log(0.02) = 5.6`

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
H(p) = p_1 * s_1         + p_2 * s_2         + ... + p_n * s_n

    = p_1 * log(1 / p_1) + p_2 * log(1 / p_2) + ... + p_n * log(1 / p_n)

    = -(p_1 * log(p_1)   + p_2 * log(p_2)     + ... + p_n * log(p_n))

    = -1 * sum_i (p_i * log(p_i))
```

The expected surprisal per observation `H(p)` is the "entropy" of the stochastic sequence. It is, on average, how many bits of information we expect to learn per observed outcome. In a sequence with high entropy, we are often quite surprised by what we see and learn a lot from each observation. In a sequence with low entropy, we are rarely surprised much and learn little per outcome.

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

Here we are learning less per coin toss, because we were initially less uncertain about the outcome - we were pretty sure the outcome was going to be heads to beging with.

The entropy of a stochastic sequence also gives us a lower bound for the number of bits per outcome we can expect to send in a message. For example, in our fruit sample sequence, the absolute fewest bits we can send per fruit in a lossless message are:

```python
entropy([0.6, 0.2, 0.1, 0.05, 0.03, 0.02])
# 1.7
```

For any encoding scheme, we must encode, on average, _at least_ 1.7 bits per observation. It's impossible to losslessly communicate the sequence of fruit with less than, on average, 1.7 bits per observed fruit.

### Cross Entropy

Entropy tells us the most efficient encoding possible for a given distribution of outcomes. _Cross entropy_ tells us the average message length of a particular choice of encoding. Consider our fixed length code:

- `apple   000`
- `orange  001`
- `mango   010`
- `pear    011`
- `kiwi    100`
- `durian  101`

In this code, every message is 3 bits long, so the cross entropy of that code is 3 bits. We know that this is less than optimal, because the entropy of the sequence is 1.7 bits. How do we calculate cross entropy in general? We know the true distribution of fruit coming out of the barrel `p`. We can also infer a "predicted" distrbution from the length of each outcome's code `q`, where `q = 1 / 2^(len(code))`. For example, for a code of length 3, `q=0.125`.

We can then calculate cross-entropy `H(p, q)`

```
H(p, q)     = -1 * sum_i p_i * log(q_i)
```

So for our fixed length code

```python
codes = (
    # code, probability
    ('000', 0.6),   # apple
    ('001', 0.2),   # orange
    ('010', 0.1),   # mango
    ('011', 0.05),  # pear
    ('100', 0.03),  # kiwi
    ('101', 0.02),  # durian
)
cross_entropy = -1 * sum([
    pr * log(1 / 2**(len(code)), 2)
    for code, pr in codes
])
# 3
```

We can also figure out the cross entropy for our huffman-generated prefix code:

```python
codes = (
    # code, probability
    ('1', 0.6),         # apple
    ('01', 0.2),        # orange
    ('001', 0.1),       # mango
    ('0001', 0.05),     # pear
    ('00001', 0.03),    # kiwi
    ('00000', 0.02),    # durian
)
cross_entropy = -1 * sum([
    pr * log(1 / 2**(len(code)), 2)
    for code, pr in codes
])
# 1.75
```

That's pretty good! It's almost optimal, and approx 1.3 bits better per message than the fixed-length code. The _relative entropy_, or _KL divergence_ of this code is 0.05 bits, which is the difference between the entropy and the cross entropy:

```
D_kl(p || q) = H(p, q) - H(p)
```

### Cross Entropy in Machine Learning

We can use cross entropy as a loss function in a classifier Eg, in and animal classification task we can score the following prediction:

```
class       predicted       true
---------------------------------
cat         0.02            0.00
dog         0.30            0.00
fox         0.45            0.00
cow         0.01            0.00
panda       0.25            1.00
bear        0.05            0.00
dolphin     0.01            0.00
```

as follows:


```python
predictions = (
    # predicted, true
    (0.02, 0.00), # cat
    (0.30, 0.00), # dog
    (0.45, 0.00), # fox
    (0.01, 0.00), # cow
    (0.25, 1.00), # panda
    (0.05, 0.00), # bear
    (0.01, 0.00), # dolphin
)
cross_entropy = -1 * sum([
    true_pr * log(predicted_pr)
    for predicted_pr, true_pr in predictions
])
# 1.386
```

NB: It is common to use the natural log by convention in machine learning. For training neural networks it doesn't really matter what base we use, since we can extract it as a scaling factor (`log_x(z) = ln(z) / ln(x)`).

So why do we do this? Apprently we use cross-entropy as loss function because "minimizing the cross entropy is the same as maximizing the likelihood". What does that mean though?

# Maximum Likelihood Estimation

TODO

Sources:

- [Aurélien Géron](https://www.youtube.com/watch?v=ErfnhcEV1O8)
- [rdipietro.github.io](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/)
- [plus.maths.org](https://plus.maths.org/content/information-surprise)
