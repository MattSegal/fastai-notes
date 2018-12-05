### Natural Language Processing

Many subproblems in NLP. One is "language modelling": predict the next word given previous words. Kapathy's CharRNN does this character-by-character, wheras we're going to do it word-by-word.

Jeremy suggests training a language model on some corpus, and then using that model as a pre-trained model which feeds into a classification model. He thinks that fine tuning a pre-trained model is a powerful technique.

We can use a dataset of 50k IMDB movie reviews which are scored as +ve or -ve. At first we will ignore the labels and create a language model.

Text has to be turned into _tokens_ (ie words). There are some differences though, eg wasn't -> ['was', 'n\'t']

The spacy tokenizer is pretty good apparently. We don't use stemming or limitizing, but with no strong opinions on this.

Some words may appear only once or twice in the entire corpus. We don't want to try to learn these because we don't have enough data to do anything meaningful with them. As such, we define a _minimum frequency_ cutoff, where words below the minimum frequency are marked as 'unknown'.

_Vocab_ is the list of unique words that appear in the text, where each word has an index. Eg:

```
vocab = {
    0: '<unknown>',
    1: '<pad>',
    2: 'the',
    3: 'and',
    ...
}
```

So we can map tokens <--> integers.

We have two parameters instead of just "batch size" _batch size_ and _backprop through time_.

In our language model we concatenate all text into one big fuckoff list of tokens. We split that list up into batches.

Saw we had 64e6 tokens, and a batch size of 64. We would partition this list into 64 sections and arrange the data into a 1e6x64 matrix.

For this text: "Confronting the question most commonly asked of the growing number of Americans who support replacing America's uniquely inefficient and immoral for-profit healthcare system with Medicare for All—"How do we pay for it?"—a new paper released Friday by researchers at the Political Economy Research Institute (PERI) shows that financing a single-payer system would actually be quite simple, given that it would cost significantly less than the status quo."

We create this array of tokens:

```
tokens = [
    'Confronting', 'the', 'question', 'most', 'commonly', 'asked', 'of',
    'the', 'growing', 'number', 'of', 'Americans', 'who', 'support', 'replacing',
    'America\'s', 'uniquely', 'inefficient', 'and', 'immoral', 'for', 'profit',
    'healthcare', 'system', 'with', 'Medicare', 'for', 'All', '—', '"', 'How', 'do',
    'we', 'pay', 'for', 'it', '?', '"', 'a', 'new', 'paper', 'released', 'Friday',
    'by', 'researchers', 'at', 'the', 'Political', 'Economy', 'Research', 'Institute',
    '(', 'PERI', ')', 'shows', 'that', 'financing', 'a', 'single', 'payer', 'system',
    'would', 'actually', 'be', 'quite', 'simple', 'given', 'that', 'it', 'would',
    'cost', 'significantly', 'less', 'than', 'the', 'status', 'quo', '.'
]
```
And a batch size of '3', we would partition the tokens into 3 as follows:

```
batches = [
    ['Confronting', 'the', 'question', 'most', 'commonly', 'asked', 'of', 'the', 'growing', 'number', 'of', 'Americans', 'who', 'support', 'replacing', "America's", 'uniquely', 'inefficient', 'and', 'immoral', 'for', 'profit', 'healthcare', 'system', 'with', 'Medicare'],
    ['for', 'All', '—', '"', 'How', 'do', 'we', 'pay', 'for', 'it', '?', '"', 'a', 'new', 'paper', 'released', 'Friday', 'by', 'researchers', 'at', 'the', 'Political', 'Economy', 'Research', 'Institute', '('],
    ['PERI', ')', 'shows', 'that', 'financing', 'a', 'single', 'payer', 'system', 'would', 'actually', 'be', 'quite', 'simple', 'given', 'that', 'it', 'would', 'cost', 'significantly', 'less', 'than', 'the', 'status', 'quo', '.']
]
```

Then with a _backprop through time_ parameter of 4, we would grab a chunk of length 4 from each batch and feed that into the GPU for processing. `bptt` is permuted a little each batch by PyTorch so that we get a little bit of data augmentation.

```
batch = [
    ['Confronting', 'the', 'question', 'most'],
    ['for', 'All', '—', '"'],
    ['PERI', ')', 'shows', 'that'],
]
```

Say we have 35k unique tokens in our vocab. These are categorical labels and we can embed them into an embedding matrix of size 35000xY where each words gets an embedding vector of length Y. Jeremy chooses Y=200, he reckons 50-600 is reasonable.

The underlying architecture is the AWD LSTM language model.

Why aren't we using a pretrained word embedding like word2vec or GloVe? Jeremy things pretrained language models are "more powerful" than using pretrained word embeddings, but that you could incorporate a pretrained word embedding into your language model.

Jeremy pickles the vocab so he can use it later.
