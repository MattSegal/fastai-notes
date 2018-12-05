### Rectified Linear Unit (ReLU) Activation

Relus are the equivalent of:

```python
def relu(x):
    return x if x > 0 else 0

```

They're cheap to compute and don't have a vanishing gradient problem, although you're kind of screwed if a bunch of your relus get stuck at activation 0, because then you have _zero_ gradient. They're quite popular.
