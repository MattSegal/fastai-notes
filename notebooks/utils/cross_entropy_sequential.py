# Create custom loss function
class CrossEntropySequential:
    def __init__(self):
        self.ce =  nn.CrossEntropyLoss()

    def __call__(self, inputs, target):
        """
        Reshape targets and inputs before passing to the loss function.
        
        inputs: batch x seq x vocab => batch*seq x vocab
        target: batch x seq         => batch*seq
        """
        vocab_size = inputs.shape[2]
        flat_inputs = inputs.view(-1, vocab_size)
        flat_target = target.view(-1)
        return self.ce(flat_inputs, flat_target)