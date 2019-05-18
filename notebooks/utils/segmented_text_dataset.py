# Updating our Dataset
class SegmentedTextDataset(TextDataset):
    def __len__(self):
        """
        How many samples there are in the dataset.
        """
        return len(self.text) // self.num_chars - 1

    def __getitem__(self, idx):
        """
        Get item by integer index,
        returns (x_1, x_2, ..., x_n), (x_2, x_3, ..., x_{n+1})
        """
        i = idx * self.num_chars
        inputs = torch.tensor([self.char_to_idx[c] for c in self.text[i:i + self.num_chars]])
        label = torch.tensor([self.char_to_idx[c] for c in self.text[i + 1:i + 1 + self.num_chars]])
        return inputs, label