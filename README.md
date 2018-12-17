# Fast AI Course

My notes and scripts and whatnot for Fast AI

forums.fast.ai
course.fast.ai

Infrastructure setup:

- AWS Ubuntu 18.04 EC2 on p2.xlarge
- configured using `scripts/config.sh`
- added personal SSH keys
- added kaggle-cli


to read:
- setosa.io/ev/
- https://forums.fast.ai/t/30-best-practices/12344
- A systematic study of the class imbalance problem in CNNs
- initialization of deep networks in the case of rectifiers... or something
- batch normalization
- early stopping - http://www.iro.umontreal.ca/~bengioy/talks/DL-Tutorial-NIPS2015.pdf
- OpenAI's gradient nosie metric for batch sizes - https://blog.openai.com/science-of-ai/
- ADAM with warm restarts - https://arxiv.org/pdf/1711.05101.pdf

to implement
- movielens collaborative filtering with PyTorch
- stochastic gradient descent with restarts
- dropout
- l2 regularization
- data augmentation
- test time augmentation
- inference from saved model on new data
- review `fit` from fastai
- gradient clipping

