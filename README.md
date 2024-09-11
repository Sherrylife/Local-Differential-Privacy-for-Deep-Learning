# Local Differential Privacy for Deep Learning
The repository contains the code to reproduce the paper ["Local Differential Privacy for Deep Learning" (IOTJ 2019)](https://ieeexplore.ieee.org/document/8894030).

## Requirement
To run the codes, first you need to install the following dependency by running the code
```python
pip install -r requirement.txt
```

## Training
```python
python main.py --dataset mnist \ # dataset name
--alpha 5 \ # privacy budget coefficient
--eps 10 \ # privacy budget
--n 2 \ # number of bits for the whole number of the binary
--m 3 \ # number of bits for the fraction of the binary representation
--epochs 2 # training rounds
```

## Note
* Running with the default configuration we can obtain a model with 93% accuracy on MNIST datset.
    The training log can be found in the ```logs``` folder.
* It seems that the implementation of UER algorithm needs a lot of time, so the total training is very slow.
* It seems that the LATENT algorithm only works well using simple MLP / CNN network in small dataset like MNIST. I double 
 whether this algorithm can be applied to more popular network like ResNet and Transformer.