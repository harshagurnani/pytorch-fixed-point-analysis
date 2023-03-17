# Fixed Point Analysis

Building my own Fixed Point Analysis for Recurrent Neural Network in Pytorch ... In progress (NOT USEABLE YET)

Modified from original code [here](https://github.com/tripdancer0916/pytorch-fixed-point-analysis)

To use the code, create a model of the `proxyRNN` class and use the `FixedPoint` object to analyse this network. This doesn't come with training code (maybe in future). So you have the option to load the weights of some pretrained model yourself. 
This has the flexibility to have your own task specific architecture and custom training, and as long as you can specify it in a simplified rnn for analysis, this code works.

### Dynamics assumed:
Main point of divergence is in the dynamics:
The hidden voltage or state variable has all the dynamics, and the observed "rate" is just via a point non-linearity:
```
dx/dt = - x  + W_in * X_in   + W_rec * r
r = ReLU(x). [or tanh(x)]
```

This means to do initialise to start the fixed point search, you need to specify both x and r. Due to the non-uniqueness of the inverse of ReLU or tanh nonlinearity, ideally you have saved these values in your simulation.


This code is being used to clamp feedback and use as constant inputs to find fixed points under those cue+feedback inputs.
