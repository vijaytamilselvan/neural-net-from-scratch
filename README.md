# neural-net-from-scratch
My custom MLP built from scratch using pure Python

 Neural Network from Scratch - Learning Summary

This project is about building a basic neural network from scratch using Python. The goal was to understand how the forward pass, backward pass, loss computation, and parameter updates work without using any machine learning libraries like PyTorch or TensorFlow.

 What I Built

I implemented a multi-layer perceptron (MLP) consisting of:

1. **Neuron**: A single unit that takes multiple inputs, multiplies them with weights, adds a bias, and applies a non-linear activation function (`tanh`).

2. **Layer**: A collection of neurons. Each neuron in the layer takes the same input but has different weights and bias values, so they produce different outputs.

3. **MLP** (Multi-Layer Perceptron): A sequence of layers connected one after another. The output of one layer is passed as input to the next layer, allowing the network to learn more complex patterns.

## Concepts I Learned

- **Forward Pass**: The input gets passed to the network to get the prediction(ypred). Each neuron performs its calculation, and the output flows through the layers.

- **Backward Pass**: After computing the loss between predicted and actual outputs, the `backward()` function computes gradients(p.grad) for each parameter using automatic differentiation. These gradients (With the help of gradient, I can make the change in actual data which are weights and biases) is used to reduce the loss.

- **Parameters**: Each neuron has its own set of parameters – a list of weights and a bias. The `parameters()` method collects all these values so they can be updated during training.(Pytorch parameters() function is quite useful to collect and work on parameters involved in the neural network).

- **Gradient Descent**: After computing gradients, I manually updated the parameters using `p.data += -learning_rate * p.grad`. This gradually reduces the loss over multiple iterations.

- **Training Loop**: I trained the model using multiple input-output pairs. For each iteration, I ran a forward pass to get predictions, computed the loss, performed a backward pass, and then updated the weights and biases.

## Training vs Testing

- During training, I provided input samples (`xs`) along with their expected outputs (`ys`). The model learned to predict the correct output by adjusting its internal parameters to reduce the loss.

- After training, I tested the model using a new input (for example, `[2.0, 3.0, -1.0]`) to check how well it performs on data it has not seen before. This helped evaluate whether the model had truly learned or just memorized the training data.

## Example Inputs Used

```python
xs = [
  [2.0, 8.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 7.0, 8.0],
  [1.0, 1.0, -9.0],
]

ys = [1.0, -1.0, -1.0, 1.0]

