## Replay memory:

`namedtuple`:
    - a simple, lightweight data structure similar to a class, but without the overhead of defining a full class
    - they contain keys similar to dictionaries, that is hashed to a particular value
    - similar to structures in c++

`object` passed in as arg when creating a class is optional
    - It is passed by default in python3

`deque`: doubly ended queue
    - 'queue': linear data structure that stores items in FIFO manner
    - 'deque': supports appending and popping at both ends of the queue
    - deque is preferred over a list when we need quicker append and pop operations from both ends of the container

`*args`, `**kwargs`:
    - are used to allow functions to accept an arbitrary number of arguments
    - useful when designing functions that need to handle a varying number of inputs
    - 'args' : non-keyword arguments
    - 'kwargs': keyword args

`random.sample()`:
    - used for random sampling without replacement

## Q-network

`torch.nn.functional`
    - has convolution functions, pooling functions, attention mechanisms, non-linear activation functions (ReLU), dropout functions, sparse functions, distance functions, loss functions, vision functions, dataparallel functions (multi-GPU, distributed)

`torch.nn.functional.relu`
    - applies rectified linear unit function element wise

`gradient calculation`
    - it is essential for training neural networks using techniques like backpropagation
    - it involves calculating how much each parameter in the network contributes to the final output's error

`torch.no_grad()`
    - disables gradient calculation. disabling gradient calculation is useful for inference
    - *inference*: during inference we are primarily interested in the model's predictions for given inputs. we will not update the model's parameters during inference
    - gradient calculation is unnecessary for inference. it is required only during the training phase to adjust the model's parameters

## Q-network Parameters
- `state_dict`
  - learnable parameters (weights and biases) are stored in `model.parameters()`
  - a state dict is simply a dictionary that maps each layer to its parameter tensor
- `AdamW`
  <!-- - TODO -->


## Optimize model:

`zip`
    - combines multiple iterators such as lists, tuples, string and dict into a single iterator of tuples
    - first element in each passed iterator is passed together and so on
    - we can also reverse the operation using the * operator
    - Eg:

```bash
    [Transisiton(state='state3', action=3), Transisiton(state='state4', action=4), Transisiton(state='state2', action=2)] 
        
    -> 
        
    Transisiton(state=('state3', 'state4', 'state2'), action=(3, 4, 2))

```

`map`
    - is used to apply a given function to every iterable such as list, tuple and returns a map object which is an iterator

`torch.cat`
    - concatenates the given sequence of tensors into the given dimension
    - all tensors must have the same shape except in the concatenating dimension or be a 1D empty tensor with size (0,)

`passing tensor directly into the DQN object`
  - passing tensor directly into the DQN object calls the __call__() method of the nn.Module class which we inherited
  - the __call()__ function calls your forward function

`torch.gather`
    - gathers values along an axis specified by dim
    - gathers values from a tensor in the shape of a given tensor(as a mask)
Eg:
```bash
a = 
tensor([[ 9.50,  1.38, -1.41, -2.17],
        [-3.56,  3.51, -4.78, -6.43],
        [ 2.71,  5.40, -5.57, -9.11],
        [-5.61, -4.02,  3.26,  5.99],
        [ 2.61,  3.43, -1.39, -5.92],
    ])

b = 
tensor([[0],
        [1],
        [0],
        [0],
        [1],
    ])

c = a.gather(1, b) # 1 - index from rows, 0 - index from columns

c
tensor([[ 9.5000],
        [ 3.5100],
        [ 2.7100],
        [-5.6100],
        [ 3.4300],
    ])
```

`torch.no_grad()`
    - disables gradient calculation

## Main training loop:

`unsqueezing`: returns a new tenor of one dimension, inserted at the specified location

Eg:
```bash
    [-0.00483294 -0.010573 0.03723028 -0.03453089]

    ->

    tensor([[-0.0048, -0.0106,  0.0372, -0.0345]], device='cuda:0')
```