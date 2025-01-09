## Basics

`Markov Decision Process`
- It is a mathematical framework to model decision making problems where the outcomes are pretty random and partly under the control of the decision-maker
- Key components of a MDP are the states, actions, transistion probabilities and rewards
- The primary goal of the MDP is to find an optimal policy, which is a rule that tells the decision maker which action to take in each state to maximize the cumulative reward over time

`Markov Property`:
- The core assumption of an MDP is the Markov property, which states that the future state of the system depends only on the current action, not on the entire history of past states and actions
- In other words, the past is irrelevant given the present


## Algorithm
- This algorithm allows agents to learn optimal policies in complex environments with high dimensional states and action spaces

**Core Concepts**
`Q-Learning`: DQN builds upon Q-learning, an algorithm that aims to learn a Q-function

`Q-function`: denoted as $Q(s,a)$, estimates the cumulative (increasing by successive addition) future reward for taking action 'a' in state 's' and then following the optimal policy thereafter

- DQN replaces the traditional Q-table with a neural network
- NN takes the current state as input and outputs the estimated Q-values for all possible actions in that state. This enables the algorithm to handle continuous state spaces and large action spaces

`Experience Replay`
- DQN utilizes an experience replay memory, which stores past experiences (state, action, next_state, reward) as tuples
- During training, the agent randomly samples batches of experiences from the replay memory. This helps to break the correlation between consecutive experiences and improves the stability of learning

`Target network`
- DQN uses a separate target network, which has the same architecture as the main Q-network but its weights are updated less frequently
- It is used to compute the target Q-values during training, enhancing the stability of the learning process

**Steps**
1. Initialize
- Create main Q-network and target Q-network
- Initialize the experience replay memory

2. Interact with the environment
- Observe the intial state 's'
- Select an action 'a' using an exploration-exploitation strategy(eg: $\epsilon-greedy$)
- Execute the action 'a' in the env and observe the reward 'r' and next_state 'n_s'
- Store the experience(s, a, n_s, r) in the replay memory

3. Train the Q-network
- Sample a batch of experience from the replay memory
For each experience in the batch
- Compute the target Q-value using the target network
- Calculate the loss between the predicted Q-value of the main Q-network and the target Q-value
- Update the weights of the main Q-network through backpropagation

4. Update the target network
- Periodically update the target networks weights with the main Q-network's weights (either through a hard update or a soft update)

5. Repeat steps 2-4:
- Continue interacting with the environment, training the Q-network and updating the target network until convergence or a desired performance level is achieved

**Advantages of DQN**
1. Handles high-dimensional state and action spaces
2. Improves stability and convergence compared to traditional Q-learning
3. Enables learning from offline data

**Tips**
1. Convergence:
- There's no single definitive metric to guarantee convergence. But we do have some imdicators and techniques that can be used to assess convergence
- Convergence is not always guaranteed. DQN can sometimes get stuck in a local optima or fail to converge completely
- The choice of hyperparameters (learning rate, exploration rate etc.) can significantly impact convergence. Careful tuning is often necessary

`Reward trends`:
- As the agent learns, the average reward per episode should generally increase. However, we should be cautious of overfitting

`Reward stability`:
- After a certain point, the average reward should stabilize and fluctuate within a narrow range. This suggests that the agent has found a good policy

`Loss function`:
- The training loss of the Q-function should decrease over time. This indicates that the network is improving its ability to predict Q-values accordingly

`Loss stabilization`:
- Similar to rewards, the loss should eventually stabilize, suggesting that the network has converged to a local minimum

`Visual inspection`:
- If the agent is consistently taking actions that lead to high rewards and demonstrates intelligence then it is a good sign of convergence

`Evaluation episodes`:
- Evaluate the agent on separate episodes outside training. This helps to assess the agent's generalization ability and prevent overfitting

## Target Q-network
- The target Q-network plays a crucial role in stabilising the training process
- By providing a more stable target, the target network helps the main Q-network to converge more smoothly and efficiuently to the optimal Q-function

`Target`:
- Target refers to the expected future return for taking a specific action in a given state
- This expected return is used to guide the learning process of the main network

`Breaking correlation`
- In standard Q-learning, the target Q-value (used to compute the loss) is calculated using the main Q-network
- This creates a strong correlation between the target and the network being trained, leading to instability and oscillations in the learning process
- The target network helps break this correlation by providing a more stable target for the main Q-network to learn from

`Reducing oscillations`
- Frequent changes in the target Q-values can cause the main Q-network to oscillate and struggle to converge
- The target networks weights are updated less frequently (eg: every c steps), providing a more stable target and reducing the oscillations

## Replay memory:
- In a typical RL scenario, the experiences are hightly correlated. The current state is heavily influenced by the previous state and actions taken in the past significantly impact the current situatuion
- If the network learns from this sequence of correlated experiences it can lead to significant instability. The network might overfit to short term patterns or become sensitive to recent experiences, hindering its ability to generalize a policy
- Sampling random batches of experiences from the replay buffer helps to break the temporal correlations between consecutive experiences. This leads to more stable and robust learning
- Learning directly from a stream of correlated experiences can cause the Q-values to oscillate significantly. Sampling from the replay buffer introduces a degree of randomness, which helps to smooth out these oscillations and improve the convergence of the learning process


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

`Adam optimizer`
-  It is a popular optimization algorithm used in training deep learning models
- Adam: Adaptive Moment Estimation. Combines best aspects of 2 other optimization algorithms. AdaGrad and RMSprop
- AdaGrad: Adapts the learning rate of each parameter, giving smaller learning rates to parameters that have seen a lot of updates
- RMSprop: addresses a limitation of AdaGrad where the learning rate can become too small

`AdamW`
- Built on top of Adam optimizer
- Primary difference lies in how AdamW handles weight decay
- Adam integrates weight decay into gradient calculation, which can sometimes lead to issues, especially in large models
- AdamW decouples weight decay from the gradient update process. Meaning, weight decay is applied directly to the model's weights after the gradient update step. It leads to improved generalization and reduced overfitting

`amsgrad`
- boolean to choose whether to use AMSGrad variant of the algorithm

`weight decay`
- Weight decay is a regularization technique to prevent overfitting

*Working:*
- Penalizes large weights: During training weight decay adds a penalty term to the loss function
- By adding this penalty, it encourages smaller weights. Large weights can make the model overly sensitive to the training data
- Models with smaller weights tend to generalize better to unseen data.

`optimizer.zero_grad`
- It is a method used to reset the gradients of all optimized tensors to zero before each backpropagation step
- This is necessary because, when we call loss.backwards() to compute the gradients, the calculated gradients are accumulated with any existing gradients in the `.grad` attribute of the tensors

`torch.nn.utils.clip_grad_value_`
- It is a function used to clip the gradients of model parameters to a specific value
- In DL, especially in RNN, gradients can sometimes grow exponentially during backpropagation, leading to numerical instability
- This is to prevent exploding gradients. This stabilizes the training process

`optimizer.step()`
- It is a function that updates the model's parameters, which were computed during the backward pass (`loss.backward()`). These gradients are stored in the `.grad` attribute of each parameter tensor
- It determines how much to adjust each parameter using the gradients and the optimizers specific rules (eg: learing rate, momentum etc)
- It modifies the model's parameters according to the calculated updates

`order of method calls`
- zero_grad: it is crucial to call this before calculating the gradients using `loss.backward()` to ensure that the gradients are not accumulated from previous iterations
- loss.backward(): after zeroing the gradients, we compute the gradients of the loss function with respect to the model's parameters using backpropagation. They are stored in the `.grad` attribute of each parameter tensor
- optimizer.step(): the optimizer uses the calculated gradients stores in the `.grad` attribute of each parameter tensor to update the model's parameters based on the specific optimizer algorithm

## Q-network Parameters
`state_dict`
- learnable parameters (weights and biases) are stored in `model.parameters()`
- a state dict is simply a dictionary that maps each layer to its parameter tensor

## Optimize model:

`zip`
- combines multiple iterators such as lists, tuples, string and dict into a single iterator of tuples
- first element in each passed iterator is passed together and so on
- we can also reverse the operation using the * operator

Eg:

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
```python
a = tensor([[ 9.50,  1.38, -1.41, -2.17],
        [-3.56,  3.51, -4.78, -6.43],
        [ 2.71,  5.40, -5.57, -9.11],
        [-5.61, -4.02,  3.26,  5.99],
        [ 2.61,  3.43, -1.39, -5.92],
    ])

b = tensor([[0],
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
- disables gradient calculation temporarily for performance optimization
- gradient calculation is essential for training neural network using techniques like backpropagation. It involves calculating how much each parameter in the network contributes to the final output's error
- gradient is disabled for the purpose of inference
- during inference, we are primarily interested in the model's prediction (outputs) for given inputs. we don't need to update the model's parameters during inference

`backward()`
- It is a crucial method used during backpropagation process in neural network training
- it calculates the gradients of a given tensor i.e. it determines how much each parameter in the network contributes to the final output's error
- backward() accumulates gradients in the leaf tensors. This means that if you call backward() multiple times without zeroing the gradients first, the gradients wil be summed up

```python
import torch

x = torch.randn(3, requires_grad=True)  # Create a tensor with requires_grad=True
y = x * 2 
z = y * y 
out = z.mean()  # Calculate the mean of the tensor

out.backward()  # Compute gradients

print(x.grad)  # Access the gradients of the tensor x
```

`L1Loss`
- measures the mean absolute error (MAE) between each element in the predicted values and actual target values
- L1Loss is less sensitive to outliers compared to L2Loss (Mean Squared Error). This is because it penalizes errors proportional to their magnitude, without squaring the difference
- L1Loss is commonly used in regression tasks where you want to minimize the average absolute difference between predictions and actual values
- Also used when our data has outliers and you want to encourage sparsity in your model's weights

$$
L1\_loss = (1/n) * Σ |predicted\_value - actual\_value|
$$

`SmoothL1Loss`
- computes a smoother version of the L1Loss
- combines L1 and L2 loss. for smaller errors, it behaves like L2 loss (MSE), which is differentiable at 0. for large errors it behaves like L1 loss (MAE), which is less sensitive to outliers

Eg:
$$
loss(x, y) = 0.5 * x^2 / beta      if |x| < beta
\\
loss(x, y) = |x| - 0.5 * beta     otherwise
$$
- x is difference between the predicted value and the actual value


## Main training loop:

`unsqueezing`: returns a new tenor of one dimension, inserted at the specified location

Eg:
```bash
[-0.00483294 -0.010573 0.03723028 -0.03453089]

->

tensor([[-0.0048, -0.0106,  0.0372, -0.0345]], device='cuda:0')
```

`soft update of the target network's weights`
- a soft update of the target network's weights refers to a gradual and continuous adjustment of the target network's parameters towards the main Q-network
- in DQN, a separate target network is used to compute the target Q-values during training. This network has the same architecture as the main Q-network but its weights are updated less frequently
- hard update: a traditional update is to periodically perform a hard update i.e. weights are directly copied from the main Q-network to the target network. This can lead to instability in training
- soft update: a smoother approach is to perform a soft update i.e. the target networks weights are updated gradually towards the main Q-network's weights by a weighted average
- soft updates help to stabilize the training process. it allows for a more gradual and continuous learning

## Plot function
`plt.figure`
- creates a new figure or activate an exisiting figure

`torch.tensor.unfold`
- subtensors are extracted with a sliding window approach, moving along a specified dimension of the tensor

```python
import torch

x = torch.arange(10).reshape(2, 5)  # Create a sample tensor
print(x)
# tensor([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

y = x.unfold(1, 3, 2)  # Unfold along dimension 1 (columns), size 3, step 2
print(y)
# tensor([[[0, 1, 2],
#          [2, 3, 4]]],
#        [[[5, 6, 7],
#          [7, 8, 9]]]])
```