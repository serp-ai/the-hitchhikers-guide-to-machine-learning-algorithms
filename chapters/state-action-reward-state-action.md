# State-Action-Reward-State-Action

Examples & Code

SARSA (State-Action-Reward-State-Action) is a temporal difference on-policy algorithm used in reinforcement learning to train a Markov decision process model on a new policy. This algorithm falls under the category of reinforcement learning, which focuses on how an agent should take actions in an environment to maximize a cumulative reward signal.

{% embed url="https://youtu.be/cvrCNpe3BuE?si=2kZNsSUfWmYWW7GF" %}

## State-Action-Reward-State-Action: Introduction

| Domains          | Learning Methods | Type                |
| ---------------- | ---------------- | ------------------- |
| Machine Learning | Reinforcement    | Temporal Difference |

The State-Action-Reward-State-Action (SARSA) algorithm is an on-policy reinforcement learning algorithm used to train a Markov decision process model based on a new policy. SARSA is a type of temporal difference learning method that updates its Q-values based on the current state, action, reward, next state, and next action. Unlike some other reinforcement learning algorithms, SARSA takes into account the current policy being pursued when updating its Q-values, making it particularly useful for problems where the agent cannot completely explore the state-action space.

Because it is an on-policy algorithm, SARSA learns by interacting with the environment using the same policy that it is improving. This means that it may take longer to converge on an optimal policy than off-policy algorithms like Q-learning, but it has the advantage of being more stable and able to handle stochastic environments. SARSA is commonly used in problems with discrete state and action spaces, such as gridworld and cartpole simulations, and has also been adapted for continuous state and action spaces.

As a reinforcement learning algorithm, SARSA is particularly useful for tasks where feedback is delayed or sparse, such as playing a game of chess or controlling a robot. By gradually updating its Q-values based on the reward received for each action taken, SARSA can learn to make better decisions over time and ultimately arrive at an optimal policy for the given task.

With its flexibility and robustness, the SARSA algorithm has become an essential tool in the field of artificial intelligence and machine learning, allowing engineers to create intelligent systems that can learn and adapt to new challenges and environments.

## State-Action-Reward-State-Action: Use Cases & Examples

SARSA is an on-policy algorithm that is commonly used in reinforcement learning to train a Markov decision process model on a new policy. It falls under the category of temporal difference learning methods, which is a type of machine learning that learns from experience and adjusts its predictions based on the difference between predicted and actual outcomes.

One of the most notable use cases of SARSA is in robotic control. For example, SARSA can be used to teach a robot to navigate a maze by providing it with a reward for reaching the end and penalizing it for hitting a wall. The robot uses SARSA to learn the optimal path to take through the maze based on its current state and the actions it takes.

Another use case of SARSA is in game playing. SARSA can be used to train an agent to play a game by rewarding it for winning and penalizing it for losing. The agent learns the optimal actions to take based on the current state of the game and the actions it takes.

Furthermore, SARSA has been used in autonomous vehicle control. The algorithm can be used to teach a self-driving car to navigate through traffic by providing it with a reward for reaching its destination and penalizing it for causing an accident. SARSA allows the car to learn from its experiences and make better decisions in the future.

Lastly, SARSA has been used in natural language processing. The algorithm can be used to teach a chatbot to respond to user queries by rewarding it for providing accurate and helpful responses and penalizing it for providing irrelevant or incorrect responses. SARSA allows the chatbot to learn from its interactions with users and provide better responses over time.

## Getting Started

SARSA (State-Action-Reward-State-Action) is an on-policy algorithm used in reinforcement learning to train a Markov decision process model on a new policy. It is a type of Temporal Difference learning method, specifically in the category of Reinforcement Learning.

To get started with implementing SARSA, you can follow these steps:

1. Define the environment and the agent
2. Initialize the Q-table with zeros
3. Set hyperparameters: learning rate, discount factor, exploration rate, and maximum number of episodes
4. For each episode: 1. Reset the environment and get the initial state 2. Choose an action using an epsilon-greedy policy based on the Q-table 3. Take the action and observe the reward and the next state 4. Choose the next action using the same epsilon-greedy policy 5. Update the Q-table using the SARSA update rule 6. Update the state and action 7. If the episode is done, break

```
import gym
```

```
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Agent():
    def __init__(self, state_size, action_size, seed, lr=0.01, gamma=0.99, tau=1e-3, buffer_size=int(1e5), batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = self.loss_fn(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

env = gym.make('CartPole-v0')
env.seed(0)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = Agent(state_size=state_size, action_size=action_size, seed=0)

n_episodes = 1000
max_t = 1000
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995

eps = eps_start
for i_episode in range(1, n_episodes+1):
    state = env.reset()
    score = 0
    for t in range(max_t):
        action = agent.act(state, eps)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    eps = max(eps_end, eps_decay*eps)
    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
    if np.mean(scores_window)>=195.0:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        break

```

## FAQs

### What is SARSA?

SARSA (State-Action-Reward-State-Action) is an on-policy algorithm used in reinforcement learning to train a Markov decision process model on a new policy.

### What is the abbreviation for SARSA?

The abbreviation for State-Action-Reward-State-Action is SARSA.

### What type of algorithm is SARSA?

SARSA is a temporal difference learning algorithm.

### What learning methods are used with SARSA?

SARSA is typically used in reinforcement learning, which involves an agent learning to make decisions in an environment by maximizing a reward signal.

## State-Action-Reward-State-Action: ELI5

Imagine you're a baby learning to walk. You take a step forward and feel the ground beneath your feet. That's the **state**. You take another step and feel your balance starting to shift. That's the **action**. You stagger forward, but manage to stay on your feet. That's the **reward**.

The next time you try to take a step, your brain remembers that last reward and adjusts your actions accordingly. That's the **state-action-reward-state- action** algorithm, also known as SARSA.

In simpler terms, SARSA is a way for machines to learn from their actions and adjust their behavior based on the feedback they receive. It's often used in reinforcement learning, where an agent interacts with an environment and receives rewards or punishments based on its actions. By training a Markov decision process model on a new policy, SARSA helps the machine make more informed decisions in the future.

With SARSA, machines can "learn" like a baby learning to walk, taking steps forward and adjusting based on the feedback they receive. It's a powerful tool in the world of artificial intelligence and machine learning that helps agents make smarter decisions and achieve better outcomes.

\*\[MCTS]: Monte Carlo Tree Search [State Action Reward State Action](https://serp.ai/state-action-reward-state-action/)
