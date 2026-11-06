# Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium

---
## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman optimality equation.

---

## Problem Statement


To find the optimal policy for an agent navigating through a FrozenLake environment using the Value Iteration algorithm. The agent must reach the goal while avoiding holes by maximizing the expected cumulative reward.

---

## Software Requirements
- Programming Language: Python
- Platform: Google Colab / Jupyter Notebook
- Libraries: Gymnasium, NumPy
- Environment: FrozenLake-v1

---

## Environment Description

FrozenLake is a grid-based environment in which an agent moves on a frozen surface to reach a goal. The environment contains four types of tiles:

- S – Starting position
- F – Frozen surface 
- H – Hole 
- G – Goal 

The agent can move in four directions: Left, Down, Right and Up. The environment is slippery, so the agent may not always move in the intended direction.

---

## MDP Representation

The FrozenLake environment is represented using the Markov Decision Process (MDP).

MDP = (S, A, P, R, γ)

- S (States): 16 grid positions
- A (Actions): Left, Down, Right and Up.
- P (Transition Probability): Probability of moving from one state to another.
- R (Reward): Reward received for reaching the goal.
- γ (Discount Factor): 0.99.

---

## Theory

Value Iteration is a model-based Reinforcement Learning algorithm used to find the optimal policy of an environment. It repeatedly updates the value of each state using the Bellman Optimality Equation until the values become stable.

The algorithm calculates the maximum expected reward for each action and selects the action with the highest value. Finally, it derives the optimal policy from the state-action values.

Bellman Optimality Equation:

V(s) = maxₐ Σ P(s'|s,a) [R + γV(s')]

Where:

- V(s) – Value of the current state.
- P(s'|s,a) – Transition probability.
- R – Immediate reward.
- γ – Discount factor.
- V(s') – Value of the next state.



## Algorithm

- Start the program.
- Import Gymnasium and NumPy libraries.
- Create the FrozenLake environment.
- Initialize the state-value function with zeros.
- Set the discount factor and convergence threshold.
- For every state, calculate the expected value of all possible actions.
- Update the state value using the maximum action value.
- Repeat the process until the maximum change is less than the threshold.
- Calculate the optimal policy using the Q-values.
- Display the number of iterations, optimal state-value function and optimal policy.
- Stop the program.



## Python Program


```python

import gymnasium as gym
import numpy as np

# Create FrozenLake Environment
env = gym.make("FrozenLake-v1", is_slippery=True)

# Value Iteration Algorithm
def value_iteration(env, gamma=0.99, theta=1e-8):

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    iteration = 0

    while True:
        delta = 0

        for s in range(n_states):
            v = V[s]

            for a in range(n_actions):
                Q[s, a] = sum(
                    prob * (reward + gamma * V[next_state] * (not done))
                    for prob, next_state, reward, done
                    in env.unwrapped.P[s][a]
                )

            V[s] = np.max(Q[s])
            delta = max(delta, abs(v - V[s]))

        iteration += 1

        if delta < theta:
            break

    # Derive policy from Q-table
    policy = np.argmax(Q, axis=1)

    return V, policy, iteration


# Execute Value Iteration
V, policy, iterations = value_iteration(env)

print("Name: SASMINA S")
print("Register Number: 212225230254")

print("Value Iteration Completed")
print("Number of Iterations:", iterations)

print("\nOptimal State-Value Function:")
print(np.round(V.reshape(4, 4), 4))

# Action Symbols
action_symbols = {
    0: "L",
    1: "D",
    2: "R",
    3: "U"
}

policy_grid = np.array(
    [action_symbols[a] for a in policy]
).reshape(4, 4)

print("\nOptimal Policy:")
print(policy_grid)

env.close()
```

---

## Output

<img width="397" height="354" alt="image" src="https://github.com/user-attachments/assets/ef772ede-cb0c-456d-bf7e-818395ef92da" />


---

## Result

Thus, the Value Iteration algorithm was successfully implemented using the Gymnasium FrozenLake environment. The optimal state-value function and optimal policy were computed using the Bellman Optimality Equation.

---

## Inference

Value Iteration repeatedly updates the state values until convergence. It helps the agent identify the best action in each state to maximize the expected reward. Thus, the agent learns an optimal policy for navigating the FrozenLake environment.

---

