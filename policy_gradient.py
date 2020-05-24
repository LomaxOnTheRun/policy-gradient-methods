import gym
import matplotlib.pyplot as plt
import numpy as np

MAX_EPISODES = 10
GAMMA = 0.9

# Environment solved at 195 steps for 100 consecutive episodes
SUCCESS_EPISODES = 100
SUCCESS_STEPS = 195


def get_action(policy, state, env):
    """
    Select an action to take based on the current policy and state.

    TODO: Use NN for this.
    """
    return env.action_space.sample()  # Random action


def get_action_probability(policy, state, action):
    """
    Get the probability of selecting the given action given the current policy.

    TODO: Use NN for this.
    """
    return 1e-10


def update_policy(policy, rewards, action_probs):
    """
    Update the policy with rewards gained throughout episode.

    Discounted rewards are a sum of all future rewards, where the further into the
    future a reward is, the more it's discounted:

        G_t = Σ_t γ^t * r_t

    TODO: Upgrade NN.
    """
    num_steps = len(rewards)
    policy_gradient = []
    for t in range(num_steps):
        # Calculate discounted reward
        discounted_reward = 0
        for i, reward in enumerate(rewards[t:]):
            discounted_reward += (GAMMA ** i) * reward

        gradient = -discounted_reward * np.log(action_probs[t])
        policy_gradient.append(gradient)

    # TODO: Upgrade NN


env = gym.make("CartPole-v0")

# TODO: Build a policy NN with Theano
policy = None

# Used for final graph
episode_steps = []

for episode in range(MAX_EPISODES):
    state = env.reset()
    rewards = []
    action_probs = []

    for step in range(SUCCESS_STEPS + 1):
        # env.render()
        action = get_action(policy, state, env)
        action_prob = get_action_probability(policy, state, action)

        state, reward, done, _ = env.step(action)

        rewards.append(reward)
        action_probs.append(action_prob)

        if done:
            update_policy(policy, rewards, action_probs)
            episode_steps.append(step)
            print(f"Episode {episode + 1}:\tAgent lasted {step} steps")
            break

env.close()

# Plot results
plt.plot(episode_steps)
plt.xlabel("Episode")
plt.ylabel("Steps taken in episode")
plt.show()
