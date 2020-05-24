import gym
import matplotlib.pyplot as plt

MAX_EPISODES = 10

# Environment solved at 195 steps for 100 consecutive episodes
SUCCESS_EPISODES = 100
SUCCESS_STEPS = 195


def get_action(policy, state, env):
    """
    Select an action to take based on the current policy and state.

    TODO: Use NN for this.
    """
    return env.action_space.sample()  # Random action


def get_probability(policy, state, action):
    """
    Get the probability of selecting the given action given the current policy.

    TODO: Use NN for this.
    """
    pass


def update_policy(policy, rewards, probabilities):
    """
    Update the policy with rewards gained throughout episode.

    TODO: Actually do this
    """
    pass


env = gym.make("CartPole-v0")

# TODO: Build a policy NN with Theano
policy = None

# Used for final graph
episode_steps = []

for episode in range(MAX_EPISODES):
    state = env.reset()
    rewards = []
    probabilities = []

    for step in range(SUCCESS_STEPS + 1):
        # env.render()

        action = get_action(policy, state, env)
        probability = get_probability(policy, state, action)

        state, reward, done, _ = env.step(action)

        rewards.append(reward)
        probabilities.append(probability)

        if done:
            update_policy(policy, rewards, probabilities)
            episode_steps.append(step)
            print(f"Episode {episode + 1}:\tAgent lasted {step} steps")
            break

env.close()

# Plot results
plt.plot(episode_steps)
plt.xlabel("Episode")
plt.ylabel("Steps taken in episode")
plt.show()
