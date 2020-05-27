import os

# TODO: Actually fix hese things instead of just ignoring them
# Reduce logging from TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


MAX_EPISODES = 1000
ALPHA = 1e-4  # Policy gradient learning rate
GAMMA = 0.99  # Reward decay rate
RANDOM_SEED = 0

# Environment solved at 195 steps for 100 consecutive episodes
SUCCESS_EPISODES = 100
SUCCESS_STEPS = 195


class PolicyNN:
    def __init__(self, env):
        self.state_shape = env.observation_space.shape  # the state space
        self.action_shape = env.action_space.n  # the action space
        self.model = self.build_model()  # build model

    def build_model(self):
        """
        Build a neural network with a single hidden layer. Softmax is used for the
        final layer as we want to avoid the weights from ever reaching 0 or 1.
        """
        model = keras.Sequential()
        model.add(keras.Input(shape=self.state_shape))
        model.add(keras.layers.Dense(24, activation="relu"))
        model.add(keras.layers.Dense(12, activation="relu"))
        # model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dense(self.action_shape, activation="softmax"))
        model.compile(
            loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=0.01),
        )
        return model

    def get_action(self, state):
        """
        Select an action to take based on the current policy and state. Also return the
        probability of having selected that action given the state in the current
        policy.
        """
        state = state.reshape((1, *self.state_shape))  # Prefix with batch size
        action_probs = self.model.predict(state).flatten()
        action_probs /= np.sum(action_probs)
        action = np.random.choice(self.action_shape, p=action_probs)
        return action, action_probs

    def update(self, states, actions, action_probs, rewards):
        """
        Update the policy with rewards gained throughout episode.

        Discounted rewards are a sum of all future rewards, where the further into the
        future a reward is, the more it's discounted:

            G_t = Σ_t γ^t * r_t

        A gradient is the difference between the probability of an action being taken
        in a state and a binary value of if the action was actually chosen. To
        calculate this we calculate an encoded action, which would look like:

            [0.  0.  1.  0.]  # Action 3 of 4 possible actions selected
        
        """
        num_steps = len(states)
        assert len(actions) == num_steps
        assert len(action_probs) == num_steps
        assert len(rewards) == num_steps

        rewards = np.vstack(rewards)

        # Calculate discounted reward
        discounted_rewards = []
        for t in range(num_steps):
            discounted_reward = 0
            for i, reward in enumerate(rewards[t:]):
                discounted_reward += (GAMMA ** i) * reward
            discounted_rewards.append(discounted_reward)

        # Normalize discounted rewards
        mean_rewards = np.mean(discounted_rewards)
        std_rewards = np.std(discounted_rewards) + 1e-7  # Avoiding zero div
        norm_discounted_rewards = (discounted_rewards - mean_rewards) / std_rewards

        # Calculate the gradients
        gradients = []
        for t in range(num_steps):
            encoded_action = np.eye(self.action_shape)[actions[t]]
            gradients.append(encoded_action - action_probs[t])

        states_matrix = np.vstack(states)
        targets_matrix = action_probs + (
            ALPHA * (norm_discounted_rewards * np.vstack(gradients))
        )

        self.model.train_on_batch(states_matrix, targets_matrix)


env = gym.make("CartPole-v0")

# Set random seed
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
env.seed(RANDOM_SEED)

# Set up graph
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

policy = PolicyNN(env)

# Used for final graph
episode_steps = []
success_episodes = 0

for episode in range(MAX_EPISODES):
    state = env.reset()
    states = []
    actions = []
    action_probs = []
    rewards = []

    for step in range(1, SUCCESS_STEPS + 1):
        # Interact with the world
        action, action_prob = policy.get_action(state)
        new_state, reward, done, _ = env.step(action)

        # Record the details
        states.append(state)
        actions.append(action)
        action_probs.append(action_prob)
        rewards.append(reward)

        state = new_state

        if done:
            policy.update(states, actions, action_probs, rewards)
            episode_steps.append(step)
            print(f"Episode {episode + 1}:\tAgent lasted {step} steps")

            # Update graph each step
            ax.plot(episode_steps, "blue")
            fig.canvas.draw()
            fig.canvas.flush_events()

            break

    if episode_steps[-1] == SUCCESS_STEPS:
        success_episodes += 1
    else:
        success_episodes = 0

    if success_episodes == SUCCESS_EPISODES:
        print(f"Environment completed after {episode} episodes")

# Plot results
plt.plot(episode_steps)
plt.xlabel("Episode")
plt.ylabel("Steps taken in episode")
plt.show()
