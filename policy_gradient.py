import numpy as np
from tensorflow import keras

from shared import run_environment


ALPHA = 1e-4  # Policy gradient learning rate
GAMMA = 0.99  # Reward decay rate


class MCPolicyGradient:
    def __init__(self, env):
        self.state_shape = env.observation_space.shape  # the state space
        self.num_actions = env.action_space.n  # the action space
        self.model = self.build_model()

        # Save observations as this a Monte-Carlo algorithm (updates at end of episode)
        self.states = []
        self.actions = []
        self.actions_probs = []
        self.rewards = []

    def build_model(self):
        """
        Build a neural network with a single hidden layer. Softmax is used for the
        final layer as we want to avoid the weights from ever reaching 0 or 1.
        """
        model = keras.Sequential()
        model.add(keras.Input(shape=self.state_shape))
        model.add(keras.layers.Dense(12, activation="relu"))
        model.add(keras.layers.Dense(12, activation="relu"))
        model.add(keras.layers.Dense(self.num_actions, activation="softmax"))
        model.compile(
            loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=0.01),
        )
        return model

    def get_action(self, state):
        """
        Select an action to take based on the current policy and state. Also return the
        probability of having selected each action given the state in the current
        policy.
        """
        state = state.reshape((1, *self.state_shape))  # Prefix with batch size
        actions_prob = self.model.predict(state).flatten()
        actions_prob /= np.sum(actions_prob)
        action = np.random.choice(self.num_actions, p=actions_prob)
        return action, actions_prob

    def update(self, state, action, actions_prob, reward, done):
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
        # In not the end of the episode, just keep track of values
        if not done:
            self.states.append(state)
            self.actions.append(action)
            self.actions_probs.append(actions_prob)
            self.rewards.append(reward)
            return

        num_steps = len(self.states)
        assert len(self.actions) == num_steps
        assert len(self.actions_probs) == num_steps
        assert len(self.rewards) == num_steps

        # Calculate discounted reward
        rewards = np.vstack(self.rewards)
        discounted_rewards = []
        for t in range(num_steps):
            discounted_reward = 0
            for i, reward in enumerate(self.rewards[t:]):
                discounted_reward += (GAMMA ** i) * reward
            discounted_rewards.append(discounted_reward)

        # Normalize discounted rewards
        mean_rewards = np.mean(discounted_rewards)
        std_rewards = np.std(discounted_rewards) + 1e-7  # Avoiding zero div
        norm_discounted_rewards = (discounted_rewards - mean_rewards) / std_rewards
        norm_discounted_rewards = norm_discounted_rewards.reshape((-1, 1))

        # Calculate the gradients
        gradients = np.zeros((num_steps, self.num_actions))
        for t in range(num_steps):
            action = self.actions[t]
            gradients[t][action] = 1 - self.actions_probs[t][action]

        states_matrix = np.vstack(self.states)
        targets_matrix = np.vstack(self.actions_probs) + (
            ALPHA * norm_discounted_rewards * gradients
        )

        self.model.train_on_batch(states_matrix, targets_matrix)

        # Reset all retained info
        self.states = []
        self.actions = []
        self.actions_probs = []
        self.rewards = []


run_environment(MCPolicyGradient)
