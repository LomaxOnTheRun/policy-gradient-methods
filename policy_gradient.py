import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from environments import get_env_info
from shared import run_environment

from tensorflow.compat.v1 import disable_v2_behavior

# Required to avoid TF v2 errors
disable_v2_behavior()


ALPHA = 0.0005  # Policy gradient learning rate
GAMMA = 0.99  # Reward decay rate


class MCPolicyGradient:
    def __init__(self, env):
        self.state_shape = env.observation_space.shape  # the state space
        self.num_actions = env.action_space.n  # the action space
        self.policy_model, self.predict_model = self.build_models()

        # Save observations as this a Monte-Carlo algorithm (updates at end of episode)
        self.states = []
        self.actions = []
        self.rewards = []

    def build_models(self):
        """
        Build a neural network. Softmax is used for the final layer as we're
        calculating probabilities.
        """
        states = keras.layers.Input(shape=self.state_shape)
        advantage = keras.layers.Input(shape=[1])
        hidden_1 = keras.layers.Dense(64, activation="relu")(states)
        hidden_2 = keras.layers.Dense(64, activation="relu")(hidden_1)
        probs = keras.layers.Dense(self.num_actions, activation="softmax")(hidden_2)

        def custom_loss(y_true, y_pred):
            clipped_y_pred = keras.backend.clip(y_pred, 1e-10, 1 - 1e-10)
            log_likelihood = y_true * keras.backend.log(clipped_y_pred)
            loss = keras.backend.sum(-log_likelihood * advantage)
            return loss

        # Main model used for training, needs additional advantages input
        policy_model = keras.models.Model(inputs=[states, advantage], outputs=[probs])
        policy_model.compile(loss=custom_loss, optimizer=Adam(lr=ALPHA))

        # Model used for predicting, uses same weights as other model
        predict_model = keras.models.Model(inputs=[states], outputs=[probs])

        return policy_model, predict_model

    def get_action(self, state):
        """
        Select an action to take based on the current policy and state. Also return the
        probability of having selected each action given the state in the current
        policy.
        """
        state = state.reshape((1, *self.state_shape))  # Prefix with batch size
        actions_prob = self.predict_model.predict([state]).flatten()
        actions_prob /= np.sum(actions_prob)
        action = np.random.choice(self.num_actions, p=actions_prob)
        return action

    def update(self, state, action, reward, done):
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
            self.rewards.append(reward)
            return

        num_steps = len(self.states)

        # Create action encoding matrix
        actions_matrix = np.zeros([num_steps, self.num_actions])
        actions_matrix[np.arange(num_steps), self.actions] = 1

        # Calculate discounted rewards (G in the literature)
        discounted_rewards = np.zeros((num_steps, 1))
        for t in range(num_steps):
            for i, reward in enumerate(self.rewards[t:]):
                discounted_rewards[t] += (GAMMA ** i) * reward

        # Normalize discounted rewards (baseline for REINFORCE algorithm)
        mean_rewards = np.mean(discounted_rewards)
        std_rewards = np.std(discounted_rewards)
        if np.std(discounted_rewards) <= 0:
            std_rewards = 1  # Avoiding zero div
        advantages = (discounted_rewards - mean_rewards) / std_rewards

        states_matrix = np.array(self.states)

        self.policy_model.train_on_batch([states_matrix, advantages], actions_matrix)

        # Reset all retained info
        self.states = []
        self.actions = []
        self.rewards = []


# env_info = get_env_info("CartPole-v0")
env_info = get_env_info("CartPole-v1")
# env_info = get_env_info("LunarLander-v2")
run_environment(MCPolicyGradient, env_info, display=True)
