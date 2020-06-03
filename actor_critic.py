import numpy as np
import tensorflow
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from environments import get_env_info
from shared import run_environment

from tensorflow.compat.v1 import disable_v2_behavior

# Required to avoid TF v2 errors
disable_v2_behavior()


# ACTOR_LEARNING_RATE = 0.00001  # Adam learning rate
# CRITIC_LEARNING_RATE = 0.00005  # Adam learning rate
ACTOR_LEARNING_RATE = 0.0005  # Adam learning rate
CRITIC_LEARNING_RATE = 0.0025  # Adam learning rate
GAMMA = 0.99  # Reward decay rate


class ActorCriticAgent:
    def __init__(self, env):
        self.state_shape = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.actor_model, self.critic_model, self.policy_model = self.build_models()

    def build_models(self):
        """
        Build two models, one for actor (the model which will select actions based on a
        state) and one for the critic (which builds an estimation of states' values).

        Separate models are used for training the actor and for using the actor to
        select actions as the learning model requires an extra input of the critic's
        value approximation for that state.

        Note that all three models share the same hidden layers.
        """
        states = Input(shape=self.state_shape)
        value_delta = Input(shape=[1])
        # hidden_1 = Dense(1024, activation="relu")(states)
        # hidden_2 = Dense(512, activation="relu")(hidden_1)
        hidden_1 = Dense(64, activation="relu")(states)
        hidden_2 = Dense(64, activation="relu")(hidden_1)
        probs = Dense(self.num_actions, activation="softmax")(hidden_2)
        value = Dense(1, activation="linear")(hidden_2)

        def custom_loss(y_true, y_pred):
            clipped_y_pred = K.clip(y_pred, 1e-10, 1 - 1e-10)
            log_likelihood = y_true * K.log(clipped_y_pred)
            loss = K.sum(-log_likelihood * value_delta)
            return loss

        # Model to train the actor
        actor_model = Model(inputs=[states, value_delta], outputs=[probs])
        actor_model.compile(loss=custom_loss, optimizer=Adam(lr=ACTOR_LEARNING_RATE))

        # Model to train the critic and get critic values for states
        critic_model = Model(inputs=[states], outputs=[value])
        critic_model.compile(
            loss="mean_squared_error", optimizer=Adam(lr=CRITIC_LEARNING_RATE)
        )

        # Model to return probabilities for actions given a state
        policy_model = Model(inputs=[states], outputs=[probs])

        return actor_model, critic_model, policy_model

    def get_action(self, state):
        """
        Select an action to take based on the current policy and state. Also return the
        probability of having selected each action given the state in the current
        policy.
        """
        state_matrix = np.array([state])  # Put in batch by itself
        actions_prob = self.policy_model.predict(state_matrix)[0]
        actions_prob /= np.sum(actions_prob)
        action = np.random.choice(self.num_actions, p=actions_prob)
        return action

    def update(self, state, action, reward, new_state, done):
        """
        TODO: Explain this, look at theory.
        """
        # Put in batch by themselves
        state_matrix = np.array([state])
        new_state_matrix = np.array([new_state])

        # Calculate value delta between old and new states
        state_value = self.critic_model.predict(state_matrix)
        new_state_value = self.critic_model.predict(new_state_matrix)
        new_state_target = reward + (GAMMA * new_state_value)
        if done:
            new_state_target -= GAMMA * new_state_value
            # If end of episode don't use new state value
            # new_state_target = np.array([[reward]])
        value_delta = new_state_target - state_value

        # Create action encoding matrix
        actions = np.zeros([1, self.num_actions])
        # actions[0, action] = 1
        actions[np.arange(1), action] = 1

        # self.actor_model.train_on_batch([state, value_delta], actions)
        # self.critic_model.train_on_batch(state, new_state_target)
        self.actor_model.fit([state_matrix, value_delta], actions, verbose=0)
        self.critic_model.fit(state_matrix, new_state_target, verbose=0)


env_info = get_env_info("CartPole-v0")
# env_info = get_env_info("CartPole-v1")
# env_info = get_env_info("LunarLander-v2")
run_environment(ActorCriticAgent, env_info)
