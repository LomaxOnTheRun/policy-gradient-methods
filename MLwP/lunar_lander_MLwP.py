import gym
import matplotlib.pyplot as plt
import numpy as np
from tf_v2.policy_gradient_cheating import Agent


#
# Copied directly from https://www.youtube.com/watch?v=IS0V8z8HXrM
#


if __name__ == "__main__":
    agent = Agent(
        ALPHA=0.0005,
        input_dims=8,
        GAMMA=0.99,
        n_actions=4,
        layer1_size=64,
        layer2_size=64,
    )
    env = gym.make("LunarLander-v2")
    score_history = []

    n_episodes = 2000

    for i in range(n_episodes):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward)
            observation = observation_
            score += reward
        score_history.append(score)

        agent.learn()

        print(
            "episode ",
            i,
            "score %.1f" % score,
            "average_score %1.f" % np.mean(score_history[-100:]),
        )

    plt.plot(score_history)
    plt.show()
