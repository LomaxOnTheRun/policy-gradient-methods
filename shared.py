import os

# TODO: Actually fix hese things instead of just ignoring them
# Reduce logging from TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


MAX_EPISODES = 2000

# Environment solved at 195 steps for 100 consecutive episodes
SUCCESS_EPISODES = 100
SUCCESS_SCORE = 195


def run_environment(policy_class, random_seed=None, display=False):
    """
    Run through the environment with a given policy.
    """
    env = gym.make("CartPole-v0")
    # env = gym.make("LunarLander-v2")

    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        env.seed(random_seed)

    policy = policy_class(env)

    # Graph to display progress
    fig = plt.figure()
    ax = fig.add_subplot(111)
    episode_scores = []
    average_scores = []

    # Stops final graph
    if display:
        print("y")
        plt.ion()

    for episode in range(MAX_EPISODES):
        state = env.reset()
        step = 0
        score = 0
        done = False

        while not done:
            step += 1

            # Interact with the world
            action = policy.get_action(state)
            new_state, reward, done, _ = env.step(action)

            # Always give option of updating, some methods wait until end of run
            policy.update(state, action, reward, done)

            state = new_state
            score += reward

            if done:
                episode_scores.append(score)
                average_scores.append(np.mean(episode_scores[-100:]))
                print(
                    f"Episode {episode + 1}\t"
                    + f"Score: {score:.1f}\t"
                    + f"Average score: {average_scores[-1]:.1f} steps"
                )

                # Update graph each step
                if display:
                    print("x")
                    ax.plot(episode_scores, "blue")
                    # ax.plot(average_scores, "orange")
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                break

        mean_score = np.mean(episode_scores[-100:])
        if len(scores) >= SUCCESS_EPISODES and mean_score >= SUCCESS_SCORE:
            print(f"Environment completed after {episode} episodes")

    # Plot results
    plt.plot(episode_scores)
    plt.plot(average_scores)
    plt.xlabel("Episode")
    plt.ylabel("Steps taken in episode")
    plt.show()
