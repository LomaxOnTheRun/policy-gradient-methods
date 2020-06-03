import os
from time import sleep

# TODO: Actually fix hese things instead of just ignoring them
# Reduce logging from TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


MAX_EPISODES = 100000


def run_environment(
    policy_class, env_info, random_seed=None, display=False, render=False
):
    """
    Run through the environment with a given policy.
    """
    # Get env info
    env_success_episodes = env_info["success_episodes"]
    env_success_score = env_info["success_score"]
    env = gym.make(env_info["name"])

    print(f"Running environment {env_info['name']}")

    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        env.seed(random_seed)

    # Build policy agent
    policy = policy_class(env)

    # Update graph each step, disables final graph
    if display:
        plt.ion()

    # Graph to display progress
    fig = plt.figure()
    ax = fig.add_subplot(111)
    episode_scores = []
    average_scores = []

    for episode in range(MAX_EPISODES):
        state = env.reset()
        score = 0
        done = False

        while not done:
            if render and episode % 100 == 0:
                env.render()
                sleep(0.01)

            # Interact with the world
            action = policy.get_action(state)
            new_state, reward, done, _ = env.step(action)

            # Always give option of updating, some methods wait until end of run
            policy.update(state, action, reward, new_state, done)

            state = new_state
            score += reward

            if done:
                episode_scores.append(score)
                average_score = np.mean(episode_scores[-env_success_episodes:])
                average_scores.append(average_score)
                print(
                    f"Episode {episode + 1}\t"
                    + f"Score: {score:.1f}\t"
                    + f"Running average: {average_scores[-1]:.1f}"
                )

                # Update graph each step
                if display:
                    ax.plot(episode_scores, "blue")
                    ax.plot(average_scores, "orange")
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                break

        if (
            len(episode_scores) >= env_success_episodes
            and average_scores[-1] >= env_success_score
        ):
            print(f"Environment completed after {episode} episodes")
            break

    # Plot results
    plt.plot(episode_scores)
    plt.plot(average_scores)
    plt.xlabel("Episode")
    plt.ylabel("Steps taken in episode")
    plt.show()
