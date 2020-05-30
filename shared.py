import os

# TODO: Actually fix hese things instead of just ignoring them
# Reduce logging from TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


MAX_EPISODES = 1000

# Environment solved at 195 steps for 100 consecutive episodes
SUCCESS_EPISODES = 100
SUCCESS_STEPS = 195


def run_environment(policy_class, random_seed=None):
    """
    Run through the environment with a given policy.
    """
    env = gym.make("CartPole-v0")

    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        env.seed(random_seed)

    policy = policy_class(env)

    # Graph to display progress
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    episode_steps = []
    success_episodes = 0

    for episode in range(MAX_EPISODES):
        state = env.reset()

        for step in range(1, SUCCESS_STEPS + 1):
            # Interact with the world
            action, actions_prob = policy.get_action(state)
            new_state, reward, done, _ = env.step(action)

            # Always give option of updating, some methods wait until end of run
            policy.update(state, action, actions_prob, reward, done)

            state = new_state

            if done:
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
