import gym

MAX_STEPS = 100

env = gym.make("CartPole-v0")
env.reset()
for _ in range(MAX_STEPS):
    env.render()
    env.step(env.action_space.sample())
env.close()
