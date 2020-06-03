# Policy Gradient Methods

This is a repo to play around with policy gradient methods so I can get some practice with them, with a view to expanding to Actor-Critic methods and then PPO methods in the future.

## Methods

### MC policy gradient (REINFORCE) with baseline

This now works well. I've copied a lot of code from the excellent video [here](https://www.youtube.com/watch?v=IS0V8z8HXrM).

This agent can only work with discrete actions spaces for now, but in theory could be expanded to work with continuous action spaces with relatively little work. It also cannot complete challenges where it gets a single reward at the very end of the episode, if it is hugely unlikely to stumble across the end my accident.

With two hidden layers (64 nodes each), this agent is able to complete the following environments:

- CartPole-v0 (in 461 episodes, averaged over 1 attempt)
- CartPole-v1 (in 1246 episodes, averaged over 1 attempt)

Environments it fails at despite being able to interface with:

- LunarLander-v2 (highest seen is ~168.8 over 100 runs but then gets worse again)
- Acrobot-v1 (it get close, but is dependent on a good run very early on)
- MountainCar-v0 (never gets even close)

### Actor-Critic

Taken a lot of inspiration from [this video](https://www.youtube.com/watch?v=2vJtbAha3To) which is from the same YouTube channel as the REINFORCE was. Actor-critic seems to work, but _much_ more slowly than REINFORCE. I'm not sure if it's because the actor and the critic share a network and learning a value is very slow, or something else. Actually, it looks like it's the learning rate(s) which are about x10 lower than with REINFORCE. However, simply increasing the learning rates leads to unstability once the agent has "mostly learned" the environment.

With two hidden layers (64 nodes each) and low learning rates (0.00001 and 0.00005), this agent is able to complete the following environments:

- CartPole-v0 (in 4059 episodes, averaged over 1 attempt)

## Further improvements

I'm currently using Tensorflow v1.15 equivalent by using `tensorflow.compat.v1.disable_v2_behavior`. I should actually upgrade to using v2 syntax.

Add argparse shared functionality to easily pass in arguments from the command line.
