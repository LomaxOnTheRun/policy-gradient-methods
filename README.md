# Policy Gradient Methods

This is a repo to play around with policy gradient methods so I can get some practice with them, with a view to expanding to Actor-Critic methods and then PPO methods in the future.

## Methods

### MC policy gradient (REINFORCE) with baseline

This now works well. I've copied a lot of code from the excellent video [here](https://www.youtube.com/watch?v=IS0V8z8HXrM).

### Actor-Critic

TODO

## Further improvements

I'm currently using Tensorflow v1.15 equivalent by using `tensorflow.compat.v1.disable_v2_behavior`. I should actually upgrade to using v2 syntax.
