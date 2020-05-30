# Policy Gradient Methods

This is a repo to play around with policy gradient methods so I can get some practice with them, with a view to expanding to Actor-Critic methods and then PPO methods in the future.

### MC policy gradient (REINFORCE)

This works ok for a while, but then breaks down after a while (a few hundred steps). Not sure why.

The other main thing I don't fully understand is how to turn the upgrade equation (S&B p328) into code:

$$\theta <-- \theta + \alpha \gamma^t \nabla ln \pi (A_t | S_t, \theta)$$

The gradient specifically in the context of neural networks is the thing I don't clearly understand.

### Further improvements

I'm currently getting errors to do with tracings, e.g.:

```
5 out of the last 21 calls to <function Model.make_train_function.<locals>.train_function at 0x7fed0c2599d0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors.
```

I think this would just require me to convert python structures into TF structures at some point, but because it's not the thing I'm most concerned with (i.e. the RL aspect of these solutions) I'm going to leave it alone for now.

Also, I suspect that running these in a Colab notebook / in a container on a remote server that has access to GPU processing would be faster, but again, it's not central for now and not important enough for me to sink the time into figuring out how to do it.
