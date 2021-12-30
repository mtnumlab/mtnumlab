# theta* = argmin over theta E [0,1] of (-log-likehood)
# where log-likelihood = l(Di theta) = Sum over dataset of weighted log-mean of theta

import silence_tensorflow.auto
import tensorflow as tf
import tensorflow_probability as tfp

# generate true model for sampling
event_original = tfp.distributions.Bernoulli(probs=0.8, name="event_true")
dataset = event_original.sample(100)
# function argument
theta_event_fit = tfp.util.TransformedVariable(0.5, bijector = tfp.bijectors.SoftClip(low=0.0, high=1.0), name="theta_event_fit")
event_fit  = tfp.distributions.Bernoulli(probs=theta_event_fit, name="event_fit")
# function to minimize
neg_log_likelihood = lambda : -tf.reduce_sum(event_fit.log_prob(dataset))
# minimizer
convergence_hist = tfp.math.minimize(loss_fn=neg_log_likelihood, optimizer=tf.optimizers.Adam(0.1), num_steps=1000)
# closed form fit
tf.reduce_mean(tf.cast(dataset, float))
print(theta_event_fit)
# theta_event_fit will converge to closet for fit, but is not close to actual theta (0.8) due to lack of dataß