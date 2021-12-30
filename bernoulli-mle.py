import silence_tensorflow.auto
import tensorflow as tf
import tensorflow_probability as tfp

# create a model with known probability
event = tfp.distributions.Bernoulli(probs=0.8)
sample_1h = event.sample(100)
sample_mean_1h = tf.reduce_mean(tf.cast(sample_1h, float))
print(sample_mean_1h)

sample_1m = event.sample(1000000)
sample_mean_1m = tf.reduce_mean(tf.cast(sample_1m, float))
print(sample_mean_1m)

assert abs(sample_mean_1m - 0.8) < abs(sample_mean_1h - 0.8)