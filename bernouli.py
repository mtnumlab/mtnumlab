import silence_tensorflow.auto
import tensorflow as tf
import tensorflow_probability as tfp

# create a model with known probability
event = tfp.distributions.Bernoulli(probs=0.8)
sample = event.sample(100)
sample_mean = tf.reduce_mean(tf.cast(sample, float))
print(sample_mean)