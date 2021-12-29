# Formulas:
# probability of dataset with bernoulli distribution
# p(D) = II (over i = 1, N-1) pow(p, i) pow(1-p, 1-i)
# w ~ Bernoulli(p)
# i E {0,1}
import silence_tensorflow.auto
import tensorflow as tf
import tensorflow_probability as tfp

event = tfp.distributions.Bernoulli(probs=0.8, name="event")

# Create dataset
dataset = event.sample(10)
print(dataset)

# Probability of the dataset
prob_dataset = tf.reduce_prod(event.prob(dataset))
print(prob_dataset)
