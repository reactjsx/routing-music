Some of my collegues, as well as many of my readers told me that they had problems using Tensorflow for their projects. Something like this:

> Hey Trung, what is the difference between tf.contrib.layers and tf.layers? Oh, and what is the tf.slim thing? And now we have the godd*** tf.estimator. What are all these for? What are we supposed to use?

To be honest, when I started using Tensorflow, I was in that situation too. Tensorflow was already pretty bulky back then, and to make things even worse, it just kept getting bigger and bigger. If you don't believe me, just look at the size of the installation file and compare it with previous versions.

So, I think I should create a series of blog posts about it :D

### Common Problems
Before diving in the details, I think I should list out the most common problems that we might face when using Tensorflow:
1. Don't know how to start

Whether it comes to importing data, creating the model or visualizing the results, we usually get confused. Technically, there are so many ways to do the exact same thing in Tensorflow. And it's the urge of doing things in the most proper way that drives us crazy.

2. Don't know what to do when things go wrong

I think this is the problem that a lot of you guys can relate to. In Tensorflow, we must first define the computation graph. Not only doing this way prevents us from modifying the graph when it's running (sometimes we just want it to be dynamic), but it also does a good job at hiding things from us, which we can't know what the hell under the hood is causing the trouble. We are the Python guys, we want things to be Pythonic!

(Talk about Eager Execution here)

### Tensorflow's Vocabulary
As I said above, one problem with Tensorflow is that there are a lot of ways to do the exact same thing. Even experienced users find it confusing sometimes.

Below, I'm gonna list out some "concepts" that are mostly confusing to beginners:

1. Low-level API

It used to be how we did everything in Tensorflow when it first came out. Want to create a fully connected layer? Create some weights, some biases and add them up!

```Python
with tf.variable_scopes('fc_layer'):
  weights = tf.get_variables(tf.float32, [in_size, out_size], 'weights')
  biases = tf.get_variables(tf.float32, [out_size], 'biases')
  output = tf.matmul(inputs, weights) + biases
```

2. tf.contrib

You are likely to come across tf.contrib.layers a lot. Basically, it's backed by the large community of Tensorflow (contribution) and it contains experimental implementation, such as a newly introduced layer, a new optimization method, or even wrappers for low-level API, etc. Although they are technically just experimental codes, they actually work well and will be merged to Tensorflow's core code in the future.

4. tf.layers

As its name is self explained, this is the package for defining layers. We can think of it as an official version of tf.contrib.layers. They basically do the same job: to make defining layers less tiresome. To create a fully-connected layer like we did above, we now need only one line:

```Python
output = tf.layers.dense(inputs, units=out_size, name='fc_layer')
```

5. tf.keras

This is legend! Keras came out when we had to write everything using low-level API. So technically it was used as the high-level API for Tensorflow. In my opinion, it did help make the community (especially researchers) adopt Tensorflow. And since Keras is officially a package of Tensorflow (from version 1.0 I think), you don't have to worry about version compatibility any more.

The great thing about Keras is, it does all the hard tasks for you. So implementing from idea to actual result is just a piece of cake. Want to create a network? Just stack up the layers! Want to train it? Just compile and fit!

```Python
model = Sequential()
model.add(Dense(out_size, activation='relu', input_shape=(in_size,)))
# Add more here
model.add(...)

# Compile
model.compile(loss='categorical_crossentropy', optimizer=SGD())

# Train
model.fit(x=inputs, y=labels)
```

6. tf.slim

Okay, this may be the most confusing one. At first, I thought that was the light-weight version of Tensorflow.

7. tf.estimator

8. tf.eager

9.  Tensorflow.js