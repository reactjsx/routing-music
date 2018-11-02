Some of my collegues, as well as many of my readers told me that they had problems using Tensorflow for their projects. Something like this:

> Hey Trung, what is the difference between tf.contrib.layers and tf.layers? Oh, and what is the TF-Slim thing? And now we have the godd*** tf.estimator. What are all these for? What are we supposed to use?

To be honest, when I started using Tensorflow, I was in that situation too. Tensorflow was already pretty bulky back then, and to make things even worse, it just kept getting bigger and bigger. If you don't believe me, just look at the size of the installation file and compare it with previous versions.

But then, I suddenly got an idea. Why not create a series of blog posts about Tensorflow ;)

### Objectives
Let's talk about what we're gonna focus on in this post. I learned this thing the hardest way, guys. I think I will make a post about what I learned from writing technical blog posts, and one of them is: do talk about the objectives first!

So, here's what we will do in this post:
- Address some confusing problems of Tensorflow
- Understand the mostly used Tensorflow modules
- (Optional) Get out hands dirty with some easy code!

Okay, let's tackle them one by one!

### Common Problems
Before diving in the details, I think I should list out the most common problems that we might face when using Tensorflow:
1. Don't know how to start

Whether it comes to importing data, creating the model or visualizing the results, we usually get confused. Technically, there are so many ways to do the exact same thing in Tensorflow. And it's the urge of doing things in the most proper way that drives us crazy.

2. Don't know what to do when things go wrong

I think this is the problem that a lot of you guys can relate to. In Tensorflow, we must first define the computation graph. Not only doing this way prevents us from modifying the graph when it's running (sometimes we just want it to be dynamic), but it also does a good job at hiding things from us, which we can't know what the hell under the hood is causing the trouble. We are the Python guys, we want things to be Pythonic!

### Tensorflow's Vocabulary
As I said above, one problem with Tensorflow is that there are a lot of ways to do the exact same thing. Even experienced users find it confusing sometimes.

Below, I'm gonna list out some "concepts" that are mostly confusing to beginners:

1. Low-level API

It used to be how we did everything in Tensorflow when it first came out. Want to create a fully connected layer? Create some weights, some biases and roll them in!

```Python
with tf.variable_scopes('fc_layer'):
  weights = tf.get_variables('weights', [5, 5, 3, 64],
                             initializer=tf.truncated_normal_initializer(stddev=5e-2))
  biases = tf.get_variables('biases', [64],
                            initializer=tf.zeros_initializer)
  output = tf.nn.conv2d(inputs, weights, [1, 1, 1, 1], padding='SAME')
  output = tf.nn.bias_add(conv, biases)
  output = tf.nn.relu(output)
```

2. tf.contrib

You are likely to come across tf.contrib.layers a lot. Basically, it's backed by the large community of Tensorflow (contribution) and it contains experimental implementation, such as a newly introduced layer, a new optimization method, or even wrappers for low-level API, etc. Although they are technically just experimental codes, they actually work well and will be merged to Tensorflow's core code in the future.

4. tf.layers

As its name is self explained, this is the package for defining layers. We can think of it as an official version of tf.contrib.layers. They basically do the same job: to make defining layers less tiresome. Using tf.contrib.layers or tf.layers to create a conv2d layer like we did above, we now need only one line:

```Python
output = tf.contrib.layers.conv2d(inputs, 64, [5, 5],
                                  weights_initializer=tf.glorot_normal_initializer)
```

Or with tf.layers:

```Python
output = tf.layers.conv2d(inputs, 64, [5, 5],
                          padding='same',
                          kernel_initializer=tf.glorot_normal_initializer)
```

I bet you wouldn't create any layers by hand from now on!

5. tf.contrib.slim (or TF-Slim)

Okay, this may be the most confusing one. At first, I thought that was the light-weight version of Tensorflow but soon enough, I realized I was wrong. **slim** only stands for fewer lines to do the same thing (comparing with low-level API). For example, to create not only one, but three conv2d layers, we only need to write one line:

```Python
slim = tf.contrib.slim
output = slim.repeat(inputs, 3, slim.conv2d, 64, [5, 5])
```

Other than that though, tf.slim can help you build an entire pipeline for training the network, which means they have some functions to help you get the losses, train or evaluate the model, etc.

Overall, TF-Slim may be a good option for fast experimenting new idea (Tensorflow's research uses TF-Slim for building the networks). What you need to take into account is, TF-Slim's codes actually came from tf.contrib (e.g. slim.conv2d is just an alias for tf.contrib.layers.conv2d), so there's no magic here.

6. tf.keras

This is legend! Keras came out when we had to write everything using low-level API. So technically it was used as the high-level API for Tensorflow. In my opinion, it did help make the community (especially researchers) adopt Tensorflow. And since Keras is officially a module of Tensorflow (from version 1.0 I think), you don't have to worry about version compatibility any more.

The great thing about Keras is, it does all the hard tasks for you. So going from idea to result is just a piece of cake. Want to create a network? Just stack up the layers! Want to train it? Just compile and call fit!

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

Although Keras is super convenient, especially for those who don't like to write code, it abstracts so many things from us. François Chollet, the author of Keras, claimed that Keras will act like an interface only, but it does have some constraints which may confuse you sometimes (Want model.fit to compute validation loss after a specific number of batches? It can't!). You may also have hard time implementing newly introduced deep-learning papers entirely by Keras since they require some minor tweaks within some layer.

7. Eager Execution

As I mentioned earlier, when implementing in Tensorflow, you must first define all the operations to form a graph. It's not until the graph is finalized (it's locked, no more in, no more out, no more update) that you can run it to see the results. Because of this, Tensorflow is hard to debug and incapable of creating dynamic graph.

So Eager Execution came out to help deal with these problems. The name is kind of weird though. I interprete it as "can't wait to execute". With the additional 2 new lines, you can now do something like: evaluate the new created variable (which is trivial but used to be impossible in Tensorflow):

```Python
tf.enable_eager_execution()
tf.executing_eagerly()
import tensorflow.contrib.eager as tfe

weights = tfe.Variable(tf.truncated_normal(shape=[2, 3], stddev=5e-2), name='weights')
weights
<tf.Variable 'weights:0' shape=(2, 3) dtype=float32, numpy=
array([[ 0.06691323, -0.01890625, -0.00283119],
       [-0.0536754 ,  0.00109388, -0.04310168]], dtype=float32)>
```

Rumor has it Eager Execution is gonna be set to default from Tensorflow 2.0. I think this move will please a lot of Tensorflow fans out there. But please bear in mind that at the moment, not everything is gonna work in Eager Execution mode (yet). So while we're waiting for Tensorflow 2.0 to be released, it's a good idea to stay updated to the latest news from Tensorflow's team and Google.

### (Optional) Let's play with Tensors!
Okay guys, this is an optional section. We're gonna see if different approaches produce exactly the same results. We're gonna create a "real" convolution2d layer, including activation functions and regularization terms, by using tf.contrib and tf.layers. We will check the similarity among their results by checking the variables and operations that they created.

Oh hold on! There's one more thing I want you to pay attention to. I will write out all the arguments whether some of them have default values. The reason is, the two modules' conv2d functions set the default values differently for the same terms! For example, padding is set to 'SAME' by default in tf.contrib.layers.conv2d, but 'valid' in case of tf.layers.conv2d. Now we're ready to move on.

1. tf.contrib

Let's start with tf.contrib. I don't want to think of the amount of work to achieve the same result by using low-level API. That's why having any kinds of high-level API will save us a ton of time and effort. Not only researchers, developers do love high-level APIs!

```Python
# The inputs we use is one image of shape (224, 224, 3)
inputs = tf.placeholder(tf.float32, [1, 224, 224, 3])

conv2d = tf.contrib.layers.conv2d(inputs=inputs,
                                  num_outputs=64,
                                  kernel_size=3,
                                  stride=1, 
                                  padding='SAME', 
                                  activation_fn=tf.nn.relu,
                                  weights_initializer=tf.initializers.truncated_normal(stddev=0.01),
                                  weights_regularizer=tf.contrib.layers.l2_regularizer(0.005),
                                  biases_initializer=tf.zeros_initializer())
```
2. tf.layers

Next, let's see how we can create a convolution2d layer with tf.layers, an official modules by the core team of Tensorflow ;) Obviously we at least expect that it can produce the same result, with less or similar or effort.

```Python
conv2d = tf.layers.conv2d(inputs=inputs,
                          filters=64,
                          kernel_size=3,
                          strides=1,
                          padding='same',
                          activation=tf.nn.relu,
                          kernel_initializer=tf.initializers.truncated_normal(stddev=0.01),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.005),
                          bias_initializer=tf.zeros_initializer())
```

It's time to compare the results. Did both of tf.contrib and tf.layers produce the layers with similar functionality? Did one of them do more than the other?

First, let's consider the variables created by above commands. (You can use the method tf.global_variables() to get all variables in the current graph)

```Python
# Variables created by tf.contrib.layers.conv2d
[<tf.Variable 'Conv/weights:0' shape=(3, 3, 3, 64) dtype=float32_ref>, <tf.Variable 'Conv/biases:0' shape=(64,) dtype=float32_ref>]

# Variables created by tf.contrib.layers.conv2d
[<tf.Variable 'conv2d_1/kernel:0' shape=(3, 3, 3, 64) dtype=float32_ref>, <tf.Variable 'conv2d_1/bias:0' shape=(64,) dtype=float32_ref>]
```
Phew, the variable sets are similar. They both created a weights Tensor, and a biases Tensor with the same shape. Notice that their names are slightly different, though.

Next, let's check if the two functions generated different sets of operations. (The command we can use is 
tf.get_default_graph().get_operations())

```Python
# Operations created by tf.contrib.layers.conv2d
<tf.Operation 'Placeholder' type=Placeholder>, 
<tf.Operation 'Conv/weights/Initializer/truncated_normal/shape' type=Const>, 
<tf.Operation 'Conv/weights/Initializer/truncated_normal/mean' type=Const>, 
<tf.Operation 'Conv/weights/Initializer/truncated_normal/stddev' type=Const>, 
<tf.Operation 'Conv/weights/Initializer/truncated_normal/TruncatedNormal' type=TruncatedNormal>, 
<tf.Operation 'Conv/weights/Initializer/truncated_normal/mul' type=Mul>, 
<tf.Operation 'Conv/weights/Initializer/truncated_normal' type=Add>, 
<tf.Operation 'Conv/weights' type=VariableV2>, 
<tf.Operation 'Conv/weights/Assign' type=Assign>, 
<tf.Operation 'Conv/weights/read' type=Identity>, 
<tf.Operation 'Conv/kernel/Regularizer/l2_regularizer/scale' type=Const>, 
<tf.Operation 'Conv/kernel/Regularizer/l2_regularizer/L2Loss' type=L2Loss>, 
<tf.Operation 'Conv/kernel/Regularizer/l2_regularizer' type=Mul>, 
<tf.Operation 'Conv/biases/Initializer/zeros' type=Const>, 
<tf.Operation 'Conv/biases' type=VariableV2>, 
<tf.Operation 'Conv/biases/Assign' type=Assign>, 
<tf.Operation 'Conv/biases/read' type=Identity>, 
<tf.Operation 'Conv/dilation_rate' type=Const>, 
<tf.Operation 'Conv/Conv2D' type=Conv2D>, 
<tf.Operation 'Conv/BiasAdd' type=BiasAdd>, 
<tf.Operation 'Conv/Relu' type=Relu>

# Operations created by tf.layers.conv2d
<tf.Operation 'Placeholder' type=Placeholder>, 
<tf.Operation 'conv2d_1/kernel/Initializer/truncated_normal/shape' type=Const>, 
<tf.Operation 'conv2d_1/kernel/Initializer/truncated_normal/mean' type=Const>, 
<tf.Operation 'conv2d_1/kernel/Initializer/truncated_normal/stddev' type=Const>, 
<tf.Operation 'conv2d_1/kernel/Initializer/truncated_normal/TruncatedNormal' type=TruncatedNormal>, 
<tf.Operation 'conv2d_1/kernel/Initializer/truncated_normal/mul' type=Mul>, 
<tf.Operation 'conv2d_1/kernel/Initializer/truncated_normal' type=Add>, 
<tf.Operation 'conv2d_1/kernel' type=VariableV2>, 
<tf.Operation 'conv2d_1/kernel/Assign' type=Assign>, 
<tf.Operation 'conv2d_1/kernel/read' type=Identity>, 
<tf.Operation 'conv2d_1/kernel/Regularizer/l2_regularizer/scale' type=Const>, 
<tf.Operation 'conv2d_1/kernel/Regularizer/l2_regularizer/L2Loss' type=L2Loss>, 
<tf.Operation 'conv2d_1/kernel/Regularizer/l2_regularizer' type=Mul>, 
<tf.Operation 'conv2d_1/bias/Initializer/zeros' type=Const>, 
<tf.Operation 'conv2d_1/bias' type=VariableV2>, 
<tf.Operation 'conv2d_1/bias/Assign' type=Assign>, 
<tf.Operation 'conv2d_1/bias/read' type=Identity>, 
<tf.Operation 'conv2d_1/dilation_rate' type=Const>, 
<tf.Operation 'conv2d_1/Conv2D' type=Conv2D>, 
<tf.Operation 'conv2d_1/BiasAdd' type=BiasAdd>, 
<tf.Operation 'conv2d_1/Relu' type=Relu>
```

Now, what is the verdict? As we can observe above. Using tf.contrib or tf.layers will save us a lot of time and prevent us from headache later on. Moreover, they create absolutely similar things. What does that mean to us? It means that it doesn't matter what your preferred module is, you can create/re-create any networks or you can even use the weights trained by the code written on the other module.

But hey, you can't see that the names are obviously different, can you? You might ask. As long as the shapes and types of variables are the same, mapping the names between the two variable sets is not that painful task. In fact, it's just no more than 5 lines of code and yeah, you only need to know how to do it. As we addressed the problems earlier in this post, Tensorflow is not hard, it is just kind of confusing.

### Conclusion
Oh, that was so long. Thank you guys for reading. Before we say goodbye, let's take a look at what we did in this post:
- We discussed why Tensorflow may seem confusing
- We talked about heavily in-use Tensorflow module
- We checked if different modules produce different results on the same task

This post is no more than an entry point, some kind of what-I-would-talk-about-when-I-talk-about-Tensorflow (I borrowed that title from Haruki Murakami, check it [here](https://www.amazon.com/What-About-Running-Vintage-International-ebook/dp/B0015DWJ8W).

Personally, I am a big fan of learning-by-doing style. Even in the deep-learning field which seems way to deeply academic, it can work out well. In the future posts, I will guide you through how we can accomplish the most common tasks with Tensorflow. Those won't make you a deep learning guru, but with a solid understanding about how to use the proper tool, and with some practice from your own, what can stop you from building amazing things?

Okay, I might have exaggerated a little bit, but honestly I hope that I can make something that you guys can benefit from. So, I'm gonna see you soon, in the next blog post of this Tensorflow series.
