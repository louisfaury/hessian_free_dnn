
import tensorflow as tf
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

dim = 10
data, target = load_boston(return_X_y=True)
_, data_dim = np.shape(data)

w = tf.Variable(np.random.normal(size=(data_dim,)), dtype=tf.float32)
b = tf.Variable(np.zeros(shape=(1,)), dtype=tf.float32)
x = tf.constant(data, dtype=tf.float32)
t = tf.constant(target, dtype=tf.float32)

params = tf.trainable_variables()
vec = [tf.placeholder(dtype=tf.float32, shape=tensor.get_shape()) for tensor in params]
delta = [tf.placeholder(dtype=tf.float32, shape=tensor.get_shape()) for tensor in params]

update = tf.group([tf.assign(p, p+d) for (p,d) in zip(params, delta)])

y = tf.reduce_sum(w*x, axis=1) + b
loss = tf.losses.mean_squared_error(t, y)

grads = tf.gradients(loss, params)
flat_grads = tf.concat([tf.reshape(tensor, [-1]) for tensor in grads], axis=0)
param_dim = flat_grads.get_shape()[0]

grads_vec = [tf.reduce_sum(g*v) for g,v in zip(grads, vec)]

Hv = tf.gradients(grads_vec, params, stop_gradients=vec)
flat_Hv = tf.concat([tf.reshape(tensor, [-1]) for tensor in Hv], axis=0)


def reshape_vector(vector):
    tensorized_vector =[]
    index = 0
    for tensor in grads:
        tensor_shape = tensor.get_shape()
        tensor_size = np.prod(tensor_shape)
        partial_vector = vector[index:index+tensor_size]
        tensorized_vector.append(np.reshape(partial_vector, tensor_shape))
        index += tensor_size
    return tensorized_vector


with tf.Session() as session:

    def hessian_vector_product(vector):
        assert(np.shape(vector)==(param_dim,))
        tensorized_vector = reshape_vector(vector)
        feed_dict = {tensor_elem: vector_elem for (tensor_elem, vector_elem) in zip(vec, tensorized_vector)}
        return session.run(flat_Hv, feed_dict=feed_dict)
    def flat_grad():
        return session.run(flat_grads)

    # Second Order
    session.run(tf.global_variables_initializer())
    linear_operator = LinearOperator(shape=(param_dim, param_dim), matvec=hessian_vector_product)
    hf_loss = []
    iters = 100
    for i in range(iters):
        hf_loss.append(session.run(loss))
        delta_theta, _ = cg(linear_operator, -flat_grad())
        tensor_delta_theta = reshape_vector(delta_theta)
        session.run(update, feed_dict={t_delta: v_delta for (t_delta, v_delta) in zip(delta, tensor_delta_theta)})
    print("\n")

    # First Order
    gd_loss = []
    optimizer = tf.train.AdamOptimizer(0.01)
    minimize = optimizer.minimize(loss)
    session.run(tf.global_variables_initializer())
    for i in range(iters):
        gd_loss.append(session.run(loss))
        session.run(minimize)


    plt.plot(np.arange(iters), hf_loss, label='hessian-free (2d order)')
    plt.plot(np.arange(iters), gd_loss, label='adam (1st order)')
    plt.legend()
    plt.show()