
import tensorflow as tf
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

np.random.seed(0)
tf.set_random_seed(0)
data, target = load_boston(return_X_y=True)
_, data_dim = np.shape(data)


x = tf.constant(data, dtype=tf.float32)
t = tf.constant(target, dtype=tf.float32)

h1 = Dense(16, activation='tanh')(x)
h2 = Dense(16, activation='tanh')(h1)
y = tf.reshape(Dense(1, activation='linear')(h2), (-1,))
loss = tf.losses.mean_squared_error(t, y)

params = tf.trainable_variables()
vec = [tf.placeholder(dtype=tf.float32, shape=tensor.get_shape()) for tensor in params]
delta = [tf.placeholder(dtype=tf.float32, shape=tensor.get_shape()) for tensor in params]

update = tf.group([tf.assign(p, p+d) for (p,d) in zip(params, delta)])


grads = tf.gradients(loss, params)
flat_grads = tf.concat([tf.reshape(tensor, [-1]) for tensor in grads], axis=0)
param_dim = flat_grads.get_shape()[0]

grads_vec = [tf.reduce_sum(g*v) for g,v in zip(grads, vec)]

Hv = tf.gradients(grads_vec, params, stop_gradients=vec)
flat_Hv = tf.concat([tf.reshape(tensor, [-1]) for tensor in Hv], axis=0)

tikhonov_damping = 1.0


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
        hessian_vec_product = session.run(flat_Hv, feed_dict=feed_dict)
        return hessian_vec_product

    def damped_hessian_vector_product(vector):
        damped_hessian_vec_product = hessian_vector_product(vector) + tikhonov_damping * vector
        return damped_hessian_vec_product

    def flat_grad():
        return session.run(flat_grads)

    # Second Order
    session.run(tf.global_variables_initializer())
    linear_operator = LinearOperator(shape=(param_dim, param_dim), matvec=damped_hessian_vector_product)
    hf_loss = []
    hf_iters = 50
    delta_theta = np.zeros((param_dim,))
    for i in range(hf_iters):

        old_loss = session.run(loss)
        hf_loss.append(old_loss)
        delta_theta, _ = cg(linear_operator, -flat_grad(), x0=delta_theta, maxiter=50)
        tensor_delta_theta = reshape_vector(delta_theta)
        session.run(update, feed_dict={t_delta: v_delta for (t_delta, v_delta) in zip(delta, tensor_delta_theta)})

        q_theta = old_loss + np.sum(delta_theta*flat_grad()) + 0.5*np.sum(delta_theta*hessian_vector_product(delta_theta))
        new_loss = session.run(loss)
        reduction_ratio = (new_loss-old_loss)/(q_theta-old_loss)

        # Levenberg heuristic for damping
        print(reduction_ratio)
        if reduction_ratio < 0.25:
            tikhonov_damping *= 1.5
        if reduction_ratio > 0.75:
            tikhonov_damping *= 2. / 3.

    # First Order
    gd_loss = []
    optimizer = tf.train.AdamOptimizer(0.01)
    minimize = optimizer.minimize(loss)
    session.run(tf.global_variables_initializer())
    gd_iters = 200
    for i in range(gd_iters):
        gd_loss.append(session.run(loss))
        session.run(minimize)


    plt.plot(np.arange(hf_iters), hf_loss, label='hessian-free (2d order)')
    plt.plot(np.arange(gd_iters), gd_loss, label='adam (1st order)')
    plt.legend()
    plt.show()