
import tensorflow as tf
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator

dim = 10
A = np.diag(np.random.uniform(1, 2, (dim,)))
b = np.random.uniform(-1, 1, (dim,))
x_star = np.linalg.solve(A, b)

A_var = tf.Variable(A, dtype=tf.float32, trainable=False)
b_var = tf.Variable(b, dtype=tf.float32, trainable=False)
x = tf.Variable(np.random.normal(size=(dim,)), dtype=tf.float32)
vec = [tf.placeholder(dtype=tf.float32, shape=(dim,), name='vec')]
params = tf.trainable_variables()

delta_x = tf.placeholder(dtype=tf.float32, shape=(dim,))
update_x = tf.assign(x, x+delta_x)

A_x = tf.reduce_sum(x*A_var, axis=1)
quadratic_cost = tf.reduce_sum(x*A_x)
linear_cost = tf.reduce_sum(b*x)

cost = 0.5*quadratic_cost - linear_cost
grads = tf.gradients(cost, params)
grads_vec = [tf.reduce_sum(g*v) for g,v in zip(grads, vec)]
Hv = tf.gradients(grads_vec, params, stop_gradients=vec)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    def hessian_vector_product(v):
        assert(np.shape(v)==(dim,))
        return session.run(Hv, feed_dict={vec[0]: v})[0]

    def grad():
        return session.run(grads)[0]

    linear_operator = LinearOperator(shape=(dim, dim), matvec=hessian_vector_product)


    deltax, _ = cg(linear_operator, -grad())
    session.run(update_x, feed_dict={delta_x: deltax})

    print(session.run(x)-x_star)