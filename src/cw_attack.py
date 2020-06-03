'''
CW

refs //
    - https://github.com/carlini/nn_robust_attacks
    - https://arxiv.org/pdf/1608.04644.pdf
'''
import numpy as np
import tensorflow as tf

class CWAttack(object):

    def __init__(self, data, sess, model):
        self.teX, self.teY = data
        self.sess = sess
        self.model = model

        self.epochs = 100
        self.sample_size = 128
        self.ind = np.random.choice(self.teX.shape[0], size=self.sample_size, replace=False)
        self.teX = self.teX[self.ind]
        self.teY = self.teY[self.ind]

        self.norm_ = 2
        self.temperature = 2
        self.const_bounds = (0.0, 1.0)
        self.alpha = 0.9
        self.min_prob = 0
        self.eps_val = 1.0

        self.X = tf.placeholder(tf.float32, (32, 28, 28, 1), name='X')
        self.eps = tf.placeholder(tf.float32, (), name='eps')
        self.Y = tf.placeholder(tf.int32, (), name='Y')

        self.at_op = tf.train.AdamOptimizer(learning_rate=0.1)

        self.adv_op, self.xadv, self.noise = self.attack_model()

    def attack_model(self):
        '''
        this CW model implementation was taking from 'https://github.com/lishiyuwhu/IMNS-Li/blob/db9dabe7d4d67051d3d03b034c40cdf0ff4ddb5a/Adversarial_sample/more%20method/attacks/cw.py', the specifics are explained in the paper. There are many more implemntations, including Carlini's one but this one had the best implemntation in my opinion.
        '''
        xshape = self.X.get_shape().as_list()
        noise = tf.get_variable('noise', xshape, tf.float32, initializer=tf.initializers.zeros)

        x_scaled = (self.X - self.const_bounds[0]) / (self.const_bounds[1] - self.const_bounds[0])

        z = tf.clip_by_value(x_scaled, 1e-8, 1-1e-8)
        xinv = tf.log(z / (1 - z)) / self.temperature

        xadv = tf.sigmoid(self.temperature * (xinv + noise))
        xadv = xadv * (self.const_bounds[1] - self.const_bounds[0]) + self.const_bounds[0]

        ybar, logits = self.model(xadv)

        ydim = ybar.get_shape().as_list()[1]

        y = tf.cond(tf.equal(tf.rank(self.Y), 0), lambda: tf.fill([xshape[0]], self.Y), lambda: tf.identity(self.Y))

        mask = tf.one_hot(y, ydim, on_value=0.0, off_value=float('inf'))

        yt = tf.reduce_max(logits - mask, axis=1)
        yo = tf.reduce_max(logits, axis=1)

        loss0 = tf.nn.relu(yo - yt + self.min_prob)

        axis = list(range(1, len(xshape)))
        ord_ = float(self.norm_)

        loss1 = tf.reduce_mean(tf.square(xadv-self.X))
        loss = self.eps_val * loss0 + loss1

        k_op = self.at_op.minimize(loss, var_list=[noise])

        return k_op, xadv, noise

    def gen_attack(self, eps_):
        '''
        generate adv samples
        '''
        print('generating attack...')
        adv = np.empty_like(self.teX)

        batch = zip(range(0, len(self.teX), 32), range(32, len(self.teX) + 1, 32))
        for i, (start, end) in enumerate(batch):

            self.sess.run(self.noise.initializer) # .. okk?
            for epoch in range(self.epochs):
                rand = np.random.choice(10)
                self.sess.run(self.adv_op, feed_dict={
                    self.X: self.teX[start:end],
                    self.Y: rand,
                    self.eps: eps_
                })

            adv[start:end] = self.sess.run(self.xadv, feed_dict={
                self.X: self.teX[start:end],
                self.Y: rand,
                self.eps: eps_
            })

        return adv