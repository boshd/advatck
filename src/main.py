import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cnn import CONVNetwork
from cw import CWAttack

def makeNoisy(data, stddev):
    '''
    Adds gaussian noise to normalized data
    '''
    for i, image in enumerate(data):
        mean = 0.0   # some constant
        stddev = stddev    # some constant (standard deviation)
        noisy_img = image + np.random.normal(mean, stddev, image.shape)
        # noisy_img_clipped = np.clip(noisy_img, 0, 1)  # we might get out of bounds due to noise`

        data[i] = noisy_img

    return data

def prepData():
    (trX, trY), (teX, teY) = tf.keras.datasets.mnist.load_data()

    trX = np.reshape(trX, [-1, 28, 28, 1])
    teX = np.reshape(teX, [-1, 28, 28, 1])
    trX = trX.astype('float32')
    teX = teX.astype('float32')

    trX /= 255
    teX /= 255

    trY = tf.keras.utils.to_categorical(trY)
    teY = tf.keras.utils.to_categorical(teY)
    ind = np.random.permutation(trX.shape[0])
    trX, trY = trX[ind], trY[ind]
    portion = int(trX.shape[0] * 0.9)
    vaX = trX[portion:]
    trX = trX[:portion]
    vaY = trY[portion:]
    trY = trY[:portion]

    return trX, trY, vaX, vaY, teX, teY

sess = tf.compat.v1.Session()

sess.run(tf.compat.v1.global_variables_initializer())

trX, trY, vaX, vaY, teX, teY = prepData()

sess.run(tf.compat.v1.local_variables_initializer())

cnn_tr_epochs = 5
network = CONVNetwork((trX, trY, vaX, vaY), sess, cnn_tr_epochs, 256)
cw = CWAttack((teX, teY), sess, network.graph)

network.train()
adv = cw.gen_attack(0.001)

network.get_acc(adv, cw.teY)

# cw.teY is different from teY because it is modified when used in the CWAttack instance.