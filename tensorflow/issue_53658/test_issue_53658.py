import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import tensorflow as tf
import pytest

def test_f():
    logits = tf.constant([[
        0.8407991, 0., 0.15966213, 0.8176781
    ]], dtype=tf.float32)

    r1 = tf.nn.softmax(logits, axis=-1)
    logits_sp = tf.sparse.from_dense(logits)
    r2 = tf.sparse.softmax(logits_sp)
    r3 = tf.sparse.to_dense(r2)

    assert not tf.reduce_all(tf.equal(r1, r3))
