import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
print(tf.__version__)
for _ in range(20):
    try:
        with tf.device("GPU:0"):
            method = "bilinear"
            extrapolation_value = -90.71406992626788
            image = tf.saturate_cast(tf.random.uniform([1, 4, 1, 8], minval=0, maxval=64, dtype=tf.int64), dtype=tf.int64)
            boxes = tf.random.uniform([4, 0], dtype=tf.float32)
            box_ind = tf.saturate_cast(tf.random.uniform([8, 0, 10, 4], minval=0, maxval=64, dtype=tf.int64), dtype=tf.int32)
            crop_size = tf.saturate_cast(tf.random.uniform([2], minval=0, maxval=64, dtype=tf.int64), dtype=tf.int32)
            res = tf.raw_ops.CropAndResize(
                method=method,
                extrapolation_value=extrapolation_value,
                image=image,
                boxes=boxes,
                box_ind=box_ind,
                crop_size=crop_size,
            )
    except:
        pass