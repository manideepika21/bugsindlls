import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import tensorflow as tf
import pytest

def test_f():
    value_range = [0.0, 5.0]
    new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
    
    indices = tf.histogram_fixed_width_bins(new_values, value_range, nbins=-5)
    
    # Test passes if bug exists: negative nbins is accepted silently
    assert list(indices.numpy()) == [0, 0, 0, 0, 0, 0]
