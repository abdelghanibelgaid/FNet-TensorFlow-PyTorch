"""
FNet Implementation with TensorFlow
"""

import tensorflow as tf
from tensorflow.keras import layers, Sequential


class FeedForward(layers.Layer):
    
    def __init__(self, dense_dim, dropout_rate, **kwargs):
        super(FeedForward, self).__init__()
        self.dense_dim = dense_dim
        self.dropout_rate = dropout_rate
        self.supports_masking = True
        self.dense_1 = layers.Dense(dense_dim, activation=tf.nn.gelu)
        self.dense_2 = layers.Dense(dense_dim)
        self.dropout = layers.Dropout(self.dropout_rate)
    
    def get_config(self):
        config = super(FeedForward, self).get_config()
        config.update({"dense_dim": self.dense_dim,
                       "dropout_rate": self.dropout_rate})
        return config
    
    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dropout(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x

class FNetEncoder(layers.Layer):
    
    def __init__(self, hidden_dim):
        super(FNetEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.supports_masking = True
        self.LayerNorm_1 = layers.LayerNormalization()
        self.LayerNorm_2 = layers.LayerNormalization()
        self.feedforward = FeedForward(self.hidden_dim, 0)
        
    def get_config(self):
        config = super(FNetEncoder, self).get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config
    
    def call(self, inputs):
        x_complex = tf.cast(inputs, tf.complex64)
        x_fft = tf.math.real(tf.signal.fft2d(x_complex))
        x_norm_1 = self.LayerNorm_1(x_fft + inputs)
        x_dense = self.feedforward(x_norm_1)
        x_norm_2 = self.LayerNorm_2(x_dense + x_norm_1)
        return x_norm_2

if __name__ == "__main__":
    model = FNetEncoder(16)
    x = tf.random.normal((32, 16))
    y = model(x)
