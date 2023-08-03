# -*- coding: utf-8 -*-
import keras
import keras.layers as layers
import einops
from keras.applications.resnet import ResNet50
import tensorflow as tf


class Conv2Plus1D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding):
        """
          A sequence of convolutional layers that first apply the convolution operation over the
          spatial dimensions, and then the temporal dimension.
        """
        super().__init__()
        self.seq = keras.Sequential([
            # Spatial decomposition
            layers.Conv3D(filters=filters,
                          kernel_size=(1, kernel_size[1], kernel_size[2]),
                          padding=padding),
            # Temporal decomposition
            layers.Conv3D(filters=filters,
                          kernel_size=(kernel_size[0], 1, 1),
                          padding=padding)
        ])

    def call(self, x):
        return self.seq(x)

    def get_config(self):
        cfg = super().get_config()
        return cfg


class ResidualMain(keras.layers.Layer):
    """
      Residual block of the model with convolution, layer normalization, and the
      activation function, ReLU.
    """

    def __init__(self, filters, kernel_size):
        super().__init__()
        self.seq = keras.Sequential([
            Conv2Plus1D(filters=filters,
                        kernel_size=kernel_size,
                        padding='same'),
            layers.LayerNormalization(),
            layers.ReLU(),
            Conv2Plus1D(filters=filters,
                        kernel_size=kernel_size,
                        padding='same'),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)

    def get_config(self):
        cfg = super().get_config()
        return cfg


class Project(keras.layers.Layer):
    """
      Project certain dimensions of the tensor as the data is passed through different
      sized filters and downsampled.
    """

    def __init__(self, units):
        super().__init__()
        self.seq = keras.Sequential([
            layers.Dense(units),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)

    def get_config(self):
        cfg = super().get_config()
        return cfg


def add_residual_block(input, filters, kernel_size):
    """
      Add residual blocks to the model. If the last dimensions of the input data
      and filter size does not match, project it such that last dimension matches.
    """
    out = ResidualMain(filters,
                       kernel_size)(input)

    res = input
    # Using the Keras functional APIs, project the last dimension of the tensor to
    # match the new filter size
    if out.shape[-1] != input.shape[-1]:
        res = Project(out.shape[-1])(res)

    return layers.add([res, out])


class ResizeVideo(keras.layers.Layer):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width  # added .experimental.preprocessing
        self.resizing_layer = layers.Resizing(self.height, self.width)

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def call(self, video):
        """
          Use the einops library to resize the tensor.

          Args:
            video: Tensor representation of the video, in the form of a set of frames.

          Return:
            A downsampled size of the video according to the new height and width it should be resized to.
        """
        # b stands for batch size, t stands for time, h stands for height,
        # w stands for width, and c stands for the number of channels.
        old_shape = einops.parse_shape(video, 'b t h w c')
        images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
        images = self.resizing_layer(images)
        videos = einops.rearrange(
            images, '(b t) h w c -> b t h w c',
            t=old_shape['t'])
        return videos




# MODEL 2

def build_feature_extractor(img_size):
    # feature_extractor = ResNet50(
    #     weights="imagenet",
    #     include_top=False,
    #     pooling="avg",
    #     input_shape=(img_size, img_size, 3),
    # )

    feature_extractor = keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    preprocess_input = keras.applications.densenet.preprocess_input

    inputs = keras.Input((img_size, img_size, 3))
    preprocessed = preprocess_input(inputs)

    #outputs = tf.concat([feature_extractor(inputs), feature_extractor2(preprocessed)], 1)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


def get_sequence_model(class_vocab, params):

    frame_features_input = keras.Input((params['max_seq_length'], params['num_features']))
    mask_input = keras.Input((params['max_seq_length'],), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.GRU(16, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(2, activation="linear")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model


# model 3: resnet50

def res_identity(x, filters):
    #renet block where dimension does not change.
    #The skip connection is just simple identity conncection
    #we will have 3 blocks and then input will be added

    x_skip = x # this will be used for addition with the residual block
    f1, f2 = filters

    #first block
    x = layers.Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=keras.regularizers.L2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.relu)(x)

    #second block # bottleneck (but size kept same with padding)
    x = layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.L2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.relu)(x)

    # third block activation used after adding the input
    x = layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=keras.regularizers.L2(0.001))(x)
    x = layers.BatchNormalization()(x)
    # x = Activation(activations.relu)(x)

    # add the input
    x = layers.Add()([x, x_skip])
    x = layers.Activation(keras.activations.relu)(x)

    return x


def res_conv(x, s, filters):
    '''
    here the input size changes
    '''
    x_skip = x
    f1, f2 = filters

    # first block
    x = layers.Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=keras.regularizers.L2(0.001))(x)
    # when s = 2 then it is like downsizing the feature map
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.relu)(x)

    # second block
    x = layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.L2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.relu)(x)

    #third block
    x = layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=keras.regularizers.L2(0.001))(x)
    x = layers.BatchNormalization()(x)

    # shortcut
    x_skip = layers.Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=keras.regularizers.L2(0.001))(x_skip)
    x_skip = layers.BatchNormalization()(x_skip)

    # add
    x = layers.Add()([x, x_skip])
    x = layers.Activation(keras.activations.relu)(x)

    return x


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "output_dim": self.output_dim,
        })
        return config


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
        })
        return config