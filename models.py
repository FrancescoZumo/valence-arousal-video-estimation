# -*- coding: utf-8 -*-
import pandas as pd
from numpy.testing._private.parameterized import param

import models_utils
import dataset_utils as ds_utils
import tensorflow as tf
import keras
import keras.layers as layers
import cloudpickle
from tensorflow.keras.applications import ResNet50
from keras.layers import Dense, Flatten
from contextlib import redirect_stdout
import numpy as np
import tensorflow_hub as hub
# Import the MoViNet model from TensorFlow Models (tf-models-official) for the MoViNet model
# from official.projects.movinet.modeling import movinet
# from official.projects.movinet.modeling import movinet_model


def run_training(x_train, x_val, y_train, y_val, accede_df, params, models, feat_indexes, model_name):

    """Data preparation"""

    # Create frames from each video file.

    # test = ds_utils.frames_or_features_from_video_file('/nas/home/fzumerle/LIRIS-ACCEDE/data/ACCEDE04894.mp4', 100,
    #                                                    frame_step=15, output_size=(224, 224))

    # Create the train set
    train_ds = tf.data.Dataset.from_generator(
        ds_utils.FrameGenerator(x_train, params, accede_df, feat_indexes=feat_indexes, training=True),
        output_signature=params['output_signature']
    )

    # Create the validation set
    val_ds = tf.data.Dataset.from_generator(
        ds_utils.FrameGenerator(x_val, params, accede_df, feat_indexes=feat_indexes),
        output_signature=params['output_signature']
    )

    # Print the shapes of the data
    ds_utils.get_next_frames_and_labels(train_ds, "train")
    ds_utils.get_next_frames_and_labels(val_ds, "validation")

    # Configure the dataset for performance

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    AUTOTUNE = tf.data.AUTOTUNE
    # if I just take one frame I want to randomize each time
    if params['cache']:
        train_ds = train_ds.cache().shuffle(10).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().shuffle(10).prefetch(buffer_size=AUTOTUNE)

    train_ds = train_ds.batch(params['batch_size'])
    val_ds = val_ds.batch(params['batch_size'])

    # Print again the shapes of the data (after batching)
    ds_utils.get_next_frames_and_labels(train_ds, "train")
    ds_utils.get_next_frames_and_labels(val_ds, "validation")

    """Models"""

    # MODEL 0 (basic, just to get started)
    if params['current_model'] == models[0]:
        net = tf.keras.applications.EfficientNetB0(include_top=False)
        net.trainable = True

        model = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Rescaling(scale=255),
            tf.keras.layers.TimeDistributed(net),
            tf.keras.layers.Dense(2),
            tf.keras.layers.GlobalAveragePooling3D(),
        ])

    # MODEL 1 - from tensorflow tutorial "Video classification with a 3D convolutional neural network"
    elif params['current_model'] == models[1]:

        input_shape = (None, params['frames_per_video'], params['frame_height'], params['frame_width'], 3)
        input = layers.Input(shape=(input_shape[1:]))
        x = input

        x = models_utils.Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = models_utils.ResizeVideo(params['frame_height'] // 2, params['frame_width'] // 2)(x)
        x = layers.Dropout(0.30)(x)

        # Block 1
        x = models_utils.add_residual_block(x, 16, (3, 3, 3))
        x = models_utils.ResizeVideo(params['frame_height'] // 4, params['frame_width'] // 4)(x)
        x = layers.Dropout(0.40)(x)

        # Block 2
        x = models_utils.add_residual_block(x, 32, (3, 3, 3))
        x = models_utils.ResizeVideo(params['frame_height'] // 8, params['frame_width'] // 8)(x)
        x = layers.Dropout(0.30)(x)

        # Block 3
        x = models_utils.add_residual_block(x, 64, (3, 3, 3))
        x = models_utils.ResizeVideo(params['frame_height'] // 16, params['frame_width'] // 16)(x)
        x = layers.Dropout(0.40)(x)

        # Block 4
        x = models_utils.add_residual_block(x, 128, (3, 3, 3))
        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(len(feat_indexes))(x)

        model = keras.Model(input, x)

        frames, label = next(iter(train_ds))
        model.build(frames)

        # Visualize the model
        print("generate model image with keras.utils.plot_model")
        keras.utils.plot_model(model, to_file='plots/' + model_name + '_model.pdf', expand_nested=True, dpi=500,
                               show_shapes=True)

    elif params['current_model'] == models[2]:

        input_shape = (None, params['frames_per_video'], params['frame_height'], params['frame_width'], 3)
        input_im = layers.Input(shape=(input_shape[1:]))
        x = layers.ZeroPadding3D(padding=(3, 3, 3))(input_im)   # changed to 3D

        # 1st stage
        # here we perform maxpooling, see the figure above
        x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(tf.keras.activations.relu)(x)
        x = layers.MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(x)    # changed to 3d

        # 2nd stage
        # frm here on only conv block and identity block, no pooling
        x = models_utils.res_conv(x, s=1, filters=(64, 256))
        x = models_utils.res_identity(x, filters=(64, 256))
        x = models_utils.res_identity(x, filters=(64, 256))

        # 3rd stage
        x = models_utils.res_conv(x, s=2, filters=(128, 512))
        x = models_utils.res_identity(x, filters=(128, 512))
        x = models_utils.res_identity(x, filters=(128, 512))
        x = models_utils.res_identity(x, filters=(128, 512))

        # 4th stage
        x = models_utils.res_conv(x, s=2, filters=(256, 1024))
        x = models_utils.res_identity(x, filters=(256, 1024))
        x = models_utils.res_identity(x, filters=(256, 1024))
        x = models_utils.res_identity(x, filters=(256, 1024))
        x = models_utils.res_identity(x, filters=(256, 1024))
        x = models_utils.res_identity(x, filters=(256, 1024))

        # 5th stage
        x = models_utils.res_conv(x, s=2, filters=(512, 2048))
        x = models_utils.res_identity(x, filters=(512, 2048))
        x = models_utils.res_identity(x, filters=(512, 2048))

        # ends with average pooling and dense connection
        x = layers.AveragePooling3D((2, 2, 2), padding='same')(x)   # changed to 3d

        x = layers.Flatten()(x)
        # output layer modified for regression
        # x = layers.Dense(8192, activation='relu')(x)
        x = layers.Dense(len(feat_indexes), activation='linear')(x)     # kernel_initializer='he_normal'

        # define the model
        model = keras.Model(inputs=input_im, outputs=x, name='Resnet50')

    elif params['current_model'] == models[3]:
        # RESNET50
        model = keras.Sequential()

        # 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
        # NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
        pretrained_model = ResNet50(
            include_top=False,
            pooling='avg',
            weights='imagenet',
            input_shape=(params['frame_height'], params['frame_width'], 3)
        )

        for layer in pretrained_model.layers:
            layer.trainable = False

        model.add(pretrained_model)
        model.add(Flatten())
        #model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        #model.add(Dense(256, activation='relu'))
        # 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
        model.add(Dense(len(feat_indexes), activation='linear'))

    elif params['current_model'] == models[4]:

        frame_features_input = keras.Input((params['frames_per_video'], params['num_features']))
        #mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

        # Refer to the following tutorial to understand the significance of using `mask`:
        # https://keras.io/api/layers/recurrent_layers/gru/
        x = keras.layers.GRU(16, return_sequences=True)(
            frame_features_input, #mask=mask_input
        )
        x = keras.layers.GRU(8)(x)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.Dense(8, activation="relu")(x)
        output = keras.layers.Dense(len(feat_indexes), activation="linear")(x)

        model = keras.Model(frame_features_input, output)
    elif params['current_model'] == models[5]:
        sequence_length = params['frames_per_video']
        embed_dim = params['num_features']
        dense_dim = 4
        num_heads = 1

        inputs = keras.Input(shape=(None, None))
        x = models_utils.PositionalEmbedding(
            sequence_length, embed_dim, name="frame_position_embedding"
        )(inputs)
        x = models_utils.TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(len(feat_indexes), activation="linear")(x)
        model = keras.Model(inputs, outputs)

    elif params['current_model'] == models[6]:
        input_shape = (None, params['frames_per_video'], params['num_features'])
        input_im = layers.Input(shape=(input_shape[1:]))

        model = keras.Sequential([
            input_im,
            layers.Flatten(),
            layers.Dense(2048, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(512, activation='relu'),
            layers.Dense(len(feat_indexes), activation='linear')
        ])

    # compile and fit the model
    model.compile(optimizer=params['optimizer'],
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanSquaredError(),
                           tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'models/' + model_name + '_checkpoint', save_weights_only=False, save_best_only=True, monitor='val_loss',
        verbose=1
    )

    checkpoint_tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='logs/' + model_name
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=params['patience'], monitor='val_loss', verbose=1
    )

    with open(model_name + "_summary.txt", 'w') as f:
        with redirect_stdout(f):
            model.summary()

    hist = model.fit(
        train_ds,
        epochs=params['n_epochs'],
        validation_data=val_ds,
        callbacks=[
            checkpoint,
            early_stop,
            checkpoint_tensorboard
        ]
    )

    # save history with cloudpickle and model with model.save
    train_hist_dict = {
        "history": hist.history,
        "params": params
    }

    with open('trainHistoryDicts/' + model_name, 'wb') as file_pi:
        cloudpickle.dump(train_hist_dict, file_pi)
    model.save('/nas/home/fzumerle/VA_video_estimation/models/' + model_name)

