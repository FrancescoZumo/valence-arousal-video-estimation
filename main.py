# -*- coding: utf-8 -*-

# The way this tutorial uses the `TimeDistributed` layer requires TF>=2.10
import dataset_utils as ds_utils
import models_utils
import plots
import os
import pandas as pd
import keras
from keras import layers
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cloudpickle
from os.path import isfile, join
import time

train_model = False
verbose = 0

models = {
    0: "dummy",
    1: "3D_CNN",
    2: "CNN_RNN"
}

params = dict(
    current_model=models[0],
    frames_per_video=10,  # frames per video, default 10
    # Define the dimensions of one frame in the set of frames created
    HEIGHT=224,
    WIDTH=224,
    n_epochs=10,
    batch_size=32,  # default 32
    patience=20,
    learning_rate=1e-4,
    dataset_size=120,  # full dataset: 9800 videos
)
params['train_size'] = int(params['dataset_size'] * 0.7)
params['val_test_size'] = int(params['dataset_size'] * 0.3)
params['optimizer'] = tf.keras.optimizers.Adam(
    learning_rate=params['learning_rate'], beta_1=0.9, beta_2=0.999, amsgrad=False  # default False
)

if params['current_model'] == models[0]:
    model_name = params['current_model']
else:
    model_name = params['current_model'] + '_pat' + str(params['patience']) + '_lr' + str(params['learning_rate'])

print('Model name: ', model_name)
time.sleep(1)
if train_model:
    print('Training will begin with following parameters:\n')
    for i in params:
        print('{}:'.format(i), params[i])
else:
    print('The current model will just be loaded and evaluated')
time.sleep(1)

# maybe useless, could be removed
files = [f for f in os.listdir('/nas/home/fzumerle/data') if isfile(join('/nas/home/fzumerle/data', f))]
files.sort()

accede_df = pd.read_csv(r'/nas/home/fzumerle/LIRIS-ACCEDE-annotations/annotations/ACCEDEranking.txt', sep='\t')

# normalize va labels
accede_df = ds_utils.normalize_columns(accede_df, columns=[4, 5], verbose=1)

if verbose:
    print('check that functions are built correctly:')
    print('row 0 of annotation file:\n\n', accede_df.loc[accede_df['name'] == files[0]], '\n')
    print('valence, arousal, valenceVariance, arousalVariance extracted from file 0:\n\n', files[0],
          ds_utils.get_valence(files[0], accede_df), ds_utils.get_arousal(files[0], accede_df),
          ds_utils.get_valence_va(files[0], accede_df), ds_utils.get_arousal_va(files[0], accede_df))

# get list of videos
unzip_folder = '/nas/home/fzumerle/data/'
video_list = [unzip_folder + f for f in os.listdir(unzip_folder) if
              isfile(join(unzip_folder, f))]
video_list.sort()
# get list of labels
va_labels = accede_df[['valenceValue', 'arousalValue']].values.tolist()

# split dataset in train and test
X_train, X_test, y_train, y_test = train_test_split(video_list, va_labels, train_size=params['train_size'],
                                                    test_size=params['val_test_size'], random_state=42)

# split test in validation and test
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# check total of videos imported
video_total = len(X_train) + len(X_val) + len(X_test)
print(f"Total videos: {video_total}")

# Create the training set
output_signature = (tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=2, dtype=tf.float32))

if train_model:

    """## Create frames from each video file"""

    # Create the train set
    train_ds = tf.data.Dataset.from_generator(
        ds_utils.FrameGenerator(X_train, params['frames_per_video'], accede_df, training=True),
        output_signature=output_signature)

    # Create the validation set
    val_ds = tf.data.Dataset.from_generator(ds_utils.FrameGenerator(X_val, params['frames_per_video'], accede_df),
                                            output_signature=output_signature)

    # Print the shapes of the data
    ds_utils.print_dataset_shapes(train_ds, "train")
    ds_utils.print_dataset_shapes(val_ds, "validation")

    """##Configure the dataset for performance"""

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    AUTOTUNE = tf.data.AUTOTUNE
    # changed to 10 shuffle

    train_ds = train_ds.cache().shuffle(10).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().shuffle(10).prefetch(buffer_size=AUTOTUNE)

    train_ds = train_ds.batch(params['batch_size'])
    val_ds = val_ds.batch(params['batch_size'])

    # Print again the shapes of the data (after batching)
    ds_utils.print_dataset_shapes(train_ds, "train")
    ds_utils.print_dataset_shapes(val_ds, "validation")

    # MODEL 1 (basic, just to get started)
    if params['current_model'] == models[0]:
        net = tf.keras.applications.EfficientNetB0(include_top=False)
        net.trainable = False

        model = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Rescaling(scale=255),
            tf.keras.layers.TimeDistributed(net),
            tf.keras.layers.Dense(2),
            tf.keras.layers.GlobalAveragePooling3D(),
        ])

    # MODEL 2 - from tensorflow tutorial "Video classification with a 3D convolutional neural network"
    elif params['current_model'] == models[1]:

        input_shape = (None, params['frames_per_video'], params['HEIGHT'], params['WIDTH'], 3)
        input = layers.Input(shape=(input_shape[1:]))
        x = input

        x = models_utils.Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = models_utils.ResizeVideo(params['HEIGHT'] // 2, params['WIDTH'] // 2)(x)

        # Block 1
        x = models_utils.add_residual_block(x, 16, (3, 3, 3))
        x = models_utils.ResizeVideo(params['HEIGHT'] // 4, params['WIDTH'] // 4)(x)

        # Block 2
        x = models_utils.add_residual_block(x, 32, (3, 3, 3))
        x = models_utils.ResizeVideo(params['HEIGHT'] // 8, params['WIDTH'] // 8)(x)

        # Block 3
        x = models_utils.add_residual_block(x, 64, (3, 3, 3))
        x = models_utils.ResizeVideo(params['HEIGHT'] // 16, params['WIDTH'] // 16)(x)

        # Block 4
        x = models_utils.add_residual_block(x, 128, (3, 3, 3))

        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Flatten()(x)
        # x = layers.Dense(10)(x)
        x = layers.Dense(2)(x)  # modified for getting two values

        model = keras.Model(input, x)

        frames, label = next(iter(train_ds))
        model.build(frames)

        # Visualize the model
        keras.utils.plot_model(model, expand_nested=True, dpi=60, show_shapes=True)

    elif params['current_model'] == models[2]:
        # TODO
        print("not implemented yet")

    # compile and fit the model
    model.compile(optimizer=params['optimizer'],
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanSquaredError(),
                           tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])

    hist = model.fit(train_ds,
                     epochs=params['n_epochs'],
                     validation_data=val_ds,
                     callbacks=tf.keras.callbacks.EarlyStopping(patience=params['patience'], monitor='val_loss',
                                                                verbose=1))

    # save history with cloudpickle and model with model.save
    train_hist_dict = {
        "history": hist.history,
        "params": params
    }

    with open('trainHistoryDicts/' + model_name, 'wb') as file_pi:
        cloudpickle.dump(train_hist_dict, file_pi)
    model.save('/nas/home/fzumerle/VA_video_estimation/models/' + model_name)


with open('trainHistoryDicts/' + model_name, 'rb') as file_pi:
    train_hist_dict = cloudpickle.load(file_pi)
history = train_hist_dict['history']
params = train_hist_dict['params']

# if needed load pretrained model
if not train_model:
    model = tf.keras.models.load_model('models/' + model_name)

# Check its architecture
model.summary()

# let's evaluate the model
test_ds = tf.data.Dataset.from_generator(ds_utils.FrameGenerator(X_test, params['frames_per_video'], accede_df),
                                         output_signature=output_signature)

# provo a mettere un unico batch gigante per vedere se riesco a testare su tutto dopo
# TODO
test_ds = test_ds.batch(int(params['val_test_size'] / 2))

# Print the shapes of the data
test_frames, test_labels = ds_utils.print_dataset_shapes(test_ds, "test")

# evaluate model
results = model.evaluate(test_frames, test_labels, batch_size=1)  # che numero metto?

print("test loss, test metrics:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`


print("Generate predictions")
predictions = model.predict(test_frames)

if verbose:
    n_of_examples = 5
    print("val_predicted\tval_ground_truth\taro_predicted\taro_ground_truth")
    for i in range(0, n_of_examples):
        print("{:.6f}\t\t".format(predictions[i, 0]), "{:.6f}\t\t\t".format(test_labels[i, 0]),  # valence
              "{:.6f}\t\t".format(predictions[i, 1]), "{:.6f}\t".format(test_labels[i, 1]))  # arousal

    # check that files are sorted correctly after processing
    for i in range(0, int(params['val_test_size'] / 2)):
        if not ("{:.5f}".format(ds_utils.get_valence(X_test[i][-15:], accede_df)) == "{:.5f}".format(
                test_labels[i, 0])):
            print("Warning: ground truth/test subject mismatch!")
            print("valence file " + str(i) + " before processing: " + "{:.5f}".format(
                ds_utils.get_valence(X_test[i][-15:], accede_df)) + ", after processing: " + "{:.5f}".format(
                test_labels[i, 0]))
            break

print("plot train history")
plots.plot_history(model_name, history, results)

feat_names = ['valence', 'arousal']
n_of_predictions = min(100, int(params['val_test_size'] / 2))

print("plot predictions for first", n_of_predictions, "samples")
plots.plot_predictions(model_name, feat_names, test_labels, predictions, n_of_predictions)

print("plot stats")
plots.plot_stats(model_name, history, results, params)

print("Completed")
