# -*- coding: utf-8 -*-

# Francesco Zumerle
# Code used for



# Code wrote starting from the Tensorflow tutorial https://www.tensorflow.org/tutorials/video/video_classification

# For tensorboard on local machine
# on local machine
# ssh -N -f -L localhost:6006:localhost:6006 username@x.x.x.x
# on remote machine
# tensorboard --logdir <path> --port 6006

import dataset_utils as ds_utils
import models as implemented_models
import plots
import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cloudpickle
from os.path import isfile, join
import time
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

train_model = False # perform train with the current params and settings
model_version = 'experiment' # additional string for filename
dataset_folder = '/nas/home/fzumerle/LIRIS-ACCEDE/data'

# All the following flags operate after loading the desired model
model_to_load = '3D_CNN_pat100_lr1e-05' # by default last checkpoint is loaded
model_history_checkpoint = 301 # manually set depending on early stopping

plot_test_results = False # compute metrics on test set
test_continuous = False # test model's performance on continuous liris-accede
test_again = False # test model's performance on AGAIN dataset (only Arousal predictions)
test_gameplay = False # test model's performance on any provided gameplay video


verbose = 1

models_available = {
    0: "efficientNet", # dummy model for testing
    1: "3D_CNN", # chosen model

    # other experiments, some showed lower performance,
    # others had other issues and are not guaranteed to work.
    2: "custom_resnet_v2",
    3: "ResNet50",
    4: "ResNet50+GRU",
    5: "ResNet50+TransformerEncoder",
    6: "Resnet50+DenseLayers"
}

# parameters must be adapted according to current model
params = dict(
    current_model=models_available[1],
    features=['valence', 'arousal'],    # choose from columns inside dataframe
    feature_range=[0, 5],
    feat_norm_range=[-1, 1],
    frame_height=224,
    frame_width=224,
    frames_per_video=6,
    frame_step=4,
    num_features=1024,
    global_annotations=False, # TODO
    n_epochs=1000,
    batch_size=32,
    patience=100,
    learning_rate=1e-5,
    dataset_size=8000, # full dataset: 9800 videos. 8000 videos * 6 frames --> 30Gb RAM
    balance_dataset=False, # not used, since we decided to maintain dataset values distributions (not uniform)
    cache=True,
    dataset_feat_extraction = False # used for models receiving features instead of frames
)

# train val test split policy: 80-10-10
params['train_size'] = int(params['dataset_size'] * 0.8)
params['val_test_size'] = int(params['dataset_size'] * 0.2)
# optimizer parameters
params['optimizer'] = tf.keras.optimizers.Adam(
    learning_rate=params['learning_rate'], beta_1=0.9, beta_2=0.999, amsgrad=False  # default False
)

# output signature is set according to current parameters
if params['dataset_feat_extraction']:
    params['output_signature'] = (
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=len(params["features"]), dtype=tf.float32)
    )
else:
    if params['frames_per_video'] == 1:
        params['output_signature'] = (
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=len(params["features"]), dtype=tf.float32)
        )
    else:
        params['output_signature'] = (
            tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=len(params["features"]), dtype=tf.float32)
        )

# classes for balancing train val test sets
n_classes = 10
n_bins = n_classes + 1
df_feat_columns = {
    2: "valence_rank",
    3: "arousal_rank",
    4: "valence",
    5: "arousal"
}
def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val][0]

feat_indexes = [get_keys_from_value(df_feat_columns, f) for f in params['features']]

# get model filename:
if train_model:
    if params['current_model'] == models_available[0]:
        model_name = params['current_model'] + model_version
    else:
        model_name = params['current_model'] + model_version + '_pat' + str(params['patience']) + '_lr' + str(
            params['learning_rate']) + ('_downsampled' if params['balance_dataset'] else '')
else:
    model_name = model_to_load

print('Model name: ', model_name)
time.sleep(1)

print('Training will begin with following parameters:\n')
for i in params:
    print('{}:'.format(i), params[i])

# load video list of training dataset
files = [f for f in os.listdir(dataset_folder) if
         isfile(join(dataset_folder, f))]
files.sort()


accede_df = pd.read_csv(r'/nas/home/fzumerle/LIRIS-ACCEDE/annotations/ACCEDEranking.txt', sep='\t')


# auxiliary plots used for debugging
if verbose and params['feat_norm_range'] is not None:
    print("plotting distribution before normalization")
    plots.plot_labels_dist(accede_df, 'original', df_feat_columns, feat_indexes, n_classes=n_classes)

# normalize va labels
if params['feat_norm_range'] is not None:
    accede_df, ds_min_value, ds_max_value = ds_utils.normalize_columns(
        accede_df,
        feat_dict=df_feat_columns,
        custom_min=params['feat_norm_range'][0],
        custom_max=params['feat_norm_range'][1],
        feat_indexes=feat_indexes,
        verbose=1
    )

# auxiliary plots used for debugging
if verbose and params['feat_norm_range'] is not None:
    print("plotting distribution after normalization")
    plots.plot_labels_dist(accede_df, 'normalized', df_feat_columns, feat_indexes, n_classes=n_classes)

if verbose:
    print('check that functions are built correctly:')
    file_to_check = 7
    print('row 0 of annotation file:\n')
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)
    print(accede_df.loc[accede_df['name'] == files[file_to_check]])
    print('features from file:', files[file_to_check])
    for i, feat in enumerate(feat_indexes):
        print(params['features'][i], ' extracted: ', ds_utils.get_label(files[file_to_check], accede_df, feat))

# generate classes for balancing dataset
accede_df_classes, new_class_columns = ds_utils.get_va_classes(
    accede_df,
    feat_dict=df_feat_columns,
    min_value=params['feat_norm_range'][0] if params['feat_norm_range'] is not None else params['feature_range'][0],
    max_value=params['feat_norm_range'][1] if params['feat_norm_range'] is not None else params['feature_range'][1],
    feat_indexes=feat_indexes,
    n_bins=n_bins,
    verbose=verbose
)

# downsample to balance valence and arousal (not used)
accede_df_downsampled = accede_df_classes
if params['balance_dataset']:
    for i in range(0, len(new_class_columns)):
        accede_df_downsampled = ds_utils.downsample_to_balance_col(accede_df_downsampled, column_name=new_class_columns[i],
                                                               verbose=verbose)
accede_df_downsampled = accede_df_downsampled.sort_values(by=['id'])

if params['dataset_size'] > accede_df_downsampled.shape[0]:
    print("Warning: reducing dataset size to downsampled dataset size: ", accede_df_downsampled.shape[0])
    params['dataset_size'] = accede_df_downsampled.shape[0]
    params['train_size'] = int(params['dataset_size'] * 0.8)
    params['val_test_size'] = int(params['dataset_size'] * 0.2)


if verbose and params['balance_dataset']:
    print("plotting distribution after downsampling")
    plots.plot_labels_dist(accede_df_downsampled, 'downsampled', df_feat_columns, feat_indexes, n_classes=n_classes)

# get list of videos
video_list = accede_df_downsampled['name']
video_list = [dataset_folder + '/' +  name for name in video_list]

# get list of classes [last class column generated], for sets balancing
classes = accede_df_downsampled[new_class_columns[-1]].values.tolist()

# split dataset in train and test
X_train, X_val, y_train, y_val = train_test_split(video_list, classes,
                                                    train_size=params['train_size'], test_size=params['val_test_size'],
                                                    random_state=50, stratify=classes)

if verbose:
    print("plotting distribution of y_train, after train test splitting")
    plots.plot_labels_dist(y_train, 'y_train', df_feat_columns, feat_indexes, n_classes=n_classes, is_df=False)

# split test in validation and test
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, stratify=y_val, random_state=50)

# check total amount of videos imported
video_total = len(X_train) + len(X_val)  + len(X_test)
print(f"Total videos: {video_total}")
if train_model:
    implemented_models.run_training(X_train, X_val, y_train, y_val, accede_df, params, models_available,
                                    feat_indexes, model_name)
else:
    print('The current model will just be loaded and evaluated')



''' Test section '''

# load history dictionary
history_file_path = 'trainHistoryDicts/' + model_name

if os.path.isfile(history_file_path):
    with open(history_file_path, 'rb') as file_pi:
        train_hist_dict = cloudpickle.load(file_pi)
    history = train_hist_dict['history']
    params = train_hist_dict['params']

else:
    history = None

# load pretrained model
model = tf.keras.models.load_model('models/' + model_name + '_checkpoint')

# Check its architecture
model.summary()

# plot history
if history is not None:
    print("plot train history")
    plots.plot_history(model_name, history  )
else:
    print('WARNING: no history dict was provided ...')


# TEST SET RESULTS
if plot_test_results:
    test_ds = tf.data.Dataset.from_generator(
        ds_utils.FrameGenerator(
            X_test,
            params,
            accede_df,
            feat_indexes=feat_indexes,
            continuous_test=False
        ),
        output_signature=params['output_signature']
    )

    test_ds = test_ds.batch(len(X_test))

    test_frames, test_labels = ds_utils.get_next_frames_and_labels(test_ds, "test")

    # evaluate model
    results = model.evaluate(test_frames, test_labels, batch_size=1)
    print("Test set metrics: ", results)

    print("Generate predictions")
    predictions = model.predict(test_frames)

    # plot VALENCE stats
    mse = mean_squared_error(test_labels[:, 0], predictions[:, 0])
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(test_labels[:, 0], predictions[:, 0])
    print("VALENCE only metrics: ", mse, rmse, mae)
    # plot AROUSAL stats
    mse = mean_squared_error(test_labels[:, 1], predictions[:, 1])
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(test_labels[:, 1], predictions[:, 1])
    print("AROUSAL only metrics: ", mse, rmse, mae)
    if history is not None:
        print("plot FINAL stats")
        plots.plot_stats("FINAL stats", history, model_history_checkpoint, results, params)



# Test with files from liris-accede-continuous

if test_continuous:
    print("Testing model on continuous liris-accede")

    files = [f[:-4] for f in os.listdir('/nas/home/fzumerle/continuous-LIRIS-ACCEDE/continuous-movies') if
             isfile(join('/nas/home/fzumerle/continuous-LIRIS-ACCEDE/continuous-movies', f))]
    files.sort()

    print("number of files for test: ", len(files))

    n_of_predictions_dataset = 300
    max_test_files_dataset = 30
    mov_avg_window_dataset = 10

    for file_number, file in enumerate(files[0:max_test_files_dataset]):

        print(max_test_files_dataset - file_number, "files remaining ... ")

        #if file != 'Islands':
        #    continue

        cont_arousal_accede_df = pd.read_csv(r'/nas/home/fzumerle/continuous-LIRIS-ACCEDE/continuous-annotations/' + file + '_Arousal.txt', sep='\t')
        cont_valence_accede_df = pd.read_csv(r'/nas/home/fzumerle/continuous-LIRIS-ACCEDE/continuous-annotations/' + file + '_Valence.txt', sep='\t')
        cont_valence_accede_df['arousal'] = cont_arousal_accede_df['Mean']
        cont_valence_accede_df['std_arousal'] = cont_arousal_accede_df['Std']
        continuous_accede_df = cont_valence_accede_df.rename(columns={"Mean": "valence", "Std": "std_valence"})

        # normalize in new range [-1, 1], from current range [0, 1]

        curr_n_of_predictions_dataset = min(continuous_accede_df.shape[0] - 1, n_of_predictions_dataset)

        cont_feat_dict = {
            1: "valence",
            2: "std_valence",
            3: "arousal",
            4: "std_arousal",
        }

        print("testing file: ", file)

        feat_indexes = [get_keys_from_value(cont_feat_dict, f) for f in params['features']]

        if params['feat_norm_range'] != [-1, 1]:

            print('convert from current range [-1, 1] (check paper) to range determined in training: {}'.format(
                params['feat_norm_range']))

            for i, df_index in enumerate (feat_indexes):

                if verbose:
                    print("min and max ", params['features'][i], " values before normalization:",
                          continuous_accede_df.iloc[:, df_index].min(),
                          continuous_accede_df.iloc[:, df_index].max()
                          )

                continuous_accede_df, ds_min_value, ds_max_value = ds_utils.linear_conversion(
                    continuous_accede_df,
                    df_column=df_index,
                    old_min=-1,
                    old_max=1,
                    custom_min=params['feat_norm_range'][0],
                    custom_max=params['feat_norm_range'][1]
                )
                if verbose:
                    print("min and max ", params['features'][i], " values after normalization:", ds_min_value,
                          ds_max_value)

        # generate frames, in this case for the whole length of each file (~10s at 30fps, )
        test_ds = tf.data.Dataset.from_generator(
            ds_utils.FrameGenerator(
                '/nas/home/fzumerle/continuous-LIRIS-ACCEDE/continuous-movies/' + file + '.mp4',
                params,
                continuous_accede_df,
                feat_indexes=feat_indexes,
                continuous_test=True
            ),
            output_signature=params['output_signature']
        )

        test_ds = test_ds.batch(curr_n_of_predictions_dataset)

        test_frames, test_labels = ds_utils.get_next_frames_and_labels(test_ds, "test")

        # evaluate model
        # TODO: sapendo che inizio a estrarre frame a partire da ogni secondo, calcola lunghezza ciascuna predizione
        # TODO  e shifta in modo da centrare ciascuna finestra col il corrispondente ground truth
        results = model.evaluate(test_frames, test_labels, batch_size=1)

        print("test loss, test metrics:", results)

        # Generate predictions on new data using `predict`
        print("Generate predictions")
        predictions = model.predict(test_frames)

        if file_number == 0:
            #all_test_frames = test_frames
            all_test_labels = test_labels
            all_test_predictions = predictions
            all_pcorr_val = [pearsonr(test_labels[:,0], predictions[:,0])[0]]
            all_pcorr_ar = [pearsonr(test_labels[:,1], predictions[:,1])[0]]
        else:
            #all_test_frames = tf.convert_to_tensor(np.concatenate((all_test_frames.numpy(), test_frames.numpy())))
            all_test_labels = tf.convert_to_tensor(np.concatenate((all_test_labels.numpy(), test_labels.numpy())))
            all_test_predictions = np.concatenate((all_test_predictions, predictions))
            all_pcorr_val.append(pearsonr(test_labels[:,0], predictions[:,0])[0])
            all_pcorr_ar.append(pearsonr(test_labels[:,1], predictions[:,1])[0])

        if verbose:
            n_of_examples = 5
            print("val_predicted\tval_ground_truth\taro_predicted\taro_ground_truth")
            for i in range(0, n_of_examples):
                print("{:.6f}\t\t".format(predictions[i, 0]), "{:.6f}\t\t\t".format(test_labels[i, 0]),  # valence
                      "{:.6f}\t\t".format(predictions[i, 1]), "{:.6f}\t".format(test_labels[i, 1]))  # arousal

            # check that files are sorted correctly after processing, checking first feature
            for i in range(0, curr_n_of_predictions_dataset):
                if not ("{:.5f}".format(ds_utils.get_continuous_label(i, continuous_accede_df, feat_indexes[0])) == "{:.5f}".format(
                        test_labels[i, 0])):
                    print("Warning: ground truth/test subject mismatch!")
                    print(params["features"][0], "file " + str(i) + " before processing: " + "{:.5f}".format(
                        ds_utils.get_continuous_label(i, continuous_accede_df, feat_indexes[0])) + ", after processing: " +
                          "{:.5f}".format(test_labels[i, 0]))
                    break

        print("plot predictions for first", curr_n_of_predictions_dataset, "samples")
        plots.plot_predictions(
            model_name + "-" + file,
            params['features'],
            test_labels,
            predictions,
            curr_n_of_predictions_dataset,
            results=True,
            mov_avg_window=mov_avg_window_dataset,
            y_lim=[-1, 5] if params['feat_norm_range'] is None else [-1, 1],
        )

    # plot VALENCE stats
    mse = mean_squared_error(all_test_labels[:,0], all_test_predictions[:,0])
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(all_test_labels[:,0], all_test_predictions[:,0])
    pcorr = np.mean(all_pcorr_val)
    print("VALENCE metrics: ", mse, rmse, mae, pcorr)
    # plot AROUSAL stats
    mse = mean_squared_error(all_test_labels[:,1], all_test_predictions[:,1])
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(all_test_labels[:,1], all_test_predictions[:,1])
    pcorr = np.mean(all_pcorr_ar)
    print("AROUSAL metrics: ", mse, rmse, mae, pcorr)
    # plot all stats
    mse = mean_squared_error(all_test_labels, all_test_predictions)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(all_test_labels, all_test_predictions)
    pcorr = np.mean(all_pcorr_val + all_pcorr_ar)

    results = [mse, mse, rmse, mae, pcorr]
    print("FINAL test loss, test metrics:", results)


# Test with files from AGAIN dataset (only arousal annotations available)

if test_again:
    print("Testing model on AGAIN dataset")
    video_folder = '/nas/home/fzumerle/AGAIN/videos'
    files = [f[:-4] for f in os.listdir(video_folder) if
             isfile(join(video_folder, f))]

    max_test_files_again = 10
    n_of_predictions_again = 300
    mov_avg_window_again = 3

    all_test_frames = None

    cont_arousal_df = pd.read_csv(r'/nas/home/fzumerle/AGAIN/clean_data.csv')

    for file_number, file in enumerate(files[0:max_test_files_again]):

        print(max_test_files_again - file_number, "files remaining ... ")

        player_id, game, session_id = file.split('_')

        # I assume player_id + session_id uniquely identify videos
        tmp_df = cont_arousal_df[cont_arousal_df['[control]player_id'] == player_id]
        tmp_df = tmp_df[tmp_df['[control]session_id'] == session_id]
        curr_df = pd.DataFrame(columns=['Time', 'valence', 'arousal'])
        curr_df['Time'] = tmp_df['[control]time_stamp']
        curr_df['valence'] = np.zeros(curr_df.shape[0])
        curr_df['arousal'] = tmp_df['[output]arousal']

        # now I have complete series of timestamps and arousal values, but I want to have just one entry per second
        curr_df['Time'] = curr_df['Time'].astype(int)
        curr_df = curr_df.drop_duplicates(subset=['Time'], keep='last')

        n_of_predictions_again = min(curr_df.shape[0], n_of_predictions_again)

        cont_feat_dict = {
            1: "valence",
            2: "arousal",
        }
        feat_indexes = [get_keys_from_value(cont_feat_dict, f) for f in params['features']]

        if verbose:
            print("min and max arousal values before normalization:",
                  curr_df.iloc[:, feat_indexes[1]].min(),
                  curr_df.iloc[:, feat_indexes[1]].max()
                  )

        curr_df, ds_min_value, ds_max_value = ds_utils.linear_conversion(
            curr_df,
            df_column=feat_indexes[1],
            old_min=0,
            old_max=1,
            custom_min=params['feat_norm_range'][0],
            custom_max=params['feat_norm_range'][1]
        )

        if verbose:
            print("min and max arousal values after normalization:", ds_min_value,
                  ds_max_value)

        print("testing file: ", file)

        # generate frames, in this case for the whole length of each file (~10s at 30fps, )
        test_ds = tf.data.Dataset.from_generator(
            ds_utils.FrameGenerator(
                '/nas/home/fzumerle/AGAIN/videos/' + file + '.mp4',
                params,
                curr_df,
                feat_indexes=feat_indexes,
                continuous_test=True
            ),
            output_signature=params['output_signature']
        )

        test_ds = test_ds.batch(n_of_predictions_again)
        test_frames, test_labels = ds_utils.get_next_frames_and_labels(test_ds, "test")

        # evaluate model
        # only arousal predicted, values not to be considered
        results = model.evaluate(test_frames, test_labels, batch_size=1)

        print("test loss, test metrics:", results)

        # Generate predictions on new data using `predict`

        print("Generate predictions")
        predictions = model.predict(test_frames)

        filename = file.split('_')[1]

        print("plot predictions for first", n_of_predictions_again, "samples")
        plots.plot_predictions(
            'AGAIN/' + filename,
            ['arousal'],
            test_labels,
            predictions,
            n_of_predictions_again,
            results=False,
            mov_avg_window=mov_avg_window_again,
            y_lim=[-1, 5] if params['feat_norm_range'] is None else [-1, 1],
        )


# Test on any input gameplay provided

if test_gameplay:
    print ("testing gameplay videos with current model...\n")
    print("TODO: n_of_prediction must be smaller than video length, otherwise infinite loop!\n")
    video_folder = '/nas/home/fzumerle/sample_videogame'
    files = [f[:-4] for f in os.listdir(video_folder) if
             isfile(join(video_folder, f))]
    # files.sort()

    max_test_files_game = 10
    n_of_predictions_game = 118
    mov_avg_window_game = 3

    for file_number, file in enumerate(files[0:max_test_files_game]):

        print(max_test_files_game - file_number, "files remaining ... ")

        print("current file: ", file)
        # generate frames
        test_ds = tf.data.Dataset.from_generator(
            ds_utils.FrameGenerator(
                video_folder + '/' + file + '.mp4',
                params,
                accede_df=n_of_predictions_game,
                feat_indexes=feat_indexes,
                continuous_test=True,
            ),
            output_signature=params['output_signature']
        )

        test_ds = test_ds.batch(n_of_predictions_game)

        test_frames, test_labels = ds_utils.get_next_frames_and_labels(test_ds, "test")

        print("Generate predictions")
        predictions = model.predict(test_frames)

        print("plot predictions for first", n_of_predictions_game, "samples")
        plots.plot_predictions(
            model_name + "-" + file,
            params['features'],
            test_labels,
            predictions,
            n_of_predictions_game,
            mov_avg_window=mov_avg_window_game,
            y_lim=[-1, 5] if params['feat_norm_range'] is None else params['feat_norm_range']
        )
        print('generate real time video prediction plot')
        plots.real_time_plot_predictions(
            model_name,
            file,
            predictions,
            n_of_predictions_game,
            y_lim=[0, 5] if params['feat_norm_range'] is None else params['feat_norm_range']
        )

        print("remove old overlay video, if exists")
        os.system("rm -f video_plots/" + model_name + "_" + file + "_overlay.mp4")

        print('generate final overlay video')
        # produce output video overlayed
        script = "ffmpeg -i ../sample_videogame/" + file + ".mp4 -i video_plots/" + model_name + "_" + file + \
                 ".mp4 -filter_complex \"[1:v]scale=400:-1[v2];[0:v][v2]overlay=main_w-overlay_w-5:5\" "\
                 "-c:v libx264 -c:a aac -crf 28 -t " + str(n_of_predictions_game) + " video_plots/" + model_name + "_" + file + \
                 "_overlay.mp4"
        print(script)
        os.system(script)
        # remove temporary file
        print("remove real time plot temporary file")
        os.system("rm video_plots/" + model_name + "_" + file + ".mp4")

        # reference script
        # ffmpeg -i input.mov -i overlay.mov -filter_complex "[0:0][1:0]overlay[out]" -shortest -map [out] -map 0:1 -pix_fmt yuv420p -c:a copy -c:v libx264 -crf 18  output.mov

print("Completed")