# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter
from sklearn.metrics import mean_squared_error
from math import sqrt
plt.rcParams['animation.ffmpeg_path'] = '/nas/home/fzumerle/miniconda3/envs/tfhack2/bin/ffmpeg'



def plot_history(model_name, history, y_lim=None):
    """Saves plot of train and validation history."""
    plt.figure(figsize=(12, 8), dpi=300)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.ylabel('loss', fontsize=22)
    plt.xlabel('epoch', fontsize=22)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    if y_lim is not None:
        plt.ylim(y_lim)
    else:
        plt.ylim([0.25, 0.6])
    plt.legend(['train', 'val'], loc='upper right', fontsize=22)
    plt.savefig('plots/' + model_name + '_loss.pdf', dpi = 300)
    plt.close('all')


def plot_stats(model_name, history, hist_checkpoint, results, params):
    """Saves plot of model and training details."""
    plt.figure(figsize=(9, 8), dpi=300)
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.title('model: ' + model_name + ' - details', fontsize=16)
    plt.text(0, 9, 'training parameters', fontsize=15, fontweight='bold')
    plt.text(8, 9, 'metrics', fontsize=15, fontweight='bold')

    # create strings for stats
    train_text = 'Train set\nmse: {:.6f}\nrmse: {:.6f}\nmae: {:.6f}'.format(
        history['mean_squared_error'][hist_checkpoint-1], history['root_mean_squared_error'][-1], history['mean_absolute_error'][-1])
    val_text = 'Validation set\nmse: {:.6f}\nrmse: {:.6f}\nmae: {:.6f}'.format(
        history['val_mean_squared_error'][hist_checkpoint-1], history['val_root_mean_squared_error'][-1],
        history['val_mean_absolute_error'][hist_checkpoint-1])
    test_text = 'Test set\nmse: {:.6f}\nrmse: {:.6f}\nmae: {:.6f}'.format(
        results[1], results[2], results[3])

    plt.text(
        10,
        1,
        train_text + '\n\n' + val_text + '\n\n' + test_text,
        fontsize=14,
        horizontalalignment='right',
    )
    params_string = ""
    for field in params:
        if field in ['current_model', 'optimizer', 'output_signature']:
            continue
        params_string += '{}:'.format(field) + ' ' + str(params[field]) + '\n'

    plt.text(
        0,
        1,
        params_string,
        fontsize=14,
        horizontalalignment='left',
    )

    plt.axis('off')
    plt.savefig('plots/' + model_name + '_stats.pdf', dpi = 300)
    plt.close('all')


def plot_labels_dist(accede_df, specific_name, cont_feat_dict, feat_indexes, n_classes, is_df=True):
    """Saves plot of class distribution in current dataset."""
    for _, (index, name) in enumerate(cont_feat_dict.items()):
        if feat_indexes is not None and index not in feat_indexes:
            continue
        if is_df:
            plt.hist(accede_df.iloc[:, index], bins=n_classes, range=[0, max(accede_df.iloc[:, index])])
        else:
            plt.hist(accede_df)
        if feat_indexes is None:
            plt.savefig('plots/' + specific_name + '_' + 'dist.pdf')
            break
        plt.savefig('plots/' + name + '_' + specific_name + '_' + 'dist.pdf', dpi = 300)
        plt.close('all')


def plot_predictions(model_name, feat_names, test_labels, predictions, n_of_predictions, results=False, mov_avg_window=None, y_lim=[-1, 1]):
    """Saves plot of predicted labels and ground truth labels."""
    feats = ['valence', 'arousal']
    for i in range(0, len(feats)):
        if feats[i] not in feat_names:
            continue

        plt.figure(figsize=(20, 11), dpi=300)
        # plot ground truth
        plt.plot(range(len(test_labels[0:n_of_predictions, i])), test_labels[0:n_of_predictions, i])

        # plot predictions
        if mov_avg_window is None:
            plt.plot(range(len(test_labels[0:n_of_predictions, i])), predictions[0:n_of_predictions, i], color = 'red')

        # plot moving average predictions
        else:
            x = range(len(test_labels[0:n_of_predictions, i]))
            y = predictions[0:n_of_predictions, i]
            average_y = []
            for ind in range(len(y) - mov_avg_window + 1):
                average_y.append(np.mean(y[ind:ind + mov_avg_window]))
            for ind in range(mov_avg_window - 1):
                average_y.insert(0, np.nan)
            plt.plot(x, average_y, color = 'red')

        # Pearson correlation:
        pred = pd.Series(predictions[:, i])
        pred_ma = pd.Series(average_y)
        gt = pd.Series(test_labels[:, i])
        if mov_avg_window is None:
            pearson_corr = pred.corr(gt)
            mse = mean_squared_error(pred, gt)
        else:
            pearson_corr = pred_ma.corr(gt)
            mse = mean_squared_error(pred_ma[mov_avg_window:len(pred_ma)], gt[mov_avg_window:len(pred_ma)])

        if results:
            test_text = 'Metrics\nmse: {:.6f}\nrmse: {:.6f}\nPearsCorr: {:.6f}'.format(
                mse, sqrt(mse), pearson_corr)
            plt.text(0, y_lim[0]*0.7, test_text , fontsize=12,
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

        #plt.title('model: ' + model_name + ' - ' + feat_names[i] + ' prediction of first ' + str(n_of_predictions) + ' seconds', fontsize=20)
        plt.xlabel('seconds', fontsize=45)
        plt.ylabel('arousal', fontsize=45)
        plt.ylim(y_lim)
        plt.grid()
        plt.xticks([0, 60, 120], fontsize=40)
        plt.yticks([-1, 0, 1], fontsize=40)
        plt.legend(['ground truth', 'predicted', 'predictions moving average'], loc='upper left', fontsize=45)
        plt.savefig('plots/' + model_name + '_' + feats[i] + '_test_predictions.pdf', dpi = 300)
        plt.close('all')


def real_time_plot_predictions(model_name, file, predictions, n_of_predictions, y_lim=[-1, 1]):

    fig = plt.figure(figsize=(10, 6))
    l, = plt.plot([], [])
    l2, = plt.plot([], [])
    plt.title(file + ' - prediction of first ' + str(n_of_predictions) + ' seconds', fontsize=10)

    plt.xlabel('seconds')
    plt.ylabel('values')
    plt.legend(['valence', 'arousal'], loc='upper left')

    plt.ylim(y_lim)
    plt.xlim(0, n_of_predictions)

    metadata = dict(title=file + 'predictions', artist='francesco_zumerle')
    writer = FFMpegWriter(fps=1, metadata=metadata)

    time = []
    valence = []
    arousal = []

    with writer.saving(fig, 'video_plots/' + model_name + '_' + file + '.mp4', dpi=300):
        for xval in range(n_of_predictions):
            plt.xlim(xval - 60, xval)
            time.append(xval)
            valence.append(predictions[xval, 0])
            arousal.append(predictions[xval, 1])
            l.set_data(time, valence)
            l2.set_data(time, arousal)
            writer.grab_frame()
    plt.close('all')
