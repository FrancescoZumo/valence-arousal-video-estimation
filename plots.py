# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


def plot_history(model_name, history, results):
    # summarize training history
    plt.figure(figsize=(10, 8), dpi=200)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model: ' + model_name + ' - ' + 'loss', fontsize=20)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('plots/' + model_name + '_loss.png')


def plot_predictions(model_name, feat_names, test_labels, predictions, n_of_predictions):
    for i in range(0, len(feat_names)):
        plt.figure(figsize=(20, 10), dpi=200)
        # plt.scatter(test_labels[:, i], predictions[:, i])
        plt.scatter(range(len(test_labels[0:n_of_predictions, i])), test_labels[0:n_of_predictions, i])
        plt.scatter(range(len(test_labels[0:n_of_predictions, i])), predictions[0:n_of_predictions, i])
        # plot lines between pred and ground truth
        for j in range(0, n_of_predictions):
            plt.vlines(x=j, ymin=min(float(test_labels[j, i]), predictions[j, i]),
                       ymax=max(float(test_labels[j, i]), predictions[j, i]), color='#FF0F0F68')
        plt.title('model: ' + model_name + ' - ' + feat_names[i] + ' prediction of first ' + str(n_of_predictions) + ' test videos', fontsize=20)
        plt.xlabel('ground truth')
        plt.ylabel('predicted')
        plt.ylim([-0.2, 1.2])
        plt.grid()
        plt.legend(['ground truth', 'predicted'], loc='upper left')
        plt.savefig('plots/' + model_name + '_' + feat_names[i] + '_test_predictions.png')


def plot_stats(model_name, history, results, params):

    plt.figure(figsize=(7, 5.5), dpi=250)
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.title('model: ' + model_name + ' - details', fontsize=20)
    plt.text(0, 9, 'training parameters', fontsize=15, fontweight='bold')
    plt.text(8, 9, 'metrics', fontsize=15, fontweight='bold')

    # create strings for stats
    train_text = 'Train set\nmse: {:.6f}\nrmse: {:.6f}\nmae: {:.6f}'.format(
        history['mean_squared_error'][-1], history['root_mean_squared_error'][-1], history['mean_absolute_error'][-1])
    val_text = 'Validation set\nmse: {:.6f}\nrmse: {:.6f}\nmae: {:.6f}'.format(
        history['val_mean_squared_error'][-1], history['val_root_mean_squared_error'][-1],
        history['val_mean_absolute_error'][-1])
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
        if field in ['current_model', 'optimizer']:
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
    plt.savefig('plots/' + model_name + '_stats.png')
    plt.close('all')

