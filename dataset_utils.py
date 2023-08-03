# -*- coding: utf-8 -*-

import tensorflow as tf
import cv2
import numpy as np
import random
from pathlib import Path
import pandas as pd

import models_utils


# define functions to get annotations from dataset
def get_label(filename, accede_df, column):
    """
    Get label from selected column for filename

    Args:
        filename: file from LIRIS-ACCEDE dataset
        accede_df: dataframe generated from clean_data.csv, from LIRIS-ACCEDE dataset
        column: column, corresponding to feature/stat

    Return:
        float value
    """
    row = accede_df.loc[accede_df['name'] == filename]
    return row.iloc[0, column]


def get_continuous_label(second, accede_df, column):
    """
    Get label from selected column for filename

    Args:
        second: Time of valence/arousal annotation
        accede_df: dataframe generated from clean_data.csv, from LIRIS-ACCEDE dataset
        column: column, corresponding to feature/stat

    Return:
        float value
    """
    row = accede_df.loc[accede_df['Time'] == second]
    return row.iloc[0, column]


def linear_conversion(df, df_column, old_min, old_max, custom_min, custom_max):
    old_range = (old_max - old_min)
    if old_range == 0:
        df.iloc[:, df_column] = custom_min
    else:
        custom_range = (custom_max - custom_min)
        df.iloc[:, df_column] = (((df.iloc[:,
                                     df_column] - old_min) * custom_range) / old_range) + custom_min
    curr_min_value = df.iloc[:, df_column].min()
    curr_max_value = df.iloc[:, df_column].max()
    return df, curr_min_value, curr_max_value


def normalize_columns(accede_df, feat_dict, custom_min=None, custom_max=None, feat_indexes=[4, 5], verbose=0):
    """
    Normalize valence and arousal annotations in 0-1 range

    Args:
        accede_df: input dataframe
        feat_dict: dict containing indexes inside df
        feat_indexes: columns to normalize, default are valence and arousal
        custom_min: used for custom range
        custom_max: used for custom range
        verbose: if 1, prints min and max values before and after normalization

    Return:
      dataframe with normalized columns
    """
    curr_min_value = None
    curr_max_value = None

    # normalization formula: (x - min) / (max - min)
    for feat in feat_indexes:  # columns
        if verbose:
            print("min and max ", feat_dict[feat], " values before normalization:",
                  accede_df.iloc[:, feat].min(), accede_df.iloc[:, feat].max())

        accede_df.iloc[:, feat] = (accede_df.iloc[:, feat] - accede_df.iloc[:, feat].min()) / (
                accede_df.iloc[:, feat].max() - accede_df.iloc[:, feat].min())
        # linear conversion: NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
        if custom_min is not None and custom_max is not None and (custom_min != 0 or custom_max != 1):
            accede_df, curr_min_value, curr_max_value = linear_conversion(accede_df, feat, 0, 1, custom_min, custom_max)
        if verbose:
            print("min and max ", feat_dict[feat], " values after normalization:", curr_min_value, curr_max_value)

    if curr_max_value is None or curr_min_value is None:
        raise Exception("no min or max values were determined, check function usage")
    return accede_df, curr_min_value, curr_max_value


def get_va_classes(accede_df, feat_dict, min_value, max_value, feat_indexes, n_bins=11, verbose=0):
    """
    returns df with additional class columns for valence and arousal
    classes are assigned based on feature value
    Args:
        accede_df: original df
        feat_dict: feat index dictionary
        min_value: min value of normalized features
        max_value: min value of normalized features
        feat_indexes: features used in model training
        n_bins: desired number of classes +1 to be assigned
        verbose: print additional info
    Returns:
        accede_df: df with classes assigned
    """
    bins = np.linspace(min_value, max_value, num=n_bins)
    bins = np.delete(bins, 0, 0)
    new_columns = []
    if verbose:
        print("assign ", len(bins), "classes to videos, one for each interval in this list: ", bins)
    # for each feature (column in dataframe)
    for _, (feat_index, feat_name) in enumerate(feat_dict.items()):
        if feat_index not in feat_indexes:
            continue
        new_columns.append("class_" + feat_name)
        accede_df[new_columns[-1]] = None
        # for each video (row in dataframe)
        for video in range(0, len(accede_df.iloc[:, feat_index])):
            # assign class to current video
            curr_value = accede_df.iloc[video, feat_index]
            for candidate_class in bins:
                if curr_value <= candidate_class:
                    accede_df.loc[video, new_columns[-1]] = str("{:.1f}".format(candidate_class))
                    break
                if candidate_class == bins[-1]:
                    print("WARNING: this video was not labelled: ", video)
                    print("current feature value", curr_value, "current candidate class", candidate_class)
    return accede_df, new_columns


def downsample_to_balance_col(accede_df_classes, column_name, verbose=0):
    """
    Downsample df to balance class distribution

    Args:
        accede_df_classes: input dataframe with classes
        column_name: column containing classes obtained by one of the dataset's features (valence or arousal)
        verbose: if 1, prints additional checks

    Return:
        downsampled dataframe
    """
    print("downsampling to balance ", column_name)
    val_class_dist = accede_df_classes[column_name].value_counts()
    val_classes = list(val_class_dist.index)
    val_classes.sort()
    max_sample_per_class = val_class_dist[val_class_dist.idxmin()]
    for idx, val_class in enumerate(val_classes):
        # create dataframe with current class
        tmp = accede_df_classes.loc[accede_df_classes[column_name] == val_class]
        # downsample current dataframe
        tmp = tmp.sample(max_sample_per_class, random_state=60)
        # add it to final dataframe
        if idx == 0:
            accede_df_downsampled = tmp
        else:
            accede_df_downsampled = pd.concat([accede_df_downsampled, tmp])
    if verbose:
        print("expected number of rows (max_sample_per_class * n_classes)", max_sample_per_class * len(val_classes))
        print("final number of rows: ", accede_df_downsampled.shape[0])

    return accede_df_downsampled


def format_frames(frame, output_size):
    """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded.
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
    """
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

def prepare_single_video(frames, n_frames, num_features, img_size):
    frame_features = np.zeros(shape=(1, n_frames, num_features), dtype="float32")
    feature_extractor = models_utils.build_feature_extractor(img_size)

    # Pad shorter videos.
    if len(frames) < n_frames:
        diff = n_frames - len(frames)
        padding = np.zeros((diff, img_size, img_size, 3))
        frames = np.concatenate(frames, padding)

    frames = frames[None, ...]

    # Extract features from the frames of the current video.
    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(n_frames, video_length)
        for j in range(length):
            if np.mean(batch[j, :]) > 0.0:
                frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
            else:
                frame_features[i, j, :] = 0.0

    return frame_features

def frames_or_features_from_video_file(video_path, n_frames, frame_step, output_size=(224, 224), second=None,
                                       global_annotations=False, dataset_feat_extraction=False, num_features=None):
    """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.
      second: choose specific second from video
      frame_step: distance between frames extracted.
      global_annotations: change behavior according to dataset annotations (global or continuous)
      dataset_feat_extraction: change mode between frame extraction and feat extraction
      num_features: dimension of features
    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
    """
    # Read each video frame by frame
    result = []
    src = cv2.VideoCapture(str(video_path)) # open video

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT) # get frame count
    frame_rate = src.get(cv2.CAP_PROP_FPS)  # get frame rate

    if global_annotations:
        frame_step = int(np.floor(video_length * frame_rate / n_frames))
        need_length = video_length
    else:
        need_length = 1 + (n_frames - 1) * frame_step


    # ret is a boolean indicating whether read was successful, frame is the image itself
    ret = False
    while not ret:
        if second is not None:
            start = second * frame_rate
        elif need_length > video_length:
            start = 0
        else:
            # in case of single frame, randomize position each time
            if n_frames == 1:
                max_start = video_length - 1
            else:
                max_start = video_length - need_length
            start = random.randint(0, max_start + 1)    # choose random starting point

        src.set(cv2.CAP_PROP_POS_FRAMES, start)
        ret, frame = src.read()
    # normalize color
    #frame = cv2.normalize(frame, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    frame = format_frames(frame, output_size)
    result.append(frame)

    for _ in range(n_frames - 1):   # for each frame
        for _ in range(frame_step): # cycle through frames until frame step
            ret, frame = src.read()
        if ret:
            #normalize color
            #frame = cv2.normalize(frame, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))

    src.release()
    result = np.array(result)[..., [2, 1, 0]]

    # reduce dimensionality in case of single frame
    if result.shape[0] == 1:
        result = result[0, :, :, :]

    # Extract features from the frames of the current video.

    if dataset_feat_extraction:
        '''
        feature_extractor = models_utils.build_feature_extractor(output_size[0])
        frames_features = np.zeros(
            shape=(n_frames, num_features), dtype="float32"
        )

        for i, batch in enumerate(result):
            length = min(n_frames, video_length)
            frames_features[i, :] = feature_extractor.predict(
                result[i:i+1, :]# che senso ha
            )
        '''
        frames_features = prepare_single_video(result, n_frames, num_features, output_size[0])
        # for now returning features of single video
        return frames_features[0, ...]
    else:
        return result


class FrameGenerator:
    def __init__(self, path, params, accede_df, feat_indexes, training=False, continuous_test=False):
        """ Returns a set of frames with their associated label.

            Args:
                path: Video file paths.
                n_frames: Number of frames.
                training: Boolean to determine if training dataset is being created.
        """
        self.path = path
        self.n_frames = params['frames_per_video']
        self.frame_height = params['frame_height']
        self.frame_width = params['frame_width']
        self.training = training
        self.accede_df = accede_df
        self.feat_indexes = feat_indexes
        self.global_annotations = params['global_annotations']
        # self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        # self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))
        self.continuous_test = continuous_test
        self.frame_step = params['frame_step']
        self.dataset_feat_extraction = params['dataset_feat_extraction']
        self.num_features = params['num_features']

    def get_files_and_class_names(self):
        # video_paths = list(self.path.glob('*/*.mp4'))
        if self.continuous_test:
            if isinstance(self.accede_df, int): # I know, not a good programming example
                self.path = [self.path] * self.accede_df
            else:
                self.path = [self.path] * self.accede_df.shape[0]
        video_paths = list(map(Path, self.path))
        # classes = [p.parent.name for p in video_paths]
        return video_paths  # , classes

    def __call__(self):
        if isinstance(self.accede_df, int) and not self.continuous_test:
            raise Exception("DEBUG: this is not supposed to happen")
        # video_paths, classes = self.get_files_and_class_names()
        video_paths = self.get_files_and_class_names()
        video_filenames = [str(f)[-15:] for f in video_paths]
        # print("example filename", video_filenames[0])
        tuples = list(zip(video_paths, video_filenames))

        if self.training:
            random.shuffle(tuples)

        if self.continuous_test:
            if isinstance(self.accede_df, int):
                seconds = self.accede_df
            else:
                seconds = self.accede_df.shape[0]

            for path, filename in tuples:  # just take the first path, which is the same
                break
            for second in range(seconds):
                # here I take each second n frames with provided frame step
                video_features = frames_or_features_from_video_file(
                    path,
                    self.n_frames,
                    frame_step=self.frame_step,
                    second=second,
                    output_size=(self.frame_width, self.frame_height),
                    global_annotations=self.global_annotations,
                    dataset_feat_extraction=self.dataset_feat_extraction,
                    num_features=self.num_features
                )
                # add feature values
                va_values = []
                for i in self.feat_indexes:
                    if isinstance(self.accede_df, int):
                        va_values.append(None)
                    else:
                        va_values.append(get_continuous_label(second, self.accede_df, i))
                # label = self.class_ids_for_name[classname] # Encode labels
                yield video_features, va_values
        else:
            for path, filename in tuples:
                video_features = frames_or_features_from_video_file(
                    path,
                    self.n_frames,
                    frame_step=self.frame_step,
                    output_size=(self.frame_width, self.frame_height),
                    global_annotations=self.global_annotations,
                    dataset_feat_extraction=self.dataset_feat_extraction,
                    num_features=self.num_features
                )
                # add feature values
                va_values = []
                for i in self.feat_indexes:
                    va_values.append(get_label(filename, self.accede_df, i))
                # label = self.class_ids_for_name[classname] # Encode labels
                yield video_features, va_values


def get_next_frames_and_labels(ds, name):
    """
    Prints shape of input features, returns frames anf labels

    Args:
        ds: data structure containing video frames
        name: name of current set (train, val, test)

    Returns:
        object: frames and labels
    """
    frames, labels = next(iter(ds))
    print(f'Shape of {name} set of frames: {frames.shape}')
    print(f'Shape of {name} labels: {labels.shape}')

    return frames, labels

