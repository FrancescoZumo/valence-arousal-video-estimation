# -*- coding: utf-8 -*-

import tensorflow as tf
import cv2
import numpy as np
import random
from pathlib import Path


# define functions to get annotations from dataset
def get_labels(filename, accede_df, column):
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


def get_valence(filename, accede_df):
    return get_labels(filename, accede_df, 4)


def get_arousal(filename, accede_df):
    return get_labels(filename, accede_df, 5)


def get_valence_va(filename, accede_df):
    return get_labels(filename, accede_df, 6)


def get_arousal_va(filename, accede_df):
    return get_labels(filename, accede_df, 7)


def normalize_columns(accede_df, columns=[4, 5], verbose=0):
    """
    Normalize valence and arousal annotations in 0-1 range

    Args:
        accede_df: input dataframe
        columns: columns to normalize, default are valence and arousal
        verbose: if 1, prints min and max values before and after normalization

    Return:
      dataframe with normalized columns
    """

    if verbose:
        print("min and max valence values:", accede_df.iloc[:, 4].min(), accede_df.iloc[:, 4].max())
        print("min and max arousal values:", accede_df.iloc[:, 5].min(), accede_df.iloc[:, 5].max())

    # normalization formula: (x - min) / (max - min)
    for column in columns:  # columns
        accede_df.iloc[:, column] = (accede_df.iloc[:, column] - accede_df.iloc[:, column].min()) / (
                accede_df.iloc[:, column].max() - accede_df.iloc[:, column].min())

    if verbose:
        print("min and max valence values after normalization:", accede_df.iloc[:, 4].min(), accede_df.iloc[:, 4].max())
        print("min and max arousal values after normalization:", accede_df.iloc[:, 5].min(), accede_df.iloc[:, 5].max())

    return accede_df


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


def frames_from_video_file(video_path, n_frames, output_size=(224, 224), frame_step=15):
    """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
    """
    # Read each video frame by frame
    result = []
    src = cv2.VideoCapture(str(video_path))

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    # ret is a boolean indicating whether read was successful, frame is the image itself
    ret, frame = src.read()
    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]

    return result


class FrameGenerator:
    def __init__(self, path, n_frames, accede_df, training=False):
        """ Returns a set of frames with their associated label.

            Args:
                path: Video file paths.
                n_frames: Number of frames.
                training: Boolean to determine if training dataset is being created.
        """
        self.path = path
        self.n_frames = n_frames
        self.training = training
        self.accede_df = accede_df
        # self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        # self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

    def get_files_and_class_names(self):
        # video_paths = list(self.path.glob('*/*.mp4'))
        video_paths = list(map(Path, self.path))
        # classes = [p.parent.name for p in video_paths]
        return video_paths  # , classes

    def __call__(self):
        # video_paths, classes = self.get_files_and_class_names()
        video_paths = self.get_files_and_class_names()
        video_filenames = [str(f)[-15:] for f in video_paths]
        # print("example filename", video_filenames[0])
        tuples = list(zip(video_paths, video_filenames))

        if self.training:
            random.shuffle(tuples)

        for path, filename in tuples:
            video_frames = frames_from_video_file(path, self.n_frames)
            # next I will add arousal
            va_values = [get_valence(filename, self.accede_df), get_arousal(filename, self.accede_df)]
            # here I must add va values

            # label = self.class_ids_for_name[classname] # Encode labels
            yield video_frames, va_values


def print_dataset_shapes(ds, name):
    """
    Prints shape of input features

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
