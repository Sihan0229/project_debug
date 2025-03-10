#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import keras
import cv2


try:
    import scipy
    # scipy.ndimage cannot be accessed until explicitly imported
    from scipy import ndimage
except ImportError:
    scipy = None


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_affine_transform(x, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,
                           row_axis=0, col_axis=1, channel_axis=2,
                           fill_mode='nearest', cval=0., order=1):
    """Applies an affine transformation specified by the parameters given.

    # Arguments
        x: 2D numpy array, single image.
        theta: Rotation angle in degrees.
        tx: Width shift.
        ty: Heigh shift.
        shear: Shear angle in degrees.
        zx: Zoom in x direction.
        zy: Zoom in y direction
        row_axis: Index of axis for rows in the input image.
        col_axis: Index of axis for columns in the input image.
        channel_axis: Index of axis for channels in the input image.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        order: int, order of interpolation

    # Returns
        The transformed version of the input.
    """
    if scipy is None:
        raise ImportError('Image transformations require SciPy. '
                          'Install SciPy.')
    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        shear = np.deg2rad(shear)
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)
        x = np.rollaxis(x, channel_axis, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [ndimage.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=order,
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def image_process(x_img,
                  horizontal_flip=False,
                  vertical_flip=False,
                  rotation_range=0,
                  shear_range=0.,
                  seed=None,
                  ):
    if seed is not None:
        np.random.seed(seed)

    flip_horizontal = (np.random.random() < 0.5) * horizontal_flip
    flip_vertical = (np.random.random() < 0.5) * vertical_flip

    if rotation_range:
        theta = np.random.uniform(
            -rotation_range,
            rotation_range)
    else:
        theta = 0
    if shear_range:
        shear = np.random.uniform(
            -shear_range,
            shear_range)
    else:
        shear = 0
    x_img = apply_affine_transform(x=x_img, theta=theta, shear=shear)

    data_format = keras.backend.image_data_format()
    channel_axis = 3
    row_axis = 1
    col_axis = 2
    if data_format == 'channels_first':
        channel_axis = 1
        row_axis = 2
        col_axis = 3
    if data_format == 'channels_last':
        channel_axis = 3
        row_axis = 1
        col_axis = 2
    img_row_axis = row_axis - 1
    img_col_axis = col_axis - 1
    if flip_horizontal:
        x_img = flip_axis(x_img, img_col_axis)

    if flip_vertical:
        x_img = flip_axis(x_img, img_row_axis)

    return x_img


if __name__ == "__main__":
    # from PIL import Image

    image_path = r'../DataBase_mura/pic/0/00004-image1.png'
    IN_SIZE = (256, 256)

    # img = Image.open(image_path)
    img = cv2.imread(image_path)
    # img.show()
    img = cv2.resize(img,(IN_SIZE[0], IN_SIZE[1]),interpolation=cv2.INTER_LINEAR)
    # img = img.resize((IN_SIZE[0], IN_SIZE[1]), interpolation=cv2.INTER_LINEAR)
    img = np.array(img)
    img = image_process(x_img=img,
                        horizontal_flip=True,
                        vertical_flip=True,
                        rotation_range=10,
                        shear_range=0.)


    # pic = Image.fromarray(img)
    # pic.show()
