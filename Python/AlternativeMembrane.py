import numpy as np
from scipy import signal
import cv2


def AlternativeMembraneSeamlessCloning(source, target, mask, offset, F):
    """
    Performs the Poisson seamless cloning as we have seen in class based on the article of Perez 2003.
    In this implementation we assume all the pixels on the frame of the mask are 0.
    :param source: A source image S some region of which we’d like to clone.
    :param target: A target image T, onto which we’d like to perform the seamless cloning.
    :param mask: A binary mask M (with the same height and width as S), where the non-zero values indicate those pixels
                of S that should be cloned onto T; You may assume a single connected non-zero region.
    :param offset: The offset (x,y) of the upper left corner of the mask M with respect to the upper left corner of T.
                    Note that the offsets could be negative. Note that T could be of different height/width than S and
                    M, but it may be assumed that T is large enough to contain the part of S that we’d like to
                    seamlessly clone onto it. In other words, the non-zero region of M fits inside T after the offset.

    :return: The result of seamless cloning of pixels from the source image into the target image as specified by the
    mask and the offset (x, y).
    """
    result = np.zeros(target.shape)

    # Splits the RGB channels of the given images.
    bSource, gSource, rSource = cv2.split(source)
    bTarget, gTarget, rTarget = cv2.split(target)
    source_channels = [bSource, gSource, rSource]
    target_channels = [bTarget, gTarget, rTarget]

    for i in range(3):
        target_channel = target_channels[i]
        source_channel = source_channels[i]

        # Crops the given mask, source image, and target image according to their overlap.
        target_crop_row_range = np.clip(np.array([offset[0], offset[0] + mask.shape[0]]), 0, target_channel.shape[0])
        target_crop_column_range = np.clip(np.array([offset[1], offset[1] + mask.shape[1]]), 0, target_channel.shape[1])
        cropped_target = target_channel[target_crop_row_range[0]:target_crop_row_range[1],
                         target_crop_column_range[0]:target_crop_column_range[1]]
        mask_crop_row_range = np.clip(np.array([-offset[0], target_channel.shape[0] - offset[0]]), 0, mask.shape[0])
        mask_crop_column_range = np.clip(np.array([-offset[1], target_channel.shape[1] - offset[1]]), 0, mask.shape[1])
        cropped_source = source_channel[mask_crop_row_range[0]:mask_crop_row_range[1],
                         mask_crop_column_range[0]:mask_crop_column_range[1]]
        cropped_mask = mask[mask_crop_row_range[0]:mask_crop_row_range[1],
                       mask_crop_column_range[0]:mask_crop_column_range[1]]

        # Constructs the linear equations system Av=b.
        m = mask_crop_row_range[1] - mask_crop_row_range[0]
        n = mask_crop_column_range[1] - mask_crop_column_range[0]

        # Removing the constraints on the pixels not to be copied.
        false_row_vector = np.zeros((1, n), dtype=bool)
        false_column_vector = np.zeros((m, 1), dtype=bool)

        a = cropped_mask == 0
        b = (np.vstack((cropped_mask[1:, :] == 1, false_row_vector)) |
             np.vstack((false_row_vector, cropped_mask[:-1, :] == 1)) |
             np.hstack((cropped_mask[:, 1:] == 1, false_column_vector)) |
             np.hstack((false_column_vector, cropped_mask[:, :-1] == 1)))

        boundary_pixels_characteristic = a & b
        boundary_pixels_extenstion = boundary_pixels_characteristic * (cropped_target - cropped_source)

        x = np.arange(-(n - 1), n)
        y = np.arange(-(m - 1), m)
        xv, yv = np.meshgrid(x, y)
        F_elementwise = np.vectorize(F)
        distance_map = np.sqrt(np.square(xv) + np.square(yv))
        # to solve the problem when F receives a zero argument, we change the only zero value in the distance map.
        # note that we assume that F is a function of the euclidean distance between pixels.
        distance_map[m - 1, n - 1] = 1
        kernel = F_elementwise(distance_map)

        weighted_mean = signal.fftconvolve(boundary_pixels_extenstion, kernel, mode='same')
        normalizer = signal.fftconvolve(boundary_pixels_characteristic, kernel, mode='same')
        membrane = weighted_mean / normalizer
        nonzero_coordinates = cropped_mask == 1
        cropped_target[nonzero_coordinates] = cropped_source[nonzero_coordinates] + membrane[nonzero_coordinates]


        target_channel[target_crop_row_range[0]:target_crop_row_range[1],
        target_crop_column_range[0]:target_crop_column_range[1]] = cropped_target

        # Combining the three channels back together.
        result[:, :, i] = target_channel

    return result