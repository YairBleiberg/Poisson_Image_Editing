import numpy as np
import scipy
import cv2
import scipy.sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


def PoissonSeamlessCloning(source, target, mask, offset, mixing):
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
    :param mixing: A flag indicating whether the guiding field is the gradient field of the source image or given by a
                    mixture of gradients from S and T.
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
        D = scipy.sparse.lil_matrix((m, m))
        D.setdiag(4)
        D.setdiag(-1, -1)
        D.setdiag(-1, 1)
        A = scipy.sparse.block_diag([D] * n).tolil()
        A.setdiag(-1, -m)
        A.setdiag(-1, m)

        flattened_cropped_mask = np.ravel(cropped_mask, order='F')
        flattened_cropped_target = np.ravel(cropped_target, order='F')

        # Removing the constraints on the pixels not to be copied.
        zero_coordinates = np.nonzero(flattened_cropped_mask == 0)[0]

        x = np.arange(-(n / 2), n / 2)
        y = np.arange(-(m / 2), m / 2)
        xv, yv = np.meshgrid(x, y)
        r = np.sqrt(xv ** 2 + yv ** 2)
        vxs, vys = np.gradient(cropped_source)

        vxt, vyt = np.gradient(cropped_target)
        v_abs_s = np.square(vxs) + np.square(vys)
        v_abs_t = np.square(vxt) + np.square(vyt)

        vx = vxs
        vy = vys
        if mixing:
            vx[v_abs_s<=v_abs_t] = vxt[v_abs_s<=v_abs_t]
            vy[v_abs_s<=v_abs_t] = vyt[v_abs_s<=v_abs_t]

        divergence_v = np.gradient(vx, axis=0) + np.gradient(vy, axis=1)
        # Extensions:
        # 1:
        # divergence_v = 0.5 * (np.gradient(vxs, axis=0) + np.gradient(vys, axis=1))
        # 2:
        # divergence_v = 2 * (np.gradient(vx, axis=0) + np.gradient(vy, axis=1))

        # 3:
        # divergence_v_1 = (np.gradient(vx, axis=0) + np.gradient(vy, axis=1))
        # divergence_v_2 = 0.5*(np.gradient(vx, axis=0) + np.gradient(vy, axis=1))
        # divergence_v = np.sin((yv-200)/20)*divergence_v_1 + (1 - np.sin((yv-200)/20))*divergence_v_2
        b = np.ravel(-divergence_v, order='F')
        A[zero_coordinates, np.clip(zero_coordinates + 1, 0, m * n - 1)] = 0
        A[zero_coordinates, np.clip(zero_coordinates - 1, 0, m * n - 1)] = 0
        A[zero_coordinates, np.clip(zero_coordinates + m, 0, m * n - 1)] = 0
        A[zero_coordinates, np.clip(zero_coordinates - m, 0, m * n - 1)] = 0
        A[zero_coordinates, zero_coordinates] = 1
        b[zero_coordinates] = flattened_cropped_target[zero_coordinates]

        A = scipy.sparse.lil_matrix.tocsc(A)
        x = spsolve(A, b)
        x = np.reshape(x, (m, n), order='F')
        target_channel[target_crop_row_range[0]:target_crop_row_range[1],
        target_crop_column_range[0]:target_crop_column_range[1]] = x

        # Combining the three channels back together.
        result[:, :, i] = target_channel

    return result