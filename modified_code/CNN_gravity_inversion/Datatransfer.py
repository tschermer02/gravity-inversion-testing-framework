from __future__ import absolute_import, division, print_function, unicode_literals

# NumPy is used for array-shape inspection and numerical scaling.
import numpy as np


def anomal_tran_pos(anomal, model_max):
    """
    Scale a gravity anomaly using the maximum density-model value.

    The transformation is:

        transformed anomaly = anomaly / model_max

    This may be used when the density contrast is positive.
    """

    # Divide every anomaly value by the maximum model value.
    return (1 / model_max) * anomal


def anomal_tran_neg(anomal, model_min):
    """
    Scale a gravity anomaly using the minimum density-model value.

    The transformation is:

        transformed anomaly = -anomaly / model_min

    If model_min is negative, the factor -1/model_min becomes positive.
    This may be used for negative-density-contrast models.
    """

    # Scale the anomaly using the negative reciprocal of model_min.
    return (-1 / model_min) * anomal


def anomal_tran_deltz(anomal, dz):
    """
    Scale the gravity anomaly by the inverse vertical grid spacing.

    The transformation is:

        transformed anomaly = anomaly / dz

    Parameters
    ----------
    anomal : numpy.ndarray
        Gravity-anomaly array.
    dz : float
        Vertical grid spacing.

    Returns
    -------
    result : numpy.ndarray
        Gravity anomaly scaled by 1/dz.
    """

    # Print the anomaly shape before scaling.
    print("shape:\n", np.shape(anomal))

    # Divide every anomaly value by the vertical grid spacing.
    result = (1 / dz) * anomal

    # Print the shape after scaling.
    # The shape should remain unchanged.
    print("shape:\n", np.shape(result))

    return result