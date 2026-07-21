import sys

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
import tensorflow as tf


def main() -> None:
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)

    print("\nPackage versions:")
    print("NumPy:", np.__version__)
    print("SciPy:", scipy.__version__)
    print("Pandas:", pd.__version__)
    print("Matplotlib:", matplotlib.__version__)
    print("h5py:", h5py.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("TensorFlow:", tf.__version__)

    print("\nTensorFlow devices:")
    for device in tf.config.list_physical_devices():
        print(device)

    density_model = np.zeros((16, 16, 16), dtype=np.float32)

    print("\nTest density model:")
    print("Shape:", density_model.shape)
    print("Data type:", density_model.dtype)

    print("\nEnvironment test completed successfully.")


if __name__ == "__main__":
    main()