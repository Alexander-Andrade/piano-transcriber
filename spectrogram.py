import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    features = np.load('datasets/features_balakirew_islamei.npy')
    labels = np.load('datasets/labels_balakirew_islamei.npy')

    f, axes = plt.subplots(1, 2)

    axes[0].imshow(features[:3000])
    axes[1].imshow(labels[:3000])

    plt.show()
