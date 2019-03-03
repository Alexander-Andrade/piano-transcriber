import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    features = np.load('datasets/features_balakirew_islamei.npy')
    labels = np.load('datasets/labels_balakirew_islamei.npy')

    start = -500
    end = -1

    f, axes = plt.subplots(1, 2)
    print(features.shape)
    # axes[0].imshow(features[-800:-200])
    # axes[1].imshow(labels[-800:-200])

    axes[0].imshow(features[start:end])
    axes[1].imshow(labels[start:end])

    plt.show()
