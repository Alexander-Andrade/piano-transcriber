import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    features = np.load('datasets/features_alb_esp4.npy')
    labels = np.load('datasets/labels_alb_esp4.npy')

    start = 0
    end = 512

    # f, axes = plt.subplots(1, 2)
    # print(features.shape)
    # # axes[0].imshow(features[-800:-200])
    # # axes[1].imshow(labels[-800:-200])
    #
    # axes[0].imshow(features[start:end])
    # axes[1].imshow(labels[start:end])

    grid = plt.GridSpec(2, 6, bottom=0.04, top=0.98, left=0.02, right=0.98)

    axes = plt.subplot(grid[:, 0])
    axes.imshow(features[start:end])

    axes = plt.subplot(grid[:, 1])
    axes.imshow(labels[start:end])

    plt.show()
