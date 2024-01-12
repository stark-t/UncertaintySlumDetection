# import packages
import os
import matplotlib.pyplot as plt
import numpy as np

# import code
import utils_config as config

def to_array(x, **kwargs):
    x = x.transpose(1, 2, 0)
    x = x * 255.0
    return x.astype(np.uint8)

def visualize(image, label, prediction=-1):
    """PLot images in one row."""
    plt.figure(figsize=(16, 5))
    plt.xticks([])
    plt.yticks([])
    label = list(label)
    if prediction != -1:
        plt.title('Label: ' + str(label.index(max(label))) + '\n' + 'Prediction:  ' + str(int(prediction)))
    else:
        plt.title('Label: ' + str(label.index(max(label))))

    # check image shape
    if isinstance(image[0,0,0], np.float32):
        if image.shape[-1] != 3:
            image = image.transpose(1, 2, 0)
        image = image[:,:,0]

    elif image.shape[-1] != 3 and isinstance(image[0,0,0], int):
        image = to_array(image)

    plt.imshow(image)
    plt.show()


