import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

def plot_sample(X, y, preds, binary_preds, ix=None):
    new="output.png"
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0])
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Biopsy')

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('Mask')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), levels=[0.5])
    ax[2].set_title('Mask Predicted')
    
    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Mask Predicted binary');
    fig.savefig("results/"+new,bbox_inches="tight",pad_inches=0)
    print("The results of Figure plotted is succesfuly saved in results folder")
