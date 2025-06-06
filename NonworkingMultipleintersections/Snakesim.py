import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import skimage as ski
import cv2 as cv

testFile = "NonworkingMultipleintersections/rotated_image.tif"

img = cv.imread(testFile, cv.IMREAD_GRAYSCALE)
ima = img[:,2000:3500]
def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


# Morphological ACWE
image0 = ski.util.img_as_float(ima)

# Initial level set
init_ls = ski.segmentation.checkerboard_level_set(image0.shape, 6)
# List with intermediate results for plotting the evolution
evolution = []
callback = store_evolution_in(evolution)
ls = ski.segmentation.morphological_chan_vese(
    image0, num_iter=230, init_level_set=init_ls, smoothing=3, iter_callback=callback
)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(image0, cmap="gray")
ax[0].set_axis_off()
ax[0].contour(ls, [0.5], colors='r')
ax[0].set_title("Morphological ACWE segmentation", fontsize=12)

ax[1].imshow(ls, cmap="gray")
ax[1].set_axis_off()
ax[1].set_title("Morphological ACWE evolution", fontsize=12)

contour_labels = []
for n, color in ((2, 'g'), (7, 'y'), (35, 'r')):
    ax[1].contour(evolution[n], [0.5], colors=color)

    # Use empty line to represent this contour in the legend
    legend_line = mlines.Line2D([], [], color=color, label=f"Iteration {n}")
    contour_labels.append(legend_line)

ax[1].legend(handles=contour_labels, loc="upper right")

# Morphological GAC
image = ski.util.img_as_float(image0)
gimage = ski.segmentation.inverse_gaussian_gradient(image)

# Initial level set
init_ls = np.zeros(image.shape, dtype=np.int8)
init_ls[10:-10, 10:-10] = 1
# List with intermediate results for plotting the evolution
evolution = []
callback = store_evolution_in(evolution)
ls = ski.segmentation.morphological_geodesic_active_contour(
    gimage,
    num_iter=230,
    init_level_set=init_ls,
    smoothing=1,
    balloon=-1,
    threshold=0.69,
    iter_callback=callback,
)

ax[2].imshow(image, cmap="gray")
ax[2].set_axis_off()
ax[2].contour(ls, [0.5], colors='r')
ax[2].set_title("Morphological GAC segmentation", fontsize=12)

ax[3].imshow(ls, cmap="gray")
ax[3].set_axis_off()
ax[3].set_title("Morphological GAC evolution", fontsize=12)

contour_labels = []
for n, color in ((0, 'g'), (100, 'y'), (230, 'r')):
    ax[3].contour(evolution[n], [0.5], colors=color)

    # Use empty line to represent this contour in the legend
    legend_line = mlines.Line2D([], [], color=color, label=f"Iteration {n}")
    contour_labels.append(legend_line)

ax[3].legend(handles=contour_labels, loc="upper right")

fig.tight_layout()
plt.show()