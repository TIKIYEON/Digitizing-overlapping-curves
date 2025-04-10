import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import invert
testFile = "C:/Users/willi/OneDrive/Skrivebord/Bachelor/Github/Digitizing-overlapping-curves/Profilelinetes/SimCurve8.tif"
# Load the image in grayscale
image = cv2.imread(testFile, cv2.IMREAD_GRAYSCALE)

# Invert the image (skeletonization works with white foregrounds)
inverted_image = invert(image)

_, binary_image = cv2.threshold(inverted_image, 127, 255, cv2.THRESH_BINARY)

skeleton = skeletonize(binary_image // 255)

# Convert skeleton to uint8 for saving
skeleton_uint8 = (skeleton * 255).astype(np.uint8)

# Save the skeletonized image
# Load the image in grayscale
skeleton = cv2.imread(testFile, cv2.IMREAD_GRAYSCALE)

# Binarize the skeleton image (ensure it's binary)
_, binary_skeleton = cv2.threshold(skeleton, 127, 255, cv2.THRESH_BINARY)
binary_skeleton = binary_skeleton // 255

# Identify intersection points
kernel = np.array([[1, 1, 1],
                   [1, 10, 1],
                   [1, 1, 1]], dtype=np.uint8)

convolved = cv2.filter2D(binary_skeleton.astype(np.uint8), -1, kernel)
intersections = (convolved > 10).astype(np.uint8)

# Keep only one intersection point
non_zero = np.transpose(np.nonzero(intersections))
if non_zero.size > 0:
    center = non_zero[len(non_zero) // 2]  # Select central point
    intersections[:] = 0
    intersections[center[0], center[1]] = 1

# Remove extra branches based on distance
dist_transform = cv2.distanceTransform(1 - binary_skeleton, cv2.DIST_L2, 5)
pruned_skeleton = binary_skeleton * (dist_transform < 50)  # Adjust distance threshold

# Recombine the single intersection point
pruned_skeleton[intersections == 1] = 1

# Save the final processed skeleton
final_skeleton = (pruned_skeleton * 255).astype(np.uint8)
cv2.imwrite('final_skeletonized_image.png', final_skeleton)

# Display result
cv2.imshow('Final Skeleton', final_skeleton)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the result
cv2.imshow('Skeletonized Image', skeleton_uint8)
cv2.waitKey(0)
cv2.destroyAllWindows()