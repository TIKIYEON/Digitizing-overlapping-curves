import imageio.v3 as imageio
import numpy as np
# Open an image file
image_path = "C:/Users/willi/OneDrive/Skrivebord/Bachelor/Github/Digitizing-overlapping-curves/Multipleintersections/T00105.las-1.tif"  # Replace with your actual image path
image = imageio.imread(image_path)



# Load the image


# Rotate the image 90 degrees counterclockwise
rotated_image = np.rot90(image)

# Save the rotated image as a TIFF file
imageio.imwrite("rotatedroot_image.tif", rotated_image)

print("Image rotated and saved successfully!")