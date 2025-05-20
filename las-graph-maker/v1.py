import imageio.v3 as iio
import numpy as np
import os


# Load image names and save them in a list to use
image_store = []
def load_images(path):
    image_store.clear()
    counter = 0
    skip_count = 0
    for filename in os.listdir(path):
        if counter > 100:
            print(skip_count)
            break
        try:
            O = iio.imread(uri = path + '/' + filename)
            image_store.append(O)
            counter += 1
            print(f'Counter: {counter}')
        except Exception as e:
            skip_count += 1
            print(f"Error loading image {filename}: {e}")
        else:
            skip_count += 1
            print(f"Skipping empty file: {filename}")

load_images("/home/tikki/Documents/School/BachelorProject/Digitizing-overlapping-curves/tif-files")
#print(len(image_store))

# Sum along columns
def horizontal_projection(line_pixels):
    if len(line_pixels.shape) > 2: # Color to grayscale
        img = np.mean(line_pixels, axis=2).astype(np.uint8)
    else:
        img = line_pixels

    binary = img < 128
    return np.sum(binary, axis=1)

# Sum along rows
def vertical_projection(line_pixels):
    return np.sum(line_pixels, axis=0)

# Image sweep using horizontal_projection
def multi_horizontal_projection(image_store):
    sweep_result = []
    for image in image_store:
        val = horizontal_projection(image)
        sweep_result.append(val)
    return sweep_result

t = multi_horizontal_projection(image_store)

def find_whitespace(projection):
    threshold = 0.05 * np.max(projection)
    mask = projection < threshold
    return mask

def multi_whitespace_mask(img_sweeps):
    mask_per_img = {}
    for idx, img in enumerate(img_sweeps, start=1):
        mask = find_whitespace(img)
        mask_per_img[f"image {idx}"] = mask
    return mask_per_img

w = multi_whitespace_mask(t)

def find_header_and_graph(mask):
    sections = []
    is_section = False
    section_start = 0

    for i, active in enumerate(mask):
        if active and not is_section:
            # Section start
            section_start = i
            is_section = True
        elif not active and is_section:
            # Section end
            sections.append((section_start, i-1))
            is_section = False
    if is_section:
        sections.append((section_start, len(mask)-1))
    return sections

# TODO: find header and graphs on alle images, store the graphs sections
# TODO: Apply vertical projection on the graph sections
# TODO: Make a function that only finds graphs so like find_header_and_graphs but vertically if possible
# TODO: Extract and save individual graphs
