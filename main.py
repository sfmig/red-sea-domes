# %%
import numpy as np
from skimage import feature, measure
import matplotlib.pyplot as plt
from pathlib import Path

# %%
def detect_file_path():
    print("Detecting file path...")
    file_path = Path.cwd() / 'Red Sea depth.asc'
    return file_path

# %%
def read_data(data_path):
    print("Reading data...")
    data = np.loadtxt(data_path, skiprows=6)
    data_16bit = data.astype(np.int16)
    return data_16bit

# %%
def filter_islands(data):
    # check the count of negative value vs positive values
    # # if negative values are more then remove the positive values
    # # if positive values are more then remove the negative values

    # # count the number of negative values
    # neg_count = np.count_nonzero(data < 0)
    # pos_count = np.count_nonzero(data > 0)

    # if neg_count > pos_count:
        # remove positive values
    #     data = np.where(data < 0, data, np.nan)
    # else:
    #     # remove negative values
    #     data = np.where(data > 0, data, np.nan)
    data = np.where(data > 0, np.nan,  data )
    return data

def filter_edges(data):
    data = np.where(abs(data) < 32767, data, np.nan)
    return data

# %%
def filter_data(data_16bit):
    print("Filtering data...")
    # data_16bit = np.multiply(data_16bit, -1)
    data_16bit = filter_islands(data_16bit)
    data_16bit = filter_edges(data_16bit)
    return data_16bit
# %%
def extract_blobs(data_16bit):
    print("Extracting blobs...")
    blobs = feature.blob_log(data_16bit, min_sigma=0, max_sigma=15, threshold=0.01)
    #print(len(blobs))
    return blobs

# %%
def perimeter_calc(contour):
    assert contour.ndim == 2 and contour.shape[1] == 2, contour.shape
 
    shift_contour = np.roll(contour, 1, axis=0)
    dists = np.sqrt(np.power((contour - shift_contour), 2).sum(axis=1))
    return dists.sum()


# %%
def classify_blobs(blobs,data_16bit):
    print("Classifying blobs...")
    # Initialize lists to store circularities and non-circular blobs
    circularities = []
    non_circular_blobs = []
    circular_blobs = []
    first_countour_list = []
    blob_counter = 0
    for blob in blobs:
        y, x, r = blob
        # Find the contour of the blob
        mask = np.zeros_like(data_16bit, dtype=np.uint8)
        mask[int(y - r):int(y + r + 1), int(x - r):int(x + r + 1)] = 1
        contour = measure.find_contours(mask, 0.5)[0]
        first_countour_list.append(contour)

        # Calculate circularity
        # print(measure.moments(contour))
        area = measure.moments(contour)[0,0]  # Corrected line
        perimeter = perimeter_calc(contour)
        # perimeter = measure.perimeter(contour)
        circularity = 4 * np.pi * area / (perimeter ** 2)
        circularities.append(circularity)

        # Set a threshold for circularity to classify non-circular blobs
        circularity_threshold = 700
        if circularity < circularity_threshold:
            non_circular_blobs.append(blob)
        else:
            circular_blobs.append(blob)
        blob_counter += 1
        if blob_counter % 100 == 0:
            print("count of blobs processed: ",blob_counter)
    return circular_blobs, non_circular_blobs

def plot_blobs(data_16bit,non_circular_blobs,circular_blobs):
    print("Plotting blobs...")
    # Display the detected and non-circular blobs (for visualization)
    fig, ax = plt.subplots()
    ax.imshow(data_16bit, cmap='gray')

    for blob in non_circular_blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        ax.add_patch(c)

    for blob in circular_blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='blue', linewidth=2, fill=False)
        ax.add_patch(c)

    plt.show()


# main function
if __name__ == "__main__":
    data_path = detect_file_path()
    print("Detected data path: ", data_path)
    data = read_data(data_path)
    print("Data read successfully")
    filtered_data = filter_data(data)
    print("Data filtered successfully")
    blobs = extract_blobs(filtered_data)
    print("Blobs extracted successfully, count of blobs: ", len(blobs))
    circular_blobs, non_circular_blobs = classify_blobs(blobs, filtered_data)
    print("Blobs classified successfully")
    plot_blobs(filtered_data,non_circular_blobs,circular_blobs)