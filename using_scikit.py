# %%
import numpy as np
from skimage import feature, measure
import matplotlib.pyplot as plt

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
data = np.loadtxt('/Users/mahmoudabdelrazek/work/personal/red_sea_domes/Red Sea depth.asc', skiprows=6)
data_16bit = data.astype(np.int16)

# %%
# data_16bit = np.multiply(data_16bit, -1)
data_16bit = filter_islands(data_16bit)
data_16bit = filter_edges(data_16bit)

# %%
blobs = feature.blob_log(data_16bit, min_sigma=0, max_sigma=15, threshold=0.01)
print(len(blobs))

# %%
# Initialize lists to store circularities and non-circular blobs
circularities = []
non_circular_blobs = []
circular_blobs = []
first_countour_list = []

def perimeter_calc(contour):
    assert contour.ndim == 2 and contour.shape[1] == 2, contour.shape
 
    shift_contour = np.roll(contour, 1, axis=0)
    dists = np.sqrt(np.power((contour - shift_contour), 2).sum(axis=1))
    return dists.sum()



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


# Display the detected and non-circular blobs (for visualization)
fig, ax = plt.subplots()
ax.imshow(data_16bit, cmap='gray')

for blob in non_circular_blobs:
    y, x, r = blob
    c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
    ax.add_patch(c)

# for contour in first_countour_list:
#     ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='green')

# for blob in circular_blobs:
#     y, x, r = blob
#     c = plt.Circle((x, y), r, color='blue', linewidth=2, fill=False)
#     ax.add_patch(c)

plt.show()

# Print circularities of all blobs
print("Circularities:", circularities)



#alternative method

blobs_filtered = blobs[blobs[:,2] < 2, :]
fig, ax = plt.subplots(1,1)
pcm = ax.pcolormesh(data_16bit)
for blob in blobs_filtered:
    y, x, r = blob
    c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
    ax.add_patch(c)