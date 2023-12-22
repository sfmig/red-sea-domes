# %%
import numpy as np
from skimage import feature, measure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mpld3
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Enable interactive plotting
%matplotlib ipympl 

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load data
placeholder_value = -32767
data = np.loadtxt('./Red Sea depth.asc', skiprows=6)
# data_16bit = data.astype(np.int16)
# shape (4193, 2662)

# visualise
# fig, ax = plt.subplots()
# ax.imshow(data)
# mpld3.display()

# plot histogram
# plt.figure()
# plt.hist(data.flatten())
# plt.axvline(
#     placeholder_value, 
#     color='r', 
#     linestyle='solid',
#     linewidth=1
# )

# replace placeholder value for nan and exclude anything above 0
threshold_value=-5000 # from histogram
data = np.where(((data > threshold_value) & (data < 0)), data, np.nan)

# visualise data after thresholding
fig, ax = plt.subplots()
im = ax.imshow(data)
cax = fig.add_axes([.75, .1, 0.05, 0.8])
fig.colorbar(im, cax=cax, orientation='vertical')
plt.show()

# %%%%%%%%%%%%%%%%%%%
# Zoom in a patch

zoom_in_patch_xy = [
    1550, 2500, 
]
zoom_in_patch_wh = [
    450, 450
]

fig, ax = plt.subplots()
ax.imshow(data)

rect = patches.Rectangle(
    zoom_in_patch_xy, 
    *zoom_in_patch_wh, 
    linewidth=1, 
    edgecolor='r', 
    facecolor='none'
)

ax.add_patch(rect)

plt.xlabel("x")
plt.ylabel("y")

# zoom in 
fig, ax = plt.subplots()
im=plt.imshow(data[
    zoom_in_patch_xy[1]:zoom_in_patch_xy[1]+zoom_in_patch_wh[1],
    zoom_in_patch_xy[0]:zoom_in_patch_xy[0]+zoom_in_patch_wh[0]
    ]
)
plt.xlabel("x")
plt.ylabel("y")
cax = fig.add_axes([.85, .1, 0.05, 0.8])
fig.colorbar(im, cax=cax, orientation='vertical')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Check watershed method for segmentation

from scipy import ndimage
np.random.seed(0)

# place markers in the lowest points
# select points of low height
n_markers = 100
markers_max_height = np.nanpercentile(data, 5) # p% of the values are lower
markers_bool = data < markers_max_height # True where candidate markers are
rc_markers = np.argwhere(markers_bool)  # row, column of candidate markers

# sample n of the candidate markers
# could also merge using centre_of_mass?
markers = np.zeros_like(data).astype(np.int16)
rc_markers_sample = rc_markers[np.random.randint(0,rc_markers.shape[0], n_markers),:] # sample rows
x_markers = rc_markers_sample[:,1] 
y_markers = rc_markers_sample[:,0]
i=0
for x, y in zip(x_markers, y_markers):
    markers[y,x] = i # not the best approach?
    i += 1

# visualise markers' locations
fig, ax = plt.subplots()
ax.imshow(data)
plt.scatter(
    x=x_markers,
    y=y_markers, 
    c='r', 
    s=30)
plt.show()

# run watershed algorithm
res1 = ndimage.watershed_ift(
    data.astype(np.uint8), 
    markers
)
plt.figure()
plt.imshow(res1) 

# plt.figure()
# plt.imshow(res1[500:2000,500:4000])

# %%%%%%%%%%%%%%%%%%%%%%%%
# Compute height iso-contours
data = np.loadtxt('./Red Sea depth.asc', skiprows=6)

# Find contours at a constant value (max+min/2 by default)
contours = measure.find_contours(data)

# plot all contours found
fig, ax = plt.subplots()
ax.imshow(data, cmap=plt.cm.gray)
for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
ax.set_xticks([])
ax.set_yticks([])
plt.show()
# %%%%%%%%%%%%%%%%%%%%%%%%
# Contour plot
data = np.loadtxt('./Red Sea depth.asc', skiprows=6)

# threshold data
threshold_value=-5000
data = np.where(((data > threshold_value) & (data < 0)), data, np.nan)

# plot contours
fig, ax = plt.subplots()
ax.axis('equal')
im=plt.contourf(data, origin="image")
cax = fig.add_axes([.9, .1, 0.05, 0.8])
fig.colorbar(im, cax=cax, orientation='vertical')
# for contour in contours: #--- sanity check
#     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Prepare for local maxima
# data 
data_raw = np.loadtxt('./Red Sea depth.asc', skiprows=6)
threshold_value=-5000
data = np.where(((data_raw > threshold_value) & (data_raw < 0)), data_raw, np.nan)

# define footprint based on outer contour
contours = measure.find_contours(data_raw)
fig, ax = plt.subplots()
ax.imshow(data)
for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)


# select a subset of points for the contour (otherwise spiky?)
contours_array = np.concatenate(contours, axis=0)
n_contour_pts = 100
idx_subset = np.random.randint(0, contours_array.shape[0], n_contour_pts)
contours_array = contours_array[idx_subset, :]

# order contour points clockwise
# https://pavcreations.com/clockwise-and-counterclockwise-sorting-of-coordinates/
angles = np.arctan2(
    contours_array[:,0]-contours_array.mean(axis=0)[0],
    contours_array[:,1]-contours_array.mean(axis=0)[1],
)  # angles relative to mean point
indices = np.argsort(angles) # sort indices by negative of the angle (- for clockwise)
contours_array_sorted = contours_array[indices,:]

# start from point of lowest x?
# idx_start = np.argmin(contours_array[:,0])
# contours_array[]


# check before sorting
fig, ax = plt.subplots()
ax.scatter(
    x=contours_array[:,1], 
    y=contours_array[:,0],
    c=list(range(contours_array.shape[0]))
)
ax.axis('equal')
ax.invert_yaxis()

# check after sorting
fig, ax = plt.subplots()
ax.scatter(
    x=contours_array_sorted[:,1], 
    y=contours_array_sorted[:,0],
    c=list(range(contours_array_sorted.shape[0]))
)
ax.axis('equal')
ax.invert_yaxis()

# %%%%%%%%%%%%%%%%%%%
# compute binary mask from outer contour

from scipy.spatial import ConvexHull

data_raw = np.loadtxt('./Red Sea depth.asc', skiprows=6)
threshold_value=-5000
data = np.where(((data_raw > threshold_value) & (data_raw < 0)), data_raw, np.nan)

# define footprint based on outer contour
contours = measure.find_contours(data_raw)
contours_array = np.concatenate(contours, axis=0)

fig, ax = plt.subplots()
ax.imshow(data)
for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)


hull = ConvexHull(contours_array)
idx_hull = hull.vertices

outer_contour_mask = measure.grid_points_in_poly(
    data.shape, 
    contours_array[idx_hull,:], 
    binarize=False
)

plt.figure()
plt.imshow(outer_contour_mask)


# %%
from skimage.morphology import square

data_raw = np.loadtxt('./Red Sea depth.asc', skiprows=6)
threshold_value=-5000 
# remove only placeholder value
data = np.where(((data_raw > threshold_value)), data_raw, np.nan)


kernel_half_size = 50 # px
coordinates = feature.peak_local_max(
    data_raw, # I need array with no nans - smooth / interpolate? np.where(np.isnan(data),1000, data), 
    min_distance=kernel_half_size,
    # exclude_border=True,
    # footprint=square(3),  # I think this is the kernel :( - can I use this to force for a shape?
    labels=outer_contour_mask.astype(int) #---> to use with discretised version?
)

# remove if close to contour?
# threshold peaks based on height? - remove peaks above zero
coordinates_filtered = np.asarray(
    [
        c
        for c in coordinates
        if data_raw[*c] < 0
    ]
)

coordinates_islands = np.asarray(
    [
        c
        for c in coordinates
        if data_raw[*c] >= 0
    ]
)

fig, ax = plt.subplots()
ax.imshow(data)
plt.scatter(
    x=coordinates[:,1],
    y=coordinates[:,0], 
    c='r', 
    s=30)
plt.show()

fig, ax = plt.subplots()
ax.imshow(data)
plt.scatter(
    x=coordinates_filtered[:,1],
    y=coordinates_filtered[:,0], 
    edgecolors='r', 
    marker='s',
    facecolors='none',
    s=kernel_half_size)
plt.show()

# %%
fig, ax = plt.subplots()
ax.imshow(data, cmap=plt.cm.gray)
plt.scatter(
    x=coordinates[:,1],
    y=coordinates[:,0], 
    c=[data_raw[*c] for c in coordinates], 
    s=10)
plt.colorbar()
plt.show()



# https://pavcreations.com/clockwise-and-counterclockwise-sorting-of-coordinates/
# def sort_coordinates(list_of_xy_coords):
#     cx, cy = list_of_xy_coords.mean(0)
#     x, y = list_of_xy_coords.T
#     angles = np.arctan2(x-cx, y-cy)
#     indices = np.argsort(-angles)
#     return list_of_xy_coords[indices]
# %%
# threshold_value=-5000
# data = np.where(((data > threshold_value) & (data < 0)), data, np.nan)

# # remove coordinates where data is nan?
# list_coordinates_new = list(coordinates)
# rc_nan = np.argwhere(np.isnan(data))
# for i, coord in enumerate(coordinates):
#     for x in rc_nan:
#         if all(coord == x):
#             list_coordinates_new.pop(i)  # seems overkill but cannot check set?

# %%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Find blobs?
