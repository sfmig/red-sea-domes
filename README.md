# Red Sea Domes Extraction
## Background
Using the bathymetry data of the read sea floor we aim to extract dome protroding out of the floor.

## Data
Data is provided as a `asc` file. The following are quick stats taken from the header of the asc file.

| Attribute    | Value          |
|--------------|----------------|
| ncols        | 2662           |
| nrows        | 4193           |
| xllcorner    | 32.358333333362|
| yllcorner    | 12.495833333373|
| cellsize     | 0.004166666667 |
| NODATA_value | -32767         |

apart from the header the file can be read as a matrix of digital numbers. Each digital number represent a pixel (cell) of the data and corrosponds to a geographic location. 

Data can be read into a `numpy array` by discarding the header lines.

## Method
The method used to extract the domes is as follows:
1. Read the data into a numpy array
2. Apply a threshold to the data to remove all islands data
3. Replace `NODATA_value` with `numpy.nan`
4. Apply [Laplacian of Gaussian (LoG)](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html#laplacian-of-gaussian-log) to extract blobs (i.e. domes)
5. Apply [find contours](https://scikit-image.org/docs/stable/auto_examples/edges/plot_contours.html) method to find the outter perimeter of each blob.
6. Calculate the `circularity` of the first contour for each blob and store it in a list. The cirularity is calculated using the following equation `circularity = 4 * np.pi * area / (perimeter ** 2)`
7. Classify the blobs to circular and non-circular based on the `circularity` value. All blobs with `circularity` value greater than the threshold are classified as circular and the rest are classified as non-circular.