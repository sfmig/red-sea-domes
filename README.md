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
The 