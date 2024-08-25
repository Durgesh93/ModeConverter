# Documentation

## `Converter` Class

The `Converter` class is designed to perform transformations and computations on video sequences and spatial coordinates.

### `__init__(self, anchor_idx, movies, SLcoords, W=64, SR=1, num_pts=None, A=0, S=0, R=0)`

Initializes the `Converter` class with the given parameters.

- **Parameters:**
- anchor_idx (int or list of int): location index of the anchor frame.
- movies (numpy.ndarray): BxNxHxW B movies each containing N frames or NxHXW or a single movie containing N frames.
- SLcoords (numpy.ndarray): The coordinates of SL(Scanline) points.
- W (int, optional): The window size for clipping (default is 64).
- SR (int, optional): The stride for shifting (default is 1).
- num_pts (int, optional): The number of points to sample along the SL (default is None).
- A (int or list of int, optional): Anchor shift within in the generated clip (default is 0).
- S (int or list of int, optional): Shift values for given scanline (default is 0).
- R (int or list of int, optional): Rotation values for given scanline (default is 0).

### `refSL(self)`

Generates reference SL (spatial line) points for the given coordinates.

- **Returns:**
  - `numpy.ndarray`: The reference SL points.

### `perSL(self)`

Generates perturbed SL (spatial line) points for the given coordinates.

- **Returns:**
  - `numpy.ndarray`: The perturbed SL points.

### `coeffs(self)`

Generates coefficients for the given coordinates.

- **Returns:**
  - `numpy.ndarray`: The coefficients for the given coordinates.

### `mcoords(self)`

- **Returns:**
  - `numpy.ndarray`: The AMM coordinates.


###  `mSLx(self)`
   - Gets the x-location  of the scanline in the generated AMM image from the clip.

### `mSL(self)`
  - `numpy.ndarray`: The SL coordinates in the AMM image.

### `pred_bcoords(self, pred_mcoords)`
- **Parameters:**
  - `pred_mcoords (numpy.ndarray)`: The manipulated coordinates to predict from.
- **Returns:**
  - `numpy.ndarray`: The Predicted coordinates in B-mode. 

### `amm(self)`
- **Returns:**
- numpy.ndarray: The AMM image created from the clip.

## `animation.MoviePlayer` Class

The `MoviePlayer` class provides functionality to play and visualize video frames using Matplotlib.

### `__init__(self, video_array, interval=30)`

Initializes the `MoviePlayer` with the video array and frame interval.

- **Parameters:**
  - `video_array` (numpy.ndarray): Array of video frames.
  - `interval` (int, optional): Interval between frames in milliseconds (default is 30).

### `register(self, hooks)`

Registers hooks to customize frame updates.

- **Parameters:**
  - `hooks` (function or list of functions): Functions to call during frame updates.

### `play(self)`

Plays the video using Matplotlib's `FuncAnimation`.

## `Cone` Class

The `Cone` class processes images or video frames to identify specific geometric features and apply transformations.

### `__init__(self, input_data)`

Initializes the `Cone` with single image or movie input.


### `left(self, angle=0)`

Finds the left segment of the cone and rotates it clockwise.

- **Parameters:**
  - `angle` (float, optional): Rotation angle in degrees (default is 0).

- **Returns:**
  - `numpy.ndarray`: Coordinates of the left segment.

### `right(self, angle=0)`

Finds the right segment of the cone and rotates it counterclockwise.

- **Parameters:**
  - `angle` (float, optional): Rotation angle in degrees (default is 0).

- **Returns:**
  - `numpy.ndarray`: Coordinates of the right segment.

## `movie.read_grayscale(file_path)`

Reads a grayscale image or video file and returns it as a NumPy array.

- **Parameters:**
  - `file_path` (str): Path to the image or video file.

- **Returns:**
  - `numpy.ndarray`: The grayscale image or video frames.