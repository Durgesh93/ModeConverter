import cv2
import numpy as np

class Cone:
    def __init__(self, input_data):
        """
        Initializes the Cone class with an image or movie data.

        Parameters:
        input_data (np.ndarray): A 2D or 3D NumPy array representing the input data.
                                 - 2D array: Single image (HXW)
                                 - 3D array: Movie (NXHXW), where N is the number of frames.

        Raises:
        ValueError: If input_data is neither a 2D nor a 3D NumPy array or if the movie has no frames.
        """
        if len(input_data.shape) == 2:
            # Single image case (HXW)
            self.process_image(input_data)
        elif len(input_data.shape) == 3:
            # Movie case (NXHXW)
            if input_data.shape[0] == 0:
                raise ValueError("The movie does not contain any frames.")
            self.process_image(input_data[0])
        else:
            raise ValueError("Input must be either a 2D or 3D NumPy array.")
    
    def process_image(self, image):
        """
        Processes a single image to find extreme points (top, left, and right).

        Parameters:
        image (np.ndarray): A 2D NumPy array representing a grayscale image.

        The method performs the following steps:
        - Applies Gaussian blur to the image.
        - Applies binary thresholding to create a binary image.
        - Uses morphological operations (dilation and erosion) to refine the binary image.
        - Detects edges using Canny edge detection.
        - Finds contours and determines the top, left, and right extreme points.
        """
        # Apply Gaussian blur
        kernel_size = (11, 11)
        blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
        
        # Apply binary thresholding
        _, binary = cv2.threshold(blurred_image, 1, 255, cv2.THRESH_BINARY)
        
        # Define structuring element for morphological operations
        kernel = np.ones((51, 51), np.uint8)
        
        # Apply dilation
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # Apply erosion
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        # Detect edges using Canny edge detection
        edges = cv2.Canny(eroded, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize variables to store the extreme points
        self.top_point = None
        self.left_point = None
        self.right_point = None
        
        # Process contours to find top, left, and right points
        for contour in contours:
            contour_points = contour.reshape(-1, 2)
            
            # Find the point with minimum y-coordinate (top)
            min_y_point = contour_points[np.argmin(contour_points[:, 1])]
            
            # Find the point with minimum x-coordinate (left)
            min_x_point = contour_points[np.argmin(contour_points[:, 0])]
            
            # Find the point with maximum x-coordinate (right)
            max_x_point = contour_points[np.argmax(contour_points[:, 0])]
            
            # Update extreme points if necessary
            if self.top_point is None or min_y_point[1] < self.top_point[1]:
                self.top_point = min_y_point
            
            if self.left_point is None or min_x_point[0] < self.left_point[0]:
                self.left_point = min_x_point
            
            if self.right_point is None or max_x_point[0] > self.right_point[0]:
                self.right_point = max_x_point

        # Convert points to NumPy arrays in (y, x) format
        self.top_point = np.array(self.top_point[::-1]) if self.top_point is not None else None
        self.left_point = np.array(self.left_point[::-1]) if self.left_point is not None else None
        self.right_point = np.array(self.right_point[::-1]) if self.right_point is not None else None

    def rotate_point(self, point, center, angle):
        """
        Rotates a point around a center by a given angle in degrees.

        Parameters:
        point (np.ndarray): The point to rotate (y, x).
        center (np.ndarray): The center of rotation (y, x).
        angle (float): The angle to rotate in degrees.

        Returns:
        np.ndarray: The rotated point (y, x).
        """
        angle_rad = np.deg2rad(-angle)
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad),  np.cos(angle_rad)]])
        rotated_point = np.dot(rotation_matrix, point - center) + center
        return rotated_point
    
    def left(self, angle=0):
        """
        Provides the coordinates of the top and left points after rotating the left point clockwise.

        Parameters:
        angle (float): The angle by which to rotate the left point clockwise (in degrees). Default is 0.

        Returns:
        np.ndarray: A 2D array containing the top and rotated left points (y, x).

        Raises:
        ValueError: If the top and left points have not been determined.
        """
        if self.top_point is not None and self.left_point is not None:
            # Rotate the left_point clockwise
            rotated_left_point = self.rotate_point(self.left_point, self.top_point, -angle)
            return np.vstack((self.top_point, rotated_left_point))
        else:
            raise ValueError("Top and Left points have not been determined.")

    def right(self, angle=0):
        """
        Provides the coordinates of the top and right points after rotating the right point counterclockwise.

        Parameters:
        angle (float): The angle by which to rotate the right point counterclockwise (in degrees). Default is 0.

        Returns:
        np.ndarray: A 2D array containing the top and rotated right points (y, x).

        Raises:
        ValueError: If the top and right points have not been determined.
        """
        if self.top_point is not None and self.right_point is not None:
            # Rotate the right_point counterclockwise
            rotated_right_point = self.rotate_point(self.right_point, self.top_point, angle)
            return np.vstack((self.top_point, rotated_right_point))
        else:
            raise ValueError("Top and Right points have not been determined.")
