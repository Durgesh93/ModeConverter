import numpy as np
from scipy.interpolate import interp1d

class view_bounds:
    """
    Computes and clips coordinates or coefficients against a defined viewport.

    This class handles the viewport clipping algorithm, which is used to determine 
    which parts of a coordinate or polynomial-defined region are visible within a specified rectangular viewport. 

    Parameters:
    coords_or_coeffs (np.ndarray or list): Coordinates or polynomial coefficients for the region.
        - If `mode='coord'`, `coords_or_coeffs` should be an array of coordinates in the format (y, x).
        - If `mode='coeff'`, `coords_or_coeffs` should be an array of polynomial coefficients.
    mode (str, optional): The mode for interpreting `coords_or_coeffs`. Can be 'coord' for coordinates or 'coeff' for polynomial coefficients. Default is 'coord'.
    viewport (tuple, optional): The rectangular viewport defined as (x_min, x_max, y_min, y_max). Default is (0, 0, 255, 255).

    Attributes:
    viewport (np.ndarray): The viewport as a 2x2 array.
    x_min (float): Minimum x-coordinate of the viewport.
    x_max (float): Maximum x-coordinate of the viewport.
    y_min (float): Minimum y-coordinate of the viewport.
    y_max (float): Maximum y-coordinate of the viewport.
    INSIDE (int): Code indicating that a point is inside the viewport.
    LEFT (int): Code indicating that a point is to the left of the viewport.
    RIGHT (int): Code indicating that a point is to the right of the viewport.
    BOTTOM (int): Code indicating that a point is below the viewport.
    TOP (int): Code indicating that a point is above the viewport.
    coords (np.ndarray): The sorted coordinates or polynomial coefficients.
    bounds (np.ndarray): The bounding coordinates of the region.

    Methods:
    _compute_code(x, y):
        Computes the region code for a given point (x, y) relative to the viewport.
        
        Parameters:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        
        Returns:
        int: The region code for the point.

    _extend_bounds(w):
        Extends the bounds of the viewport by a specified width.
        
        Parameters:
        w (float): The width by which to extend the bounds.
        
        Returns:
        np.ndarray: The extended bounds.

    _clip(anchor):
        Clips a line segment defined by `anchor` against the viewport.
        
        Parameters:
        anchor (np.ndarray): A 2x2 array defining the line segment as [(y1, x1), (y2, x2)].
        
        Returns:
        tuple:
            - np.ndarray: The clipped line segment if it intersects the viewport, otherwise an array of -1.
            - bool: True if the line segment is within the viewport, False otherwise.

    get():
        Computes and returns the clipped viewport bounds.
        
        Returns:
        tuple:
            - np.ndarray: The clipped viewport bounds.
            - bool: True if the bounds are valid, False otherwise.
    """

    def __init__(self, coords_or_coeffs, mode='coord', viewport=(0, 0, 255, 255)):
        self.viewport = np.array(viewport, dtype='float').reshape(2, 2).T
        self.x_min = self.viewport[0, 0]
        self.x_max = self.viewport[0, 1]
        self.y_min = self.viewport[1, 0]
        self.y_max = self.viewport[1, 1]

        # Defining region codes
        self.INSIDE = 0   # 0000
        self.LEFT = 1     # 0001
        self.RIGHT = 2    # 0010
        self.BOTTOM = 4   # 0100
        self.TOP = 8      # 1000

        if mode == 'coord':
            coords = coords_or_coeffs
        else:
            coeffs = coords_or_coeffs
            xcoords = np.array([0., 1.])
            coords = np.stack([np.polyval(coeffs, xcoords), xcoords], axis=-1)

        self.coords = coords[coords[:, 0].argsort()]  # Sorting coordinates with increasing order of y
        self.bounds = coords[[0, -1]]  # Coordinates in form of yx s.t y2>y1 ((y1,x1),(y2,x2))

    def _compute_code(self, x, y):
        """
        Computes the region code for a point relative to the viewport.

        Parameters:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.

        Returns:
        int: The region code for the point.
        """
        code = self.INSIDE
        if x < self.x_min:      # To the left of the rectangle
            code |= self.LEFT
        elif x > self.x_max:    # To the right of the rectangle
            code |= self.RIGHT
        if y < self.y_min:      # Below the rectangle
            code |= self.BOTTOM
        elif y > self.y_max:    # Above the rectangle
            code |= self.TOP
        return code

    def _extend_bounds(self, w):
        """
        Extends the bounds of the viewport by a specified width.

        Parameters:
        w (float): The width by which to extend the bounds.

        Returns:
        np.ndarray: The extended bounds.
        """
        a1 = self.bounds[0]
        a2 = self.bounds[1]
        if a2[1] - a1[1] == 0:  # Vertical line
            bounds = np.array([[a1[0], a1[1] + 0.1], [a2[0], a2[1]]], dtype='float')  # Slightly tilt the vertical bound
        else:
            fx = interp1d(self.bounds[:, 1], self.bounds[:, 0], fill_value='extrapolate')
            m = (self.bounds[1, 0] - self.bounds[0, 0]) / (self.bounds[1, 1] - self.bounds[0, 1])
            shiftw = lambda c, w: np.stack([fx(c[1] + w), c[1] + w], axis=0)
            c1 = shiftw(a1, -np.sign(m) * w)
            c2 = shiftw(a2, np.sign(m) * w)
            bounds = np.stack([c1, c2], axis=0)
        return bounds

    def _clip(self, anchor):
        """
        Clips a line segment against the viewport.

        Parameters:
        anchor (np.ndarray): A 2x2 array defining the line segment as [(y1, x1), (y2, x2)].

        Returns:
        tuple:
            - np.ndarray: The clipped line segment if it intersects the viewport, otherwise an array of -1.
            - bool: True if the line segment is within the viewport, False otherwise.
        """
        y1 = anchor[0, 0]
        x1 = anchor[0, 1]
        y2 = anchor[1, 0]
        x2 = anchor[1, 1]

        code1 = self._compute_code(x1, y1)
        code2 = self._compute_code(x2, y2)

        accept = False

        while True:
            # If both endpoints lie within the rectangle
            if code1 == 0 and code2 == 0:
                accept = True
                break

            # If both endpoints are outside the rectangle
            elif (code1 & code2) != 0:
                break

            # Some segment lies within the rectangle
            else:
                # At least one of the points is outside, select it
                if code1 != 0:
                    code_out = code1
                else:
                    code_out = code2

                # Find intersection point
                if code_out & self.TOP:
                    x = x1 + ((x2 - x1) / (y2 - y1)) * (self.y_max - y1)
                    y = self.y_max

                elif code_out & self.BOTTOM:
                    x = x1 + ((x2 - x1) / (y2 - y1)) * (self.y_min - y1)
                    y = self.y_min

                elif code_out & self.RIGHT:
                    y = y1 + ((y2 - y1) / (x2 - x1)) * (self.x_max - x1)
                    x = self.x_max

                elif code_out & self.LEFT:
                    y = y1 + ((y2 - y1) / (x2 - x1)) * (self.x_min - x1)
                    x = self.x_min

                # Replace point outside the clipping rectangle by intersection point
                if code_out == code1:
                    x1 = x
                    y1 = y
                    code1 = self._compute_code(x1, y1)
                else:
                    x2 = x
                    y2 = y
                    code2 = self._compute_code(x2, y2)

        if accept:
            return np.array([[y1, x1], [y2, x2]], dtype='float'), True
        else:
            return np.full_like(anchor, -1), False

    def get(self):
        """
        Computes and returns the clipped viewport bounds.

        Returns:
        tuple:
            - np.ndarray: The clipped viewport bounds.
            - bool: True if the bounds are valid, False otherwise.
        """
        diag_length = 10 * np.linalg.norm(np.diff(self.viewport, axis=-1), axis=-2)[0]
        bounds = self._extend_bounds(diag_length)
        view_bounds, valid = self._clip(bounds)
        return view_bounds, valid