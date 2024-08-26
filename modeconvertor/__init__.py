import math
import numpy as np
from scipy.interpolate import interpn, interp1d
from sklearn.neighbors import NearestNeighbors
from .vb import view_bounds
from .cone import Cone
from collections import deque


class Converter:
    def __init__(self, anchor_idx, SLcoords,movies, W=64, SR=1, num_pts=None, A=0, S=0, R=0):
        """
        Initializes the Converter class with the given parameters.

        Parameters:
        - anchor_idx (int or list of int): Indices of the anchor points.
        - movies (numpy.ndarray): The movie clips array.
        - SLcoords (numpy.ndarray): The coordinates of SL points.
        - W (int, optional): The window size for clipping (default is 64).
        - SR (int, optional): The stride for shifting (default is 1).
        - num_pts (int, optional): The number of points to sample (default is None).
        - A (int or list of int, optional): Anchor shift values (default is 0).
        - S (int or list of int, optional): Shift values for perturbation (default is 0).
        - R (int or list of int, optional): Rotation values for perturbation (default is 0).
        """
        self.SLcoords       = SLcoords
        self.is_single_item = self._is_single_item()
        shape               = movies.shape
        self.viewport       = (0, 0, shape[-1] - 1, shape[-2] - 1)
        if num_pts is None:
            self.num_pts    = max(shape[-1], shape[-2])
        else:
            self.num_pts    = num_pts - 1
        self.W              = W
        self.SR             = SR
        self.Aidx, self.A   = self._process(anchor_idx, A)
        self.clips          = self._clip_and_shift_anchor(*self._process(movies))
        self.coords,        = self._process(SLcoords)
        self.S, self.R      = self._process(S, R)
        self.pcoords        = self._perturb_coords()

    def _is_single_item(self):
        """
        Checks if the input coordinates are a single item (2D array).

        Parameters:
        - coords (numpy.ndarray): Coordinates to check.

        Returns:
        - bool: True if coordinates are a single item, False otherwise.
        """
        return self.SLcoords.ndim == 2

    def _clip_and_shift_anchor(self, movies):
        """
        Clips and shifts movie clips based on anchor points.

        Parameters:
        - movies (numpy.ndarray): The movie clips array.

        Returns:
        - numpy.ndarray: The clipped and shifted movie clips.
        """
        window_size = self.W
        stride = self.SR
        clip_l = []
    
        if window_size % 2 == 0:
            shiftl = window_size // 2
            shiftr = window_size // 2
        else:
            shiftl = window_size // 2
            shiftr = window_size // 2 + 1

        for movie, anchor_idx, anchor_shift in zip(movies, self.Aidx, self.A):
            anchor_idx = int(anchor_idx)
            anchor_shift = int(anchor_shift)
            anchor_shift = np.clip(anchor_shift, -window_size // 2, window_size // 2)
            ids = [x.item() for x in np.arange(movie.shape[-3], dtype='int')]
            window_l = []
            window_r = []
            idsl = deque(ids.copy())
            for i in range(shiftl + anchor_shift):
                idsl.rotate(stride)
                window_l.append(idsl[anchor_idx])
            idsr = deque(ids.copy())
            for i in range(shiftr - anchor_shift):
                window_r.append(idsr[anchor_idx])
                idsr.rotate(-stride)
            window = window_l[::-1] + window_r
            clip = movie.take(window, axis=-3)
            clip_l.append(clip)
        return np.stack(clip_l, axis=0)

    def _process(self, *args):
        """
        Processes input arguments to ensure they are in the correct format.

        Parameters:
        - *args: Variable number of arguments to process.

        Returns:
        - list: Processed arguments.
        """
        processed = []
        if self.is_single_item:
            for arg in args:
                if isinstance(arg, np.ndarray):
                    processed.append(arg[np.newaxis])
                elif isinstance(arg, (int,float)):
                    processed.append(np.array([arg]))
        else:
            for arg in args:
                if isinstance(arg, list):
                    arg = np.array(arg)
                elif isinstance(arg, (int,float)):
                    arg = np.full(self.SLcoords.shape[0],arg)
                processed.append(arg)
        return processed

    def _mimg_single(self, clip, per_pts):
        """
        Generates a manipulated image for a single clip using given points.

        Parameters:
        - clip (numpy.ndarray): The movie clip to process.
        - per_pts (numpy.ndarray): Perturbation points.

        Returns:
        - numpy.ndarray: The manipulated image.
        """
        q, h, w = clip.shape[-3:]
        if self.is_single_item and len(clip.shape) == 4:
            c = clip.shape[0]
        else:
            c = None
        m_image = []
        if c is not None:
            for j in range(q):
                m_image_cyx = []
                for i in range(c):
                    m_image_yx = interpn((np.arange(0, h), np.arange(0, w)), clip[i, j], per_pts)
                    m_image_cyx.append(m_image_yx)
                m_image_cyx = np.stack(m_image_cyx, axis=0)
                m_image.append(m_image_cyx)
        else:
            for j in range(q):
                m_image_yx = interpn((np.arange(0, h), np.arange(0, w)), clip[j], per_pts)
                m_image.append(m_image_yx)
        m_image = np.stack(m_image, axis=-1)
        return m_image.astype(np.uint8)

    def _compute_SL_single(self, coords):
        """
        Computes the Spatial Line (SL) and its coefficients for given coordinates.

        Parameters:
        - coords (numpy.ndarray): Coordinates to compute SL for.

        Returns:
        - tuple: (probability points, coefficients)
        """
        ref_coords = coords[:, [0, -1]]
        vb, valid = view_bounds(ref_coords, mode='coord', viewport=self.viewport).get()
        if not valid:
            vb = ref_coords
        coeffs = np.array(np.polyfit(vb[:, 1], vb[:, 0], 1), dtype=np.single)
        y, x = vb[:, 0], vb[:, 1]
        distance = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0) ** 2 + np.ediff1d(y, to_begin=0) ** 2))
        distance = distance / distance[-1]
        fx, fy = interp1d(distance, x), interp1d(distance, y)
        alpha = np.linspace(0, 1, self.num_pts)
        x_idx, y_idx = fx(alpha), fy(alpha)
        prob_points = np.stack([y_idx, x_idx], axis=-1)
        return prob_points, coeffs

    def refSL(self):
        """
        Generates reference SL (spatial line) points for the original coordinates.

        Returns:
        - numpy.ndarray: The reference SL points.
        """
        SL_list = []
        for coords in self.coords:
            SL, _ = self._compute_SL_single(coords)
            SL_list.append(SL[::self.num_pts // 10])
        SL_list = np.stack(SL_list, axis=0)
        if self.is_single_item:
            return SL_list.squeeze(axis=0)
        else:
            return SL_list

    def perSL(self):
        """
        Generates perturbed SL (spatial line) points for the perturbed coordinates.

        Returns:
        - numpy.ndarray: The perturbed SL points.
        """
        SL_list = []
        for coords in self.pcoords:
            SL, _ = self._compute_SL_single(coords)
            SL_list.append(SL[::self.num_pts // 10])
        SL_list = np.stack(SL_list, axis=0)
        if self.is_single_item:
            return SL_list.squeeze(axis=0)
        else:
            return SL_list

    def coeffs(self):
        """
        Generates coefficients for the perturbed SL (spatial line) points.

        Returns:
        - numpy.ndarray: The coefficients.
        """
        coeff_list = []
        for coords in self.pcoords:
            _, coeff = self._compute_SL_single(coords)
            coeff_list.append(coeff)
        coeff_list=np.stack(coeff_list, axis=0)
        if self.is_single_item:
            return coeff_list.squeeze(axis=0)
        else:
            return coeff_list

    def _create_nn_single(self, SL):
        """
        Creates a nearest neighbors model for given SL points.

        Parameters:
        - SL (numpy.ndarray): Spatial line points.

        Returns:
        - sklearn.neighbors.NearestNeighbors: The nearest neighbors model.
        """
        return NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(SL)

    def _perturb_coords(self):
        """
        Perturbs the coordinates by applying shifts and rotations.

        Returns:
        - numpy.ndarray: The perturbed coordinates.
        """
        pcoords_l = []
        for coords, shift, rotate in zip(self.coords, self.S, self.R):
            pcoords = []
            for co in coords:
                pivot = co.mean(axis=-1, keepdims=True)
                angle_radians = math.radians(rotate)
                rotation_matrix = np.array([[math.cos(angle_radians), -math.sin(angle_radians)],
                                            [math.sin(angle_radians), math.cos(angle_radians)]])
                pcoords.append(np.dot(co - pivot, rotation_matrix) + pivot + shift)
            pcoords = np.stack(pcoords, axis=0)
            pcoords_l.append(pcoords)
        pcoords_l = np.stack(pcoords_l, axis=0)
        return pcoords_l

    def mcoords(self):
        """
        Generates manipulated coordinates based on perturbed SL and anchor points.

        Returns:
        - numpy.ndarray: The manipulated coordinates.
        """
        SL_x = []
        for anchor_shift in self.A:
            SL_x.append(self.W//2+anchor_shift)
        SL_x = np.stack(SL_x,axis=0)
        
        m_coords = []
        for coords, slx in zip(self.pcoords, SL_x):
            SL, coeffs = self._compute_SL_single(coords)
            nn = self._create_nn_single(SL)
            d, i = nn.kneighbors(coords)
            m_ycoords_s = i.ravel()
            m_xcoords_s = np.full_like(m_ycoords_s, slx)
            m_coords_s = np.stack([m_ycoords_s, m_xcoords_s], axis=-1)
            m_coords.append(m_coords_s)
        m_coords = np.stack(m_coords, axis=0)
        if self.is_single_item:
            return m_coords.squeeze(axis=0)
        else:
            return m_coords

    def mSLx(self):
        """
        Gets the x-location of manipulated SL.

        Returns:
        - int or numpy.ndarray: The x-location of manipulated SL.
        """
        SL_x = []
        for anchor_shift in self.A:
            SL_x.append(self.W//2+anchor_shift)
        SL_x = np.stack(SL_x,axis=0)
        if self.is_single_item:
            return SL_x.squeeze(axis=0)
        else:
            return SL_x

    def mSL(self):
        """
        Generates manipulated SL (spatial line) coordinates based on anchor indices.

        Returns:
        - numpy.ndarray: The manipulated SL coordinates.
        """

        SL_x = []
        for anchor_shift in self.A:
            SL_x.append(self.W//2+anchor_shift)
        SL_x = np.stack(SL_x,axis=0)

        m_SL = []
        for slx in SL_x:
            m_SLycoords = np.arange(0, self.num_pts)[::self.num_pts // 10]
            m_SLxcoords = np.full_like(m_SLycoords, slx)
            m_SLcoords = np.stack([m_SLycoords, m_SLxcoords], axis=-1)
            m_SL.append(m_SLcoords)
        m_SL=np.stack(m_SL, axis=0)
        if self.is_single_item:
            return m_SL.squeeze(axis=0)
        else:
            return m_SL

    def pred_bcoords(self, pred_mcoords):
        """
        Predicts the original coordinates from the manipulated coordinates.

        Parameters:
        - pred_mcoords (numpy.ndarray): The manipulated coordinates to predict from.

        Returns:
        - numpy.ndarray: The predicted original coordinates.
        """
        pred_mcoords, = self._process(pred_mcoords)
        b_coords = []
        for y_pred_s, bpcoords in zip(pred_mcoords.astype(np.uint16)[:, :, 0], self.pcoords):
            SL, coeffs = self._compute_SL_single(bpcoords)
            b_coords_s = SL[y_pred_s]
            b_coords.append(b_coords_s)
        b_coords = np.stack(b_coords, axis=0)
        if self.is_single_item:
            return b_coords.squeeze(axis=0)
        else:
            return b_coords

    def bclip(self):
        if self.is_single_item:
            return self.clips.squeeze(axis=0)
        else:
            return self.clips
    
    def amm(self):
        """
        Applies the manipulated SL coordinates to the movie clips to generate manipulated images.

        Returns:
        - numpy.ndarray: The manipulated images.
        """
        m_img = []
        for clip, pcoords in zip(self.clips, self.pcoords):
            SL, coeffs = self._compute_SL_single(pcoords)
            m_img.append(self._mimg_single(clip, SL))
        m_img = np.stack(m_img, axis=0)
        if self.is_single_item:
            return m_img.squeeze(axis=0)
        else:
            return m_img
