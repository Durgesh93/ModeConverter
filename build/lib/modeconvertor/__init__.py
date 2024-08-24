import math
import numpy as np
from scipy.interpolate import interpn, interp1d
from sklearn.neighbors import NearestNeighbors
from .vb import view_bounds
from .cone import Cone
from collections import deque


class Converter:
    """
    A class to perform various transformations and computations on video sequences and coordinates.
    """

    def __init__(self, anchor_idx, movies, SLcoords, W=64, SR=1, num_pts=None, A=0, S=0, R=0):
        """
        Initialize the Converter class with the given parameters.

        Parameters:
        -----------
        anchor_idx : int or list of int
            Indices of the anchor points.
        movies : numpy.ndarray
            The movie clips array.
        SLcoords : numpy.ndarray
            The coordinates of SL points.
        W : int, optional
            The window size for clipping, by default 64.
        SR : int, optional
            The stride for shifting, by default 1.
        num_pts : int, optional
            The number of points to sample, by default None.
        A : int or list of int, optional
            Anchor shift values, by default 0.
        S : int or list of int, optional
            Shift values for perturbation, by default 0.
        R : int or list of int, optional
            Rotation values for perturbation, by default 0.
        """
        self.is_single_item = self._is_single_item(SLcoords)
        shape = movies.shape
        self.viewport = (0, 0, shape[-1] - 1, shape[-2] - 1)
        if num_pts is None:
            self.num_pts = max(shape[-1], shape[-2])
        else:
            self.num_pts = num_pts - 1
        self.W = W
        self.SR = SR

        self.Aidx, self.A = self._process(anchor_idx, A)
        self.clips = self._clip_and_shift_anchor(*self._process(movies))
        self.coords, = self._process(SLcoords)
        self.S, self.R = self._process(S, R)
        self.pcoords = self._perturb_coords()

    def _is_single_item(self, coords):
        """
        Check if the input coordinates represent a single item.

        Parameters:
        -----------
        coords : numpy.ndarray
            The input coordinates.

        Returns:
        --------
        bool
            True if the input is a single item, False otherwise.
        """
        if coords.ndim == 2:
            return True
        else:
            return False

    def _clip_and_shift_anchor(self, movies):
        """
        Clip and shift the movie frames based on the anchor indices.

        Parameters:
        -----------
        movies : numpy.ndarray
            The movie clips array.

        Returns:
        --------
        numpy.ndarray
            The clipped and shifted movie clips.
        """
        window_size = self.W
        stride = self.SR
        clip_l = []
        Aidx = []

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
            Aidx.append(np.array(len(window) // 2 + anchor_shift, dtype='int'))
        self.Aidx = np.stack(Aidx, axis=0)
        return np.stack(clip_l, axis=0)

    def _process(self, *args):
        """
        Process the input arguments to ensure they are in the correct format.

        Parameters:
        -----------
        *args : tuple
            The input arguments to process.

        Returns:
        --------
        tuple
            The processed arguments.
        """
        processed = []
        if self.is_single_item:
            for arg in args:
                if isinstance(arg, np.ndarray):
                    processed.append(arg[np.newaxis])
                elif isinstance(arg, int):
                    processed.append(np.array([arg], dtype='int'))
                elif isinstance(arg, float):
                    processed.append(np.array([arg], dtype='float'))
        else:
            for arg in args:
                if isinstance(arg, list):
                    arg = np.array(arg)
                processed.append(arg)
        return processed

    def _mimg_single(self, clip, per_pts):
        """
        Generate a manipulated image for a single clip.

        Parameters:
        -----------
        clip : numpy.ndarray
            The movie clip.
        per_pts : numpy.ndarray
            The perturbation points.

        Returns:
        --------
        numpy.ndarray
            The manipulated image.
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
        Compute the SL (spatial line) for a single set of coordinates.

        Parameters:
        -----------
        coords : numpy.ndarray
            The input coordinates.

        Returns:
        --------
        tuple
            A tuple containing the SL points and the coefficients.
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
        Generate reference SL (spatial line) points for the given coordinates.

        Returns:
        --------
        numpy.ndarray
            The reference SL points.
        """
        SL_list = []
        for coords in self.coords:
            SL, _ = self._compute_SL_single(coords)
            SL_list.append(SL[::self.num_pts // 10])
        return np.stack(SL_list, axis=0).squeeze(axis=0)

    def perSL(self):
        """
        Generate perturbed SL (spatial line) points for the given coordinates.

        Returns:
        --------
        numpy.ndarray
            The perturbed SL points.
        """
        SL_list = []
        for coords in self.pcoords:
            SL, _ = self._compute_SL_single(coords)
            SL_list.append(SL[::self.num_pts // 10])
        return np.stack(SL_list, axis=0).squeeze(axis=0)

    def coeffs(self):
        """
        Generate coefficients for the given coordinates.

        Returns:
        --------
        numpy.ndarray
            The coefficients for the given coordinates.
        """
        coeff_list = []
        for coords in self.pcoords:
            _, coeff = self._compute_SL_single(coords)
            coeff_list.append(coeff)
        return np.stack(coeff_list, axis=0).squeeze(axis=0)

    def _create_nn_single(self, SL):
        """
        Create a nearest neighbor model for the given SL points.

        Parameters:
        -----------
        SL : numpy.ndarray
            The SL points.

        Returns:
        --------
        sklearn.neighbors.NearestNeighbors
            The trained NearestNeighbors model.
        """
        return NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(SL)
