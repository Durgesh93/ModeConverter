from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import cv2

class MoviePlayer:
    def __init__(self, video_array, interval=30):
        """
        Initializes the MoviePlayer with a video array and frame interval.

        Parameters:
        video_array (np.ndarray): A 4D numpy array with shape (num_frames, height, width, channels)
                                  representing the video data.
        interval (int): Time in milliseconds between frames. Default is 30 ms.
        """
        self.video_array = video_array
        self.num_frames = video_array.shape[0]
        self.interval = interval
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        self.fig.set_facecolor('yellow')
        self.plot_hooks = []

    def update_frame(self, i):
        """
        Updates the frames of the plot for the animation.

        Parameters:
        i (int): The index of the frame to display.
        """
        self.ax1.clear()
        self.ax1.axis('off')
        self.ax1.set_facecolor('darkgray')
        self.ax2.clear()
        self.ax2.axis('off')
        self.ax2.set_facecolor('darkgray')
        
        self.ax1.imshow(self.video_array[i], cmap='gray')
        for hook in self.plot_hooks:
            hook(self.ax1, self.ax2, self.video_array)

    def register(self, hooks):
        """
        Registers one or more hooks to be called on each frame update.

        Parameters:
        hooks (callable or list of callables): Functions to be called with the current axes and video array.
        """
        if not isinstance(hooks, list):
            hooks = [hooks]
        self.plot_hooks = hooks

    def play(self):
        """
        Starts the video playback as an animation.
        """
        self.anim = FuncAnimation(self.fig, self.update_frame, frames=self.num_frames, interval=self.interval)
        plt.show()


def test_movie(num_frames, height, width, rect_size=64):
    """
    Generates a test movie with simple rectangular motion for visual testing.

    Parameters:
    num_frames (int): Number of frames in the movie.
    height (int): Height of each frame.
    width (int): Width of each frame.
    rect_size (int): Size of the moving rectangle. Default is 64.

    Returns:
    np.ndarray: A 4D numpy array of shape (num_frames, height, width) representing the generated movie.
    """
    frames = []
    for i in range(num_frames):
        frame = 0.1 * np.ones((height, width), np.float32) * 255
        pos = int((i / (num_frames - 1)) * (min(width, height) - rect_size))
        top_left = (pos, pos)
        bottom_right = (pos + rect_size, pos + rect_size)
        if bottom_right[0] <= width and bottom_right[1] <= height:
            cv2.rectangle(frame, top_left, bottom_right, (2 * i / 3 / 3, i, i // 2), -1)
        frame = np.clip(frame, 0, 255)
        frame = frame.astype(np.uint8)
        frames.append(frame)
    return np.array(frames)
