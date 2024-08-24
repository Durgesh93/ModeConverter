import cv2
import numpy as np
import imageio

def read_grayscale(file_path):
    """
    Reads a grayscale video from a file and returns it as a NumPy array.

    This function supports GIF, AVI, and MOV file formats. For GIF files, each frame is read as a grayscale image. For AVI and MOV files, each frame is converted to grayscale using OpenCV.

    Parameters:
    file_path (str): The path to the video file. The file extension determines the format of the video to be read.

    Returns:
    np.ndarray: A 3D NumPy array representing the grayscale video. The shape of the array is (N, H, W), where N is the number of frames, H is the height, and W is the width of each frame.

    Raises:
    IOError: If the video file cannot be opened (for AVI and MOV files).
    ValueError: If the file format is unsupported. Supported formats are .gif, .avi, and .mov.
    """
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension == 'gif':
        # Read GIF file as grayscale
        gif = imageio.mimread(file_path, as_gray=True)
        frames = [np.array(frame) for frame in gif]
        video_array = np.array(frames)
    
    elif file_extension in ['avi', 'mov']:
        # Read AVI or MOV file and convert frames to grayscale
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {file_path}")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame_gray)
        cap.release()
        video_array = np.array(frames)
    else:
        # Raise error for unsupported file formats
        raise ValueError("Unsupported file format. Please use .gif, .avi, or .mov.")
    
    # Ensure the array has the shape (N, H, W)
    video_array = video_array.transpose(0, 1, 2)
    return video_array
