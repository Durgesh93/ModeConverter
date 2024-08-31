from modeconvertor.movie import read_grayscale
from modeconvertor import Converter, Cone
import matplotlib.pyplot as plt
import numpy as np

def save_m_mode_image(video, output_file):
    c = Cone(video)
    SLcoords = c.left(angle=30)
    mc = Converter(anchor_idx=len(video) // 2, movies=video, SLcoords=SLcoords, SR=1, W=128, A=0, S=0, R=0, num_pts=600)
    amm = mc.amm()
    
    fig, ax_m = plt.subplots()
    ax_m.imshow(amm, cmap='gray')
    
    # Remove axis and background
    ax_m.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.show()

    # Save the M-mode image
    opt = input('Do you want to save the M-mode image? (Y/n): ').lower()
    if opt == 'y':
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# Read the B-mode movie
mov = read_grayscale('echotest.avi')

# Define the output file path
output_file_path = 'm_mode_image.png'

# Save the M-mode image
save_m_mode_image(mov, output_file_path)