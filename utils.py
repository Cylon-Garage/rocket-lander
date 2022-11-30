import os
import numpy as np
import matplotlib.image as mpimg
from subprocess import Popen


def make_transparent_dragon():
    img = mpimg.imread('images/dragon.png')
    idx = img[:, :, 3] == 1
    img[idx, 3] = 0.4
    mpimg.imsave('images/dragon_optimal.png', img)


def create_animation_video(path, output):
    if os.path.isfile(output):
        os.remove(output)
    with Popen(
            "ffmpeg -framerate 30 -pattern_type glob -i '%s/*.png' -c:v libx264 -pix_fmt yuv420p %s" % (path, output), shell=True) as p:
        pass
