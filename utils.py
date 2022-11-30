import numpy as np
import matplotlib.image as mpimg
import glob
from PIL import Image


def make_transparent_dragon():
    img = mpimg.imread('images/dragon.png')
    idx = img[:, :, 3] == 1
    img[idx, 3] = 0.4
    mpimg.imsave('images/dragon_optimal.png', img)


def create_gif(path):
    img_paths = sorted(glob.glob('%s/*.png' % path))

    imgs = [Image.open(i) for i in img_paths]

    imgs[0].save('test.gif',
                 save_all=True,
                 append_images=imgs[1:],
                 duration=100,
                 loop=0)
