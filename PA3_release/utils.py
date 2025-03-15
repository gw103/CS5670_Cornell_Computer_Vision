from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

def imread(imname, gray=False):
    if gray:
        img = np.array(Image.open(imname).convert('L')).astype(float)/255.
    else:
        img = np.array(Image.open(imname).convert('RGB')).astype(float)/255.
    return img


def normalize(img):
    img = img - np.min(img)
    img = img/np.max(img)
    return img

def imwrite(img, imname):
    Image.fromarray((img*255).astype(np.uint8)).save(imname)


def gifwrite(imglist, imname):
    imglist = [Image.fromarray((img*255).astype(np.uint8)) for img in imglist]
    imglist[0].save(imname, save_all=True, append_images=imglist[1:],duration=100,loop=1)
    

        
