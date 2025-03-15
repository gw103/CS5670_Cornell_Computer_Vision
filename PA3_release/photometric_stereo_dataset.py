import numpy as np
from scipy.signal import convolve2d
from PIL import Image
import os
import utils
def ripples(sz):
    dr = sz/10
    num=3
    y, x = np.meshgrid(np.arange(sz), np.arange(sz))
    output = np.zeros((sz,sz))
    for i in range(num):
        cx = np.random.random_sample()*sz
        cy = np.random.random_sample()*sz
        y = y - cy
        x = x - cx
        r = np.sqrt(y*y + x*x)
        output = output + np.cos(2*np.pi*r/dr)
    return output

def mountains(sz):
    num=50
    y, x = np.meshgrid(np.arange(sz), np.arange(sz))
    output = np.zeros((sz, sz))
    for i in range(num):
        centerx = np.random.random_sample()*float(sz)
        centery = np.random.random_sample()*float(sz)
        sigma = np.random.random_sample()*float(sz)/5.
        scale = np.random.random_sample()*20
        gauss = np.exp(-((y-centery)**2 + (x-centerx)**2)/(2*sigma*sigma))
        gauss = scale*gauss/np.max(gauss)
        output = output + gauss
    return output
    



def heights_to_normals(Z):
    derivative = np.zeros((3,3))
    derivative[1,0] = 0.5
    derivative[1,2] = -0.5
    dzdx = convolve2d(Z, derivative,mode='valid')
    dzdy = convolve2d(Z, derivative.T,mode='valid')
    output = np.zeros((dzdx.shape[0], dzdx.shape[1], 3))
    output[:,:,0] = -dzdx
    output[:,:,1] = -dzdy
    output[:,:,2] = 1.
    output = output/np.sqrt(np.sum(output*output, axis=2, keepdims=True))
    return output


def render(normals, albedo, light):
    shading = normals.reshape((-1,3)) @ light
    shading = shading.reshape((albedo.shape[0],albedo.shape[1],1))
    image = albedo*shading

    return image

def generate_dataset(name):
    dirname = os.path.join('data/photometric_stereo/', name)
    lights = np.load(os.path.join(dirname, 'lights.npz'))['lights']
    albedo = np.array(Image.open(os.path.join(dirname, 'albedo.png')).convert('RGB')).astype(float)/255.
    sz = albedo.shape[0]
    if name=='ripples':
        Z = ripples(sz+2)
    elif name=='mountains':
        Z = mountains(sz+2)
    normals = heights_to_normals(Z)
    np.savez(os.path.join(dirname, 'normals.npz'), normals=normals)
    imglist=[]
    for k in range(lights.shape[1]):
        image = render(normals, albedo, lights[:,[k]])
        imglist.append(image)
        image = Image.fromarray((image*255.).astype(np.uint8))
        image.save(os.path.join(dirname, 'images/{:02d}.png'.format(k)))
    utils.gifwrite(imglist, os.path.join(dirname, 'animation.gif'))

def get_dataset(name, generate=True):
    if generate:
        generate_dataset(name)
    dirname = os.path.join('data/photometric_stereo/', name)
    lights = np.load(os.path.join(dirname, 'lights.npz'))['lights']

    albedo = np.array(Image.open(os.path.join(dirname, 'albedo.png')).convert('RGB')).astype(float)/255.
    normals = np.load(os.path.join(dirname, 'normals.npz'))['normals']
    images = [utils.imread(os.path.join(dirname, 'images','{:02d}.png'.format(k))) for k in range(lights.shape[1])]
    return dict(images=images, lights=lights, albedo=albedo, normals=normals)



