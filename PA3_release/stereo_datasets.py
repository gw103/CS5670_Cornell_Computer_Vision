import numpy as np
import utils

def get_dataset(name):
    assert name in ['Teddy', 'Adirondacks'], 'Dataset name should be Teddy or Adirondacks'
    if name=='Teddy':
        left = utils.imread('data/stereo/teddy/im2.png')
        right = utils.imread('data/stereo/teddy/im6.png')
        dmax = 64
        gt = np.load('data/stereo/teddy/disparity.npz')['disparity']
    elif name=='Adirondacks':
        left = utils.imread('data/stereo/Adirondack-perfect/im0.png')
        right = utils.imread('data/stereo/Adirondack-perfect/im1.png')
        dmax = 70
        gt = np.load('data/stereo/Adirondack-perfect/disparity.npz')['disparity']
    return dict(left=left, right=right, dmax=dmax, gt=gt)

def evaluate(predicted, gt):
    good = (gt>0) & ~np.isinf(gt)
    return np.mean(np.abs(predicted[good]-gt[good])>1)

