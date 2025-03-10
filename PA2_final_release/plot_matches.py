
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

from feature_detection import computeHarrisValues, detectCorners, computeMOPSDescriptors, produceMatches

GREEN = (0, 255, 0)
RED = (0, 0, 255)
AQUAMARINE = (212, 255, 127)

def thresholdKeyPoints(keypoints, threshold):
    return [kp for kp in keypoints if kp[3] >= threshold]

def concatImages(imgs):
    # Skip Nones
    imgs = [img for img in imgs if img is not None]
    maxh = max([img.shape[0] for img in imgs]) if imgs else 0
    sumw = sum([img.shape[1] for img in imgs]) if imgs else 0
    vis = np.zeros((maxh, sumw, 3), np.uint8)
    vis.fill(255)
    accumw = 0
    for img in imgs:
        h, w = img.shape[:2]
        vis[:h, accumw:accumw+w, :] = img
        accumw += w

    return vis

def drawMatches(img1, kp1, img2, kp2, matches):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    vis = concatImages([img1, img2])

    kp_pairs = [[kp1[m[0]], kp2[m[1]]] for m in matches]
    status = np.ones(len(kp_pairs), bool)
    p1 = np.int32([kpp[0][:2] for kpp in kp_pairs])
    p2 = np.int32([kpp[1][:2] for kpp in kp_pairs]) + (w1, 0)

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.circle(vis, (x1, y1), 5, GREEN, 2)
            cv2.circle(vis, (x2, y2), 5, GREEN, 2)
        else:
            r = 5
            thickness = 6
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), RED, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), RED, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), RED, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), RED, thickness)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), AQUAMARINE)

    return vis

def drawPercent(percent, img1, img2, threshold_x, kpts1, kpts2):
    thresholdedKeypoints1 = thresholdKeyPoints(kpts1, 10 ** threshold_x)
    thresholdedKeypoints2 = thresholdKeyPoints(kpts2, 10 ** threshold_x)
    mopsDesc1 = computeMOPSDescriptors(img1, thresholdedKeypoints1)
    mopsDesc2 = computeMOPSDescriptors(img2, thresholdedKeypoints2)

    matches = produceMatches(mopsDesc1, mopsDesc2)
    matches = sorted(matches, key = lambda x : x[2])

    if matches is not None:
        matchCount = int(float(percent) * len(matches) / 100)
        matches = matches[:matchCount]
        if len(matches) != 0:
            return drawMatches(img1,thresholdedKeypoints1, img2, thresholdedKeypoints2, matches)
        else: 
            return concatImages([img1, img2])
        




    
