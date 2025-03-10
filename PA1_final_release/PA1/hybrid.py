import numpy as np
import matplotlib.pyplot as plt

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''



    # TODO-BLOCK-BEGIN
    if img.ndim == 2:
        h, w = img.shape
        c = 1
        padded_height = h + kernel.shape[0]
        padded_width = w + kernel.shape[1]
        img_padded = zero_pad(img, padded_height, padded_width)
    else:
        h, w, c = img.shape
        padded_height = h + kernel.shape[0]
        padded_width = w + kernel.shape[1]
        img_padded = zero_pad(img, padded_height, padded_width)
    new_img = np.zeros((h, w, c))

    kH, kW = kernel.shape

    for i in range(kH):
        for j in range(kW):
            new_img += kernel[i, j] * img_padded[i:i+h, j:j+w, :]

    # If the input was grayscale, remove the singleton channel dimension.
    if img.ndim == 2:
        new_img = new_img[:, :, 0]
    return new_img
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END
def zero_pad(img, pad_height, pad_width):
    if img.ndim == 2:
        h, w = img.shape
    else:
        h, w, _ = img.shape
    pad_height = pad_height - h
    pad_width = pad_width - w
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    if img.ndim == 2:
        top = np.zeros((pad_top, w))
        bottom = np.zeros((pad_bottom, w))
        left = np.zeros((h + pad_height, pad_left))
        right = np.zeros((h + pad_height, pad_right))
        padded = np.vstack((top, img, bottom))
        padded = np.hstack((left, padded, right))

    else:
        top = np.zeros((pad_top, w, 3))
        bottom = np.zeros((pad_bottom, w, 3))
        left = np.zeros((h + pad_height, pad_left, 3))
        right = np.zeros((h + pad_height, pad_right, 3))
        padded = np.vstack((top, img, bottom))
        padded = np.hstack((left, padded, right))
    return padded

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    kh, kw = kernel.shape
    kernel = kernel.flatten()
    new_kernel = []
    for i in range(1,len(kernel)+1):
        new_kernel.append(kernel[len(kernel)-i])
    new_kernel = np.array(new_kernel)
    new_kernel = new_kernel.reshape(kh, kw)
    return cross_correlation_2d(img, new_kernel)
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
    kernel = np.zeros((height, width))
    h_center = height // 2 + 1
    w_center = width // 2 + 1
    for i in range(height):
        for j in range(width):
            x = i - h_center
            y = j - w_center
            kernel[i,j] = np.exp(-(x**2 + y**2)/(2*sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    return convolve_2d(img, kernel)
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    return img - low_pass(img, sigma, size)
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)
