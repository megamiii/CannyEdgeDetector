import os
from PIL import Image
import math
import numpy as np

def gaussian_function(sigma):
    """
    Args:
        sigma (float): standard deviation of the gaussian distribution

    Return:
        gaussian kernel (numpy array)
    """

    # Your code (optional)
    return

def convolution(array, kernel):
    """
    Args:
        array (numpy array): input array
        kernel (numpy array): kernel array

    Return:
        output of image convolution (numpy array)
    """

    # Your code (optional)
    return

def gaussian_filter(input_image, sigma):
    """
    Args:
        input_image (numpy array): input array
        sigma (float): standard deviation of the gaussian distribution

    Return:
        2D gaussian filtered image (numpy array)
    """

    # Your code
    return

def sobel_filter(image):
    """
    Args:
        image (numpy array): input array

    Return:
        Sobel filtered image of gradient magnitude (numpy array),
        Sobel filtered image of gradient direction (numpy array)
    """

    # Your code
    return

def non_max_suppression(gradient_magnitude, gradient_orientation):
    """
    Args:
        gradient_magnitude (numpy array): Sobel filtered image of gradient magnitude
        gradient_orientation (numpy array): Sobel filtered image of gradient direction

    Return:
        Non-maximum suppressed image (numpy array)
    """

    # Your code
    return

def double_threshold(image, low_threshold_rate, high_threshold_rate):
    """
    Args:
        image (numpy array): Non-maximum suppressed image
        low_threshold_rate (float): used to identify non-relevant pixels
        high_threshold_rate (float): used to identify strong pixels

    Return:
        Double-thresholded image (numpy array)
    """

    # Your code
    return

def hysteresis_threshold(image):
    """
    Args:
        image (numpy array): Double-thresholded image

    Return:
        Hysteresis-thresholded image (numpy array)
    """

    # Your code
    return

if __name__ == "__main__":
    sigma = 1.5
    low_threshold_rate = 0.05
    high_threshold_rate = 0.15

    img = np.asarray(Image.open(os.path.join('images', 'son.jpg')).convert('L'))
    img = img.astype('float32')

    logdir = os.path.join('results', 'HW1_2')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    ret = gaussian_filter(img, sigma)
    if ret is not None:
        Image.fromarray(ret.astype('uint8')).save(os.path.join(logdir, 'gaussian_filter.jpeg'))
        Image.fromarray(ret.astype('uint8')).show()

    grad_magnitude, grad_direction = sobel_filter(ret)
    if grad_magnitude is not None and grad_direction is not None:
        Image.fromarray(grad_magnitude.astype('uint8')).save(os.path.join(logdir, 'sobel_filter_grad_magnitude.jpeg'))
        Image.fromarray(grad_magnitude.astype('uint8')).show()
        Image.fromarray(grad_direction.astype('uint8')).save(os.path.join(logdir, 'sobel_filter_grad_direction.jpeg'))
        Image.fromarray(grad_direction.astype('uint8')).show()

    ret = non_max_suppression(grad_magnitude, grad_direction)
    if ret is not None:
        Image.fromarray(ret.astype('uint8')).save(os.path.join(logdir, 'non_max_suppression.jpeg'))
        Image.fromarray(ret.astype('uint8')).show()

    ret = double_threshold(ret, low_threshold_rate, high_threshold_rate)
    if ret is not None:
        Image.fromarray(ret.astype('uint8')).save(os.path.join(logdir, 'double_threshold.jpeg'))
        Image.fromarray(ret.astype('uint8')).show()

    ret = hysteresis_threshold(ret)
    if ret is not None:
        Image.fromarray(ret.astype('uint8')).save(os.path.join(logdir, 'hysteresis_threshold.jpeg'))
        Image.fromarray(ret.astype('uint8')).show()