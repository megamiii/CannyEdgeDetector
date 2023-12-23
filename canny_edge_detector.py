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
    # Define the size of the 1D kernel
    kernel_size = int(6*sigma + 1) # kernel size = 7 (6 + 1) ->  3 left-pixels + central pixel + 3 right-pixels

    # Make sure the size is odd (to center the kernel perfectly over the image)
    if (kernel_size % 2 == 0):
        kernel_size += 1

    # Create a 1D grid of x coordinates centered around 0 [-3 -2 -1 0 1 2 3]
    x = np.linspace(
        -(kernel_size // 2),
        kernel_size // 2,
        kernel_size
    )
    
    # 1D Gaussian formula
    gauss = np.exp(-x**2 / (2*sigma**2))
    
    # Normalize
    gauss /= gauss.sum()
    
    # 2D Gaussian kernel is the outer product of two 1D Gaussians
    kernel = np.outer(gauss, gauss)
    
    return kernel

def convolution(array, kernel):
    """
    Args:
        array (numpy array): input array
        kernel (numpy array): kernel array

    Return:
        output of image convolution (numpy array)
    """

    # Your code (optional)
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2
    output = np.zeros_like(array)
    padding = np.pad(array, pad, mode='constant') # padded array

    rows, cols = array.shape

    for i in range(rows):
        for j in range(cols):
            output[i, j] = np.sum(padding[i:i+kernel_size, j:j+kernel_size] * kernel)
    
    return output

def gaussian_filter(input_image, sigma):
    """
    Args:
        input_image (numpy array): input array
        sigma (float): standard deviation of the gaussian distribution

    Return:
        2D gaussian filtered image (numpy array)
    """

    # Your code
    return convolution(input_image, gaussian_function(sigma))

def sobel_filter(image):
    """
    Args:
        image (numpy array): input array

    Return:
        Sobel filtered image of gradient magnitude (numpy array),
        Sobel filtered image of gradient direction (numpy array)
    """

    # Your code
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    grad_x = convolution(image, sobel_x)
    grad_y = convolution(image, sobel_y)
    
    grad_magnitude = np.sqrt(pow(grad_x, 2) + pow(grad_y, 2))
    grad_direction = np.rad2deg(np.arctan2(grad_y, grad_x)) % 180

    return grad_magnitude, grad_direction

def non_max_suppression(gradient_magnitude, gradient_orientation):
    """
    Args:
        gradient_magnitude (numpy array): Sobel filtered image of gradient magnitude
        gradient_orientation (numpy array): Sobel filtered image of gradient direction

    Return:
        Non-maximum suppressed image (numpy array)
    """

    # Your code
    # Create an empty suppressed image
    nms_image = np.zeros_like(gradient_magnitude)

    rows, cols = grad_magnitude.shape
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):

            gradient_direction = gradient_orientation[i, j] # Gradient direction at the current pixel
            
            if (0 <= gradient_direction < 22.5) or (157.5 <= gradient_direction < 180):
                p, r = gradient_magnitude[i, j - 1], gradient_magnitude[i, j + 1] # left and right pixels
            elif (22.5 <= gradient_direction < 67.5):
                p, r = gradient_magnitude[i + 1, j - 1], gradient_magnitude[i - 1, j + 1] # bottom-left and top-right pixels
            elif (67.5 <= gradient_direction < 112.5):
                p, r = gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j] # top and down pixels
            else: # elif (112.5 <= gradient_direction < 157.5)
                p, r = gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1] # top-left and bottom-right pixels

            # Compare the edge strength of the current pixel q (gradient_magnitude[i, j]) with those in positive and negative pixels (p and r)
            # If the strength at q is the largest, preserve it, and otherwise, suppress it (i.e. set to 0).
            if (gradient_magnitude[i, j] >= p) and (gradient_magnitude[i, j] >= r):
                nms_image[i, j] = gradient_magnitude[i, j]            

    return nms_image

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
    high_threshold = image.max() * high_threshold_rate
    low_threshold = image.max() * low_threshold_rate

    strong_pixel = 255
    weak_pixel = 50
    # suppressed_pixel = 0

    dt_image = np.zeros_like(image)

    strong_i, strong_j = np.where(image >= high_threshold) # Definetely an edge
    weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold)) # Maybe an edge
    # suppressed_i, suppressed_j = np.where(image < low_threshold) -> Definetely not an edge

    dt_image[strong_i, strong_j] = strong_pixel
    dt_image[weak_i, weak_j] = weak_pixel

    return dt_image

def hysteresis_threshold(image):
    """
    Args:
        image (numpy array): Double-thresholded image

    Return:
        Hysteresis-thresholded image (numpy array)
    """

    # Your code
    strong_pixel = 255
    weak_pixel = 50
    
    rows, cols = image.shape

    ht_image = image.copy()

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Check if the weak pixel is connected to a strong pixel
            if (ht_image[i, j] == weak_pixel):
                if (np.any(ht_image[i-1:i+2, j-1:j+2])):
                    ht_image[i, j] = strong_pixel # If a weak edge is connected to a strong edge in the neighborhood, it becomes an edge
                else:
                    ht_image[i, j] = 0
            
    return ht_image

if __name__ == "__main__":
    sigma = 1.5
    low_threshold_rate = 0.05
    high_threshold_rate = 0.15

    img = np.asarray(Image.open(os.path.join('images', 'son.jpg')).convert('L'))
    img = img.astype('float32')

    logdir = os.path.join('results', 'canny_edge_detector')
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