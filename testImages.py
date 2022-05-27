import cv2
import os
import numpy as np

def to_MNIST(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    correction = np.log(0.5*255)/np.log(np.mean(gray))
    gamma = np.power(gray, correction).clip(0,255).astype(np.uint8)
    contrast = cv2.convertScaleAbs(gamma, alpha=2.5)
    invert = cv2.bitwise_not(contrast)
    resized = cv2.resize(invert,(28,28))
    return resized


for image in os.listdir('images/'):
    img = cv2.imread('images/'+image)
    foo = to_MNIST(img)
    cv2.imwrite('conversions/MNIST_'+image,foo)
