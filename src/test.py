import sys
import numpy as np
import cv2
import time
sys.path.insert(0, '/'.join(sys.argv[1].split('/')[:-1]))
import pynvjpeg


if __name__ == '__main__':
    decoder = pynvjpeg.Decoder(1)
    assert(decoder.init() is True)
    filename = sys.argv[2]
    image0 = decoder.imread(filename)
    image0 = image0.astype(np.int32)
    h, w, c = image0.shape
    with open(filename, 'rb') as f:
        data = f.read()

    
    image1 = cv2.imread(filename)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image1 = image1.astype(np.int32)
    if np.sum((image1 - image0)) > h * w * c:
        exit(-1)
    exit(0)
    
