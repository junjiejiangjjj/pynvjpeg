import sys
import numpy as np
import cv2
import time
sys.path.insert(0, '/'.join(sys.argv[1].split('/')[:-1]))

import pynvjpeg



if __name__ == '__main__':
    decoder = pynvjpeg.Decoder()
    assert(decoder.init() is True)

    filename = sys.argv[2]
    start = time.time()
    for i in range(1000):
        decoder.imread(filename)
    end = time.time()
    print(end - start)
