import sys
sys.path.insert(0, '/'.join(sys.argv[1].split('/')[:-1]))

import pynvjpeg
decoder = pynvjpeg.Decoder()
print(decoder.init())
