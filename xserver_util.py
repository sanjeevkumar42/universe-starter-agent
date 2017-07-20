import ctypes
import os

import sysv_ipc
from PIL import Image
import numpy as np
import cv2

printscr = ctypes.cdll.LoadLibrary('clib/printscr.so')


# @time_it
def get_screen(x, y, w, h, display=None):
    size = w * h
    objlength = size * 3

    printscr.getScreen.argtypes = []
    result = (ctypes.c_ubyte * objlength)()

    printscr.getScreen(ctypes.c_char_p(display.encode()), x, y, w, h, result)
    im = Image.frombuffer('RGB', (w, h), result, 'raw', 'RGB', 0, 1)
    image = np.array(im.getdata(), dtype=np.uint8).reshape(h, w, 3)
    return image


def get_screen_shm(shmid, w, h):
    memory = sysv_ipc.attach(shmid)
    data = memory.read(byte_count=w * h * 3)
    im = np.frombuffer(data, dtype=np.uint8)
    return im


if __name__ == '__main__':
    im = get_screen(0, 0, 640, 480, ":1")
    cv2.imwrite('/data/1.png', im)
    # import matplotlib.pyplot as plt
    #
    # im = get_screen_shm(154763281, 1024, 768)
    # plt.imshow(im), plt.show()
