import ctypes
import os
import matplotlib.pyplot as plt
import sysv_ipc
from struct import unpack

from PIL import Image
import numpy as np
import cv2

from time_utils import time_it

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


@time_it
def get_screen_shm(shared_memory, w, h):
    fmt = w * h * 3 * 'B'
    memory_value = shared_memory.read()
    data = unpack(fmt, memory_value)
    img_data = np.array(data, dtype=np.uint8).reshape((h, w, 3))
    return img_data


if __name__ == '__main__':
    # im = get_screen(0, 0, 640, 480, ":1")
    # cv2.imwrite('/data/1.png', im)
    shared_memory = sysv_ipc.SharedMemory(9203)
    im = get_screen_shm(shared_memory, 640, 480)
    # plt.imshow(im), plt.show()
    cv2.imwrite('/data/1.png', im)
