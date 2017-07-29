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


def get_screen_shm(shared_memory):
    memory_value = shared_memory.read()
    w, h = unpack('ii', memory_value[:8])
    data = unpack('B' * w * h * 3, memory_value[8:])
    img_data = np.array(data, dtype=np.uint8).reshape((h, w, 3))
    return img_data

def get_screen_size(shared_memory):
    memory_value = shared_memory.read()
    w, h = unpack('ii', memory_value[:8])
    return w, h


if __name__ == '__main__':
    # im = get_screen(0, 0, 640, 480, ":45")
    # cv2.imwrite('/data/2.png', im)
    shared_memory = sysv_ipc.SharedMemory(9301)
    im = get_screen_shm(shared_memory)
    plt.imshow(im), plt.show()
    # plt.imshow(im), plt.show()
    cv2.imwrite('/data/1.png', im)
