import datetime
import glob
import os.path
from tkinter import Button, Tk

import imageio

from view_gif import MyLabel

IMAGE_DIR = "/Users/schiba/Projects/image-generator/images/*.png"
IMAGE_SAVE_DIR = "/Users/schiba/Projects/image-generator/images/gifs/"

filenames = glob.glob(IMAGE_DIR)

images = []
for filename in filenames:
    images.append(imageio.imread(filename))

if not os.path.isdir(IMAGE_SAVE_DIR):
    os.mkdir(IMAGE_SAVE_DIR)

export_name = os.path.join(IMAGE_SAVE_DIR,
                           f"{datetime.datetime.now().strftime('%d%m%y-%H%M%S')}.gif")
imageio.mimsave(export_name, images, format='GIF', duration=0.5)

root = Tk()
anim = MyLabel(root, export_name)
anim.pack()


def stop_it():
    anim.after_cancel(anim.cancel)


Button(root, text='stop', command=stop_it).pack()

root.mainloop()

for filename in filenames:
    os.remove(filename)
