import os
from PIL import Image

original_path = u'./Data/DRIVE/images'
saved_path = u'./Data/DRIVE1/images'
if not os.path.exists("./Data/DRIVE1/images"):
        os.makedirs("./Data/DRIVE1/images")

counts = 0
files = os.listdir(original_path)
for file in files:
    if file.endswith('tif'):
        tif_file = os.path.join(original_path, file)

        file = file[:-3] + 'png'
        png_file = os.path.join(saved_path, file)
        im = Image.open(tif_file)
        im.save(png_file)
        print(png_file)
        counts += 1

print('%d done' %counts)