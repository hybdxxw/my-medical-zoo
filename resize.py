from PIL import Image
import os.path
import glob
if not os.path.exists("./Data/DRIVE2/masks"):
    os.makedirs("./Data/DRIVE2/masks")
def convertjpg(jpgfile,outdir,width=576,height=576):
    img=Image.open(jpgfile)
    new_img=img.resize((width,height),Image.BILINEAR)
    new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
for jpgfile in glob.glob("./Data/DRIVE1/masks/*.png"):
    convertjpg(jpgfile,"./Data/DRIVE2/masks")

