# from PIL import Image
# import os.path
# import glob
# if not os.path.exists("./Data/baby2/GT_TE"): #chuangjian baocun
#     os.makedirs("./Data/baby2/GT_TE")
# def convertjpg(jpgfile,outdir,width=512,height=512):
#     img=Image.open(jpgfile)
#     new_img=img.resize((width,height),Image.BILINEAR)
#     new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
# for jpgfile in glob.glob("./Data/baby/GT_TE/*.bmp"):   #yuanlailujing
#     convertjpg(jpgfile,"./Data/baby2/GT_TE")  #baocun



#提取图片名字
import os

path_imgs = './Data/isbi/train/images'
for files in os.listdir(path_imgs):
    print(files)
    img_path = files

    with open("./Data/isbi/train.txt", "a") as f:
        f.write(str(img_path).split(".")[0] + '\n')    #split去后缀