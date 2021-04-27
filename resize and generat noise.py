import argparse
import glob
from PIL import Image
import PIL
import random
import os
import numpy as np
import random
import skimage
from skimage import util





parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_free_dir', dest='src_free_dir', default='./Data/poly/train', help='dir of data') # load the data
parser.add_argument('--src_noise_dir', dest='src_noise_dir', default='./valid/noise256', help='dir of data') # load the data
args = parser.parse_args()


def ChangeSize():    # 改变图片的尺寸
    filepaths = glob.glob(args.src_free_dir + '/*')
    print("number of training data %d" % len(filepaths))

    for i in range(len(filepaths)):
        img = Image.open(filepaths[i]).convert('RGB')
        # img = Image.open(filepaths[i])
        # img = np.array(img)
        img = img.resize((512, 512), resample=PIL.Image.BICUBIC)
        if not os.path.exists("./3"):
            os.makedirs("./3")
        # img.save(os.path.join("filepath", '%s' % (i)),img)
        save_img(os.path.join("./3", '%s' % (i)), img)
def save_img(filepath, img):
    img.save(filepath, 'jpg')




def save_npy():    # 将图片转化为数组形式保存
    files = os.listdir("./valid/ori/")
    for file in files:
        img = Image.open("./valid/ori/"+file).convert('L')
        img = img.resize((1024, 256))
        img = np.array(img, dtype="float")
        img = img.astype(np.float32)/255  # normailize the data
        img_s = np.reshape(np.array(img, dtype="uint8"), (img.size[0], img.size[1], 1))  # extend one dimension
        if not os.path.exists("./valid/"):
            os.makedirs("./valid/")
        np.save(os.path.join("./valid/", "ori_npy"), img_s)  # save inputs as npy
        print("size of inputs tensor = " + str(img_s.shape))


def save_images(filepath, ground_truth, noisy_image=None, clean_image=None):
    # assert the pixel value range is 0-255
    ground_truth = np.squeeze(ground_truth)  #delete single dim in the array
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any(): # judge whether the clean_image  is all  empty
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1) #combine three img(array) in the horizontal direction
    im = Image.fromarray(cat_image.astype('uint8')).convert('L') #convert the array into img
    im.save(filepath, 'png')


def add_noise():   # CT图象加噪
    files = os.listdir("./valid2/ori256")
    for file in files:
        img = Image.open("./valid2/ori256/"+file)
        img = np.array(img)
        img = img.astype(np.float32)/255
        noise_img = 1.66*util.random_noise(img*(0.6), mode='poisson', seed=None)   # add the poisson noise
        # noise_img = util.random_noise(img, mode='gaussian', seed=None, var=(random.randint(0, 55) / 255.0) ** 2)      # add the guass noise
        noisyimage = np.clip(255 * noise_img, 0, 255).astype('uint8')
        if not os.path.exists("./valid2/noise0.6/"):
            os.makedirs("./valid2/noise0.6/")
        save_images(os.path.join("./valid2/noise0.6/", '%s' % (file)), noisyimage)











if __name__ == '__main__':
    ChangeSize()
    # add_noise()
    #save_npy()






