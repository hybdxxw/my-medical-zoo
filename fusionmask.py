# from PIL import Image
# import matplotlib.pyplot as plt
#
# # image1 原图
# # image2 分割图
# image1 = Image.open("./Data/skin1/images/0.jpg")
# image2 = Image.open("./Data/skin1/masks/0.png")
#
# image1 = image1.convert('RGB')
# image2 = image2.convert('RGB')
#
# # 两幅图像进行合并时，按公式：blended_img = img1 * (1 – alpha) + img2* alpha 进行
# image = Image.blend(image1, image2, 0.3)
# image.save("test.png")


# import matplotlib.pyplot as plt
# from matplotlib import gridspec
# import numpy as np
# import cv2
# imgfile = './Data/skin1/images/0.jpg'
# pngfile = './Data/skin1/masks/0.png'
#
# img = cv2.imread(imgfile, 1)
# mask = cv2.imread(pngfile, 0)
#
# contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
#
# img = img[:, :, ::-1]
# img[..., 2] = np.where(mask == 1, 255, img[..., 2])
#
# plt.imshow(img)
# plt.show()

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

yuantu = "./Data/DRIVE2/images/01_test.png" # ./Data/DRIVE2/images/01_test.png
mask_path = "./Data/DRIVE2/masks/01_test.png" #./Data/DRIVE2/masks/01_test.png
#
# # 使用opencv叠加图片
# img = cv2.imread(yuantu)
# mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 将彩色mask以二值图像形式读取
# plt.imshow(img)
# plt.show()
# plt.imshow(mask)
# plt.show()
#
# # 将image的相素值和mask像素值相加得到结果
# masked = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
# plt.imshow(masked)
# plt.show()
# cv2.imwrite("mask.jpg", masked)


img1 = cv2.imread(yuantu)
img2 = cv2.imread(mask_path)
# plt.imshow(img1)
# plt.show()
# plt.imshow(img2)
# plt.show()
alpha = 0.5
meta = 1 - alpha
gamma = 0

# image = cv2.addWeighted(img1,alpha,img2,meta,gamma)
image = cv2.add(img1, img2)

cv2.imwrite("./mask1.png",image)
