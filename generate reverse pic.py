import numpy as np
from PIL import Image

a=np.array(Image.open("./Data/baby2/GT_TE/Blast_PCRM_2 x 2AB d5 left TE_Mask.png").convert('L'))
b=255-a
new=Image.fromarray(b.astype('uint8'))
new.save("./Data/reverse.png")
