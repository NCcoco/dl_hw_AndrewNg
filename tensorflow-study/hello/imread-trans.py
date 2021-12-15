import matplotlib.pyplot as plt
import numpy as np
import scipy
from PIL import Image
from scipy import ndimage
import imageio
import os
import skimage.transform as sktrans
print(scipy.__version__)
"""
由于ndimage的imread已不存在
"""
base_path= os.path.abspath(".")
ndimage.imread()
scipy.misc.imresize()
## START CODE HERE ## (PUT YOUR IMAGE NAME)
my_image = base_path + "/hello/thumbs_up.jpg"
## END CODE HERE ##

# We preprocess your image to fit your algorithm.
fname = my_image
image = np.array(imageio.imread(fname))
my_image = np.array(Image.fromarray(image).resize((64,64))).reshape((64*64*3, 1))
print(my_image)
print(my_image.shape)
# my_image_prediction = predict(my_image, parameters)

# plt.imshow(image)
# print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))

def scipy_misc_imresize(arr, size, interp='bilinear', mode=None):
    im = Image.fromarray(arr, mode=mode)
    ts = type(size)
    if np.issubdtype(ts, np.signedinteger):
        percent = size / 100.0
        size = tuple((np.array(im.size)*percent).astype(int))
    elif np.issubdtype(type(size), np.floating):
        size = tuple((np.array(im.size)*size).astype(int))
    else:
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp]) # 调用PIL库中的resize函数
    return np.array(imnew)

kk = scipy_misc_imresize(image, size=(64,64)).reshape((64*64*3,1))
print(kk)
print(kk.shape)
print(kk/255.)


oo = np.array(sktrans.resize(image, output_shape=(64,64,3))).reshape((64*64*3, 1))
print(oo)
print(oo.shape)