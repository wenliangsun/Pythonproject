import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print(os.path.join(r"/media/sunwl/sunwl/datum/roadDetect_project/Massachusetts", 'data'))

# img = plt.imread(r'./dataset/img/23429080_15.tiff')
# print(img.shape)
# plt.imshow(img)
# plt.show()

img = Image.open(r'./dataset/img/23429080_15.tiff')
label = Image.open(r'./dataset/img/23429080_15.tif').convert("L")

x, y = 0, 100
dim_data = 100
new_img = label.rotate(50)
img_arr = np.asarray(new_img,dtype="float32")
data_temp = img_arr[y: y + dim_data, x: x + dim_data]
print(data_temp)
# print(new_img.size)
# new_img.show()
# print(img_arr.shape[2])
# label.show()

# plt.imshow(data_temp,cmap='gray')
# plt.show()
# contains_class = not data_temp.max() == 0
# print(contains_class)

# arr = np.asarray(data_temp, dtype='float32') / 255
# arr = np.rollaxis(arr, 2, 0)
# arr = arr.reshape(3 * arr.shape[1] * arr.shape[2])
# print(arr.shape)
# print(arr)
# train_data = np.zeros((512, 64, 64, 3))
# train_label = np.zeros((512, 16, 16))
# batch_size = 8
# nr_elements = 512
#
# batches = [[train_data[x:x + batch_size], train_label[x:x + batch_size]]
#            for x in range(0, nr_elements, batch_size)]
# print(len(batches))
# print(type(batches[0]))
# print(len(batches[0][0]))
# print(batches[0][0].shape)
#
#




















