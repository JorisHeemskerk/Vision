import scipy.io
import numpy as np
import os
import skimage
from skimage.viewer import ImageViewer
from skimage.transform import resize

mat = scipy.io.loadmat('Lists/English/Img/lists.mat')
all_names = mat['list'][0][0][0]
all_labels = mat['list'][0][0][2]
imgSize = 13
all_img = np.empty((len(all_names), imgSize, imgSize, 3))

# make and save all images into all_img.npy
for i in range(len(all_names)):
    # TODO: miss alleen good img pakken
    filename = os.path.join('Data/Img/' + all_names[i] + '.png')
    all_img[i] =  resize(skimage.io.imread(filename), (imgSize, imgSize, 3))
    percentage = round(i/(len(all_names)-1)*100, 2)
    print('Loading all images: [{}{}{}]  {}'.format( ('=' * int(percentage//10) ), ('>' if percentage < 100 else ''), ('.' * int(10-(((percentage)//10))-1)), percentage ), end='\r')
print('')


# Derive and save all training images and labels into train_img.npy and train_labels.npy
train_indexes = mat['list'][0][0][8]
train_img = np.empty((len(train_indexes) * len(train_indexes[0]), imgSize, imgSize, 3))
train_labels = np.empty(len(train_indexes) * len(train_indexes[0]))
iterator = 0

for i in range(len(train_indexes)):
    for j in range(len(train_indexes[i])):
        if train_indexes[i][j] != 0:
            train_img[iterator] = all_img[train_indexes[i][j]-1]
            train_labels[iterator] = all_labels[train_indexes[i][j]-1]
            iterator += 1
            percentage = round(((i*len(train_indexes[i]))+j)/((len(train_indexes)*len(train_indexes[i]))-1)*100, 2)
            print('Loading all training images: [{}{}{}]  {}'.format( ('=' * int(percentage//10) ), ('>' if percentage < 100 else ''), ('.' * int(10-(((percentage)//10))-1)), percentage ), end='\r')
train_img = train_img[:iterator]
train_labels = train_labels[:iterator]
print('\nSaving training images to file.')
with open('npy/train_img.npy', 'wb') as f:
    np.save(f, train_img)
with open('npy/train_labels.npy', 'wb') as f:
    np.save(f, train_labels)


# Derive and save all testing images and labels into test_img.npy and test_labels.npy
test_indexes = mat['list'][0][0][6]
test_img = np.empty((len(test_indexes) * len(test_indexes[0]), imgSize, imgSize, 3))
test_labels = np.empty(len(test_indexes) * len(test_indexes[0]))
iterator = 0
for i in range(len(test_indexes)):
    for j in range(len(test_indexes[i])):
        if test_indexes[i][j] != 0:
            test_img[iterator] = all_img[test_indexes[i][j]-1]
            test_labels[iterator] = all_labels[test_indexes[i][j]-1]
            iterator += 1
            percentage = round(((i*len(test_indexes[i]))+j)/((len(test_indexes)*len(test_indexes[i]))-1)*100, 2)
            print('Loading all testing images: [{}{}{}]  {}'.format( ('=' * int(percentage//10) ), ('>' if percentage < 100 else ''), ('.' * int(10-(((percentage)//10))-1)), percentage ), end='\r')
test_img = test_img[:iterator]
test_labels = test_labels[:iterator]
print('\nSaving testing images to file.')
with open('npy/test_img.npy', 'wb') as f:
    np.save(f, test_img)
with open('npy/test_labels.npy', 'wb') as f:
    np.save(f, test_labels)