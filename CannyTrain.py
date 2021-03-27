import numpy as np
from skimage.viewer import ImageViewer
from scipy import ndimage as ndi
import os
from skimage import io, feature, color, exposure
from sklearn import svm
from joblib import dump

imgSize = 13

# read all training data from the npy files
print('Reading all npy files...', end='\r')
with open('npy/train_img.npy', 'rb') as f:
    train_img = np.load(f)
with open('npy/train_labels.npy', 'rb') as f:
    train_labels = np.load(f)
print('Done reading all npy files.')

# Put train images through filters and save them in train_img_edges.
train_img_edges = np.empty((len(train_img), imgSize, imgSize ))
for i in range(len(train_img)):
    img = color.rgb2gray(train_img[i])
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    edges = feature.canny(img_rescale, sigma=0)
    train_img_edges[i] = edges
    percentage = round(i/(len(train_img)-1)*100, 2)
    print('Loading all traininging images: [{}{}{}]  {}'.format( ('=' * int(percentage//10) ), ('>' if percentage < 100 else ''), ('.' * int(10-(((percentage)//10))-1)), percentage ), end='\r')
print('\nDone loading all traininging images.')
train_img_edges = np.reshape(train_img_edges, (len(train_img_edges), (len(train_img_edges[0])*len(train_img_edges[0][0]))))

# Train the SMV.
print('Training SVM.', end='\r')
clf = svm.SVC(gamma=0.001, C=100)
clf.fit(train_img_edges, train_labels)
print('Done Training SVM.')

# Save traind CLF to file.
dump(clf, 'npy/clfCanny.joblib')
