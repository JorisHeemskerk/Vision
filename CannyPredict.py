import numpy as np
from skimage.viewer import ImageViewer
from scipy import ndimage as ndi
import os
from skimage import io, feature, color, exposure
from sklearn import svm
from joblib import load

clf = load('npy/clf.joblib')

# read all testing data from the npy files
print('Reading all npy files...', end='\r')
with open('npy/test_img.npy', 'rb') as f:
    test_img = np.load(f)
with open('npy/test_labels.npy', 'rb') as f:
    test_labels = np.load(f)
print('Done reading all npy files.')

# Put test images through filters and save them in test_img_edges.
test_img_edges = np.empty((len(test_img), 100, 100))
for i in range(len(test_img)):
    img = color.rgb2gray(test_img[i])
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    edges = feature.canny(img_rescale, sigma=5)
    test_img_edges[i] = edges
    percentage = round(i/(len(test_img)-1)*100, 2)
    print('Loading all testing images: [{}{}{}]  {}'.format( ('=' * int(percentage//10) ), ('>' if percentage < 100 else ''), ('.' * int(10-(((percentage)//10))-1)), percentage ), end='\r')
print('\nDone loading all testing images.')
test_img_edges = np.reshape(test_img_edges, (len(test_img_edges), (len(test_img_edges[0])*len(test_img_edges[0][0]))))

# check the accuracy
correct = 0
for i in range(len(test_img_edges)):
    prediction = clf.predict(test_img_edges[i:i+1])
    percentage = round(i/(len(test_img_edges)-1)*100, 2)
    print('Predicting all testing images: [{}{}{}]  {}'.format( ('=' * int(percentage//10) ), ('>' if percentage < 100 else ''), ('.' * int(10-(((percentage)//10))-1)), percentage ), end='\r')
    if prediction[0] == test_labels[i]:
        correct += 1
print('\nDone predicting all testing images.')

# print accuracy
print("\n{} out of {} images are correctly predicted".format(correct, len(test_img_edges))) 
print("This is {}%\n".format(correct/len(test_img_edges)*100))
