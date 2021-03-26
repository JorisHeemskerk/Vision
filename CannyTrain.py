import numpy as np
from skimage.viewer import ImageViewer
# import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import os
from skimage import io, feature, color, exposure
from sklearn import svm
from joblib import dump

# read all training data from the npy files
print('Reading all npy files...', end='\r')
with open('npy/train_img.npy', 'rb') as f:
    train_img = np.load(f)
with open('npy/train_labels.npy', 'rb') as f:
    train_labels = np.load(f)
print('Done reading all npy files.')

# Put train images through filters and save them in train_img_edges.
train_img_edges = np.empty((len(train_img), 100, 100 ))
for i in range(len(train_img)):
    img = color.rgb2gray(train_img[i])
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    edges = feature.canny(img_rescale, sigma=5)
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
dump(clf, 'npy/clf.joblib')









# img = train_img[420]
# img = color.rgb2gray(img)
# p2, p98 = np.percentile(img, (2, 98))
# img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

# viewer = ImageViewer(img_rescale)
# viewer.show()
# normalImg = color.rgb2gray(img)
# normalEdges1 = feature.canny(img,sigma=4)
# normalEdges2 = feature.canny(img, sigma=5)
# filterImg = color.rgb2gray(img_rescale)
# filterEdges1 = feature.canny(img_rescale, sigma=4)
# filterEdges2 = feature.canny(img_rescale, sigma=5)


# # display results
# fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(nrows=2, ncols=3, figsize=(16, 6),sharex=True, sharey=True)

# ax0.imshow(normalImg, cmap=plt.cm.gray)
# ax0.axis('off')
# ax0.set_title(r'noFilter', fontsize=20)

# ax1.imshow(normalEdges1, cmap=plt.cm.gray)
# ax1.axis('off')
# ax1.set_title(r'Canny filter, $\sigma=4$', fontsize=20)

# ax2.imshow(normalEdges2, cmap=plt.cm.gray)
# ax2.axis('off')
# ax2.set_title(r'Canny filter, $\sigma=5$', fontsize=20)

# ax3.imshow(filterImg, cmap=plt.cm.gray)
# ax3.axis('off')
# ax3.set_title(r'Contrast Stretching', fontsize=20)

# ax4.imshow(filterEdges1, cmap=plt.cm.gray)
# ax4.axis('off')
# ax4.set_title(r'Canny filter, $\sigma=4$', fontsize=20)

# ax5.imshow(filterEdges2, cmap=plt.cm.gray)
# ax5.axis('off')
# ax5.set_title(r'Canny filter, $\sigma=5$', fontsize=20)

# fig.tight_layout()

# plt.show()