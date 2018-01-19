import numpy as np
from DeepLearning.python import list_save

def loadMetaMat(metaMatPath):
    from scipy.io import loadmat
    meta = loadmat(metaMatPath)
    metaArray = []
    label = []
    for item in meta['synsets']:
        label.append(item[0][1][0] + '    ' + item[0][2][0])
        metaArray.append({
            'ILSVRC2012_ID': item[0][0][0][0],
            'WNID': item[0][1][0],
            'words': item[0][2][0],
            'gloss': item[0][3][0],
            'num_children': item[0][4][0][0],
            'children': item[0][5][0],
            'wordnet_height': item[0][6][0][0],
            'num_train_images': item[0][7][0][0]
            })
    list_save(label[:1000], "ImageNet_label.txt")
    return metaArray

metaArray = loadMetaMat('meta.mat')
print(len(metaArray))
print(metaArray[999])
a = np.loadtxt("ILSVRC2012_validation_ground_truth.txt")
print(len(a))
print(a[0:50])
import tarfile
import os

