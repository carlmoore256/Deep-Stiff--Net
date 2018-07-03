from sklearn.decomposition import PCA
import numpy as np
from sklearn.manifold import MDS

import matplotlib 
#matplotlib.use('agg')
import matplotlib.pyplot as plt


## visulaizing the embedding using PCA, MDS, t-SNE
anchor=np.load('anchor_array.npy')
positive= np.load('positive_array.npy')
negative= np.load('negative_array.npy')

pca = PCA(n_components=2)
mds= MDS(n_components=2)
# a=pca.fit_transform(anchor)
# #print(pca.explained_variance_ratio_)  
# p=pca.fit_transform(positive)
# #print(pca.explained_variance_ratio_)  
# n=pca.fit_transform(negative)
# #print(a[:,0])
# #print(pca.explained_variance_ratio_)  
# print(a.shape,p.shape,n.shape)


a=mds.fit_transform(anchor)
#print(pca.explained_variance_ratio_)  
p=mds.fit_transform(positive)
# #print(pca.explained_variance_ratio_)  
n=mds.fit_transform(negative)
#print(a[:,0])
#print(pca.explained_variance_ratio_)  
print(a.shape,p.shape,n.shape)

a_x= a[:,0]
a_y= a[:,1]
#print(a_x.shape)
plt.scatter(a_x,a_y)
plt.show()
