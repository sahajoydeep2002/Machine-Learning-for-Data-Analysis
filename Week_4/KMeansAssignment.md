## Find teen market segments using k-means clustering##

In this assignment, we are assigned to run k-means clustering analysis. For this analysis, we use a dataset which represents a random sample of 30,000 U.S. high school students who had profiles on a well-known SNS in 2006. The data was sampled from senior, junior, sophomore and freshman classes with graduation years from 2006 through 2009. The full dataset can be downloaded [here](https://github.com/debinqiu/ML_Course/blob/master/Week_4/snsdata.csv). The oirginal dataset contains 40 features, but we will use only 39 features in the clustering analysis by dropping 'gradyear' variable. Also, we will transform categorical variable 'gender' into numberic values. Other features are all numeric. All above steps are completed in the following Python code. 
```python
>>> import pandas as pd
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from sklearn import preprocessing
>>> from sklearn.cluster import KMeans

# import csv data
>>> snsdata = pd.read_csv("snsdata.csv")
# drop the missing data
>>> snsdata_clean = snsdata.dropna()
>>> snsdata_clean.dtypes

# recode the categorical variable 'gender' to numeric variable
>>> snsdata_clean['gender'] = preprocessing.LabelEncoder().fit_transform(snsdata_clean['gender'])
>>> del snsdata_clean['gradyear'] # drop useless variable 'gradyear'
```

Before runing a k-means clustering analysis, we first need to standardize each feature so that each of them has mean 0 and standard deviation 1 because euclidean distance is very sensitive to the scales of variables. This can be done as follows.
```python
# standardize each variable so that mean = 0 and std = 1
>>> for name in snsdata_clean.columns:
        snsdata_clean[name] = preprocessing.scale(snsdata_clean[name]).astype('float64')
```

Now we can perform k-means clustering analysis on those 39 standardized features. For simplicity, we only examine the number of clusters from 1 to 10, although they can be up to 39 clusters. In effect, it is rather safe to only check 1-10 clusters for this dataset. The Python code to run k-means clustering is as follows.
```python
# perform k-means clustering for each k between 1 - 20   
>>> from scipy.spatial.distance import cdist

>>> clusters = range(1,20)
>>> meandist = []

>>> for k in clusters:
        model = KMeans(n_clusters = k, random_state = 123)
        model.fit(snsdata_clean)
        clusassign = model.predict(snsdata_clean)
        meandist.append(sum(np.min(cdist(snsdata_clean,model.cluster_centers_,'euclidean'), axis = 1))/snsdata_clean.shape[0])
```
Since we have claculated the mean distance for each cluster, we can now plot the Elbow graph by checking the average distance versus the number of cluster which is shown as follows. The Elbow graph suggests us to choose the number of cluster as 3 or 4 because there is a small jump when the number of cluster is 5. 
![figure_1](https://cloud.githubusercontent.com/assets/16762941/13307691/2215d220-db3b-11e5-8ef1-342db76203aa.png)
```python
# plot the elbow graph    
>>> plt.plot(clusters, meandist)
>>> plt.xlabel('Number of clusters')
>>> plt.ylabel('Average distance')
>>> plt.title('Selecting k with the Elbow Method')
>>> plt.show()
```

To visualize the separation of each cluster, canonical discriminant analyses is used to reduce the 39 clustering variables down a few variables that accounted for most of the variance in the clustering variables. We first examine results when the number of clusters k = 3. A scatterplot of the first two canonical variables by cluster indicated that **Cluster 1** and **Cluster 2** are rather packed leading to low within cluster variance, but **Cluster 3** is rather spreadout resulting in high within cluster variance. 
```python
# interpret  cluster solution
>>> from sklearn.decomposition import PCA

>>> def kmeans(k):
        model = KMeans(n_clusters = k,random_state = 123)
        model.fit(snsdata_clean)
        # plot clusters
        pca_2 = PCA(2)
        plot_columns = pca_2.fit_transform(snsdata_clean)
        cols = ['r','g','b','y','m','c']
        legentry = []
        legkey = []
        for i in range(k):
            rowindex = model.labels_ == i
            plot_ = plt.scatter(plot_columns[rowindex,0],plot_columns[rowindex,1], c = cols[i],)
            exec('sc' + str(i) + " = plot_")
            legentry.append(eval('sc' + str(i)))
            legkey.append('Cluster ' + str(i + 1))
        plt.legend(tuple(legentry),tuple(legkey),loc = 'lower right')
        plt.xlabel('Canonical variable 1')
        plt.ylabel('Canonical variable 2')
        plt.title('Scatterplot of Canonical Variables for ' + str(k) + ' Clusters')
        plt.show() 
# try k = 3 
>>> kmeans(3)
```
![figure_1-3clusters](https://cloud.githubusercontent.com/assets/16762941/13307696/26039174-db3b-11e5-91db-2ab48cc1b774.png)
Secondly, we examine results when k = 4. We can see that **Cluster 2** and **Cluster 4** are packed but **Cluster 3** is rather spreadout. Also, **Cluster 1** and **Cluster 4** are overlap too much indicating that the results of k = 3 is superior to those of k = 4. 
```python
# try k = 4 
>>> kmeans(4)
```
![figure_1-4clusters](https://cloud.githubusercontent.com/assets/16762941/13307699/27dde422-db3b-11e5-9784-d3fa5b146771.png)
Therefore, we select k = 3 and calculate the size and centroid means of each cluster as follows. We can see that **Cluster 3** has the largest number of observations, i.e., 69.93% but **Cluster 1** has only 10.71% of observations. 
```python
>>> model3 = KMeans(n_clusters = 3).fit(snsdata_clean)
>>> snsdata_clean.reset_index(level = 0, inplace = True)
>>> newclus = pd.DataFrame.from_dict(dict(zip(list(snsdata_clean['index']),list(model3.labels_))),orient = 'index')
>>> newclus.columns = ['cluster']

>>> newclus.reset_index(level = 0, inplace = True)
>>> snsdata_merge = pd.merge(snsdata_clean,newclus, on = 'index')
>>> snsdata_merge.cluster.value_counts()
Out[41]: 
2    16789
1     4647
0     2569
Name: cluster, dtype: int64
```
We finally look at the centroid means of each cluster for each features. 
```python
>>> clustergrp = snsdata_merge.groupby('cluster').mean()
>>> print ("Clustering variable means by cluster")
>>> print(clustergrp)
Clustering variable means by cluster
                index    gender       age   friends  basketball  football  \
cluster                                                                     
0        16512.845854 -0.326180 -0.050493  0.251594    0.493595  0.461757   
1        13816.222940  1.961558  0.313298 -0.169918    0.033028  0.254605   
2        14916.706177 -0.493025 -0.078991  0.008533   -0.084670 -0.141128   

           soccer  softball  volleyball  swimming    ...       blonde  \
cluster                                              ...                
0        0.265626  0.245008    0.269518  0.316393    ...     0.196089   
1       -0.025290 -0.208723   -0.167954 -0.116565    ...    -0.042956   
2       -0.033645  0.020282    0.005247 -0.016150    ...    -0.018115   

             mall  shopping   clothes  hollister  abercrombie       die  \
cluster                                                                   
0        0.891758  0.753866  1.145443   0.845561     0.864245  0.728301   
1       -0.213791 -0.434151 -0.208827  -0.127641    -0.123796 -0.067070   
2       -0.077279  0.004814 -0.117471  -0.094055    -0.097979 -0.092878   

            death     drunk     drugs  
cluster                                
0        0.540961  0.722546  0.922997  
1       -0.078481 -0.084363 -0.097185  
2       -0.061054 -0.087211 -0.114334  

[3 rows x 40 columns]
```
We only interpret the results shown in the outputs above because it is difficult to display all centroid means for all 39 features. The means on the clustering variables showed that, compared to the other clusters, **gender, age** in cluster 2 has levels on the clustering variables. All the other variables shown above in cluster 1 have high levels on the clustering variables compared with other clusters.    


Note: If you want to get the entire Python code for this assignment, please click [here](https://github.com/debinqiu/ML_Course/blob/master/Week_4/KMeansClustering.py).
