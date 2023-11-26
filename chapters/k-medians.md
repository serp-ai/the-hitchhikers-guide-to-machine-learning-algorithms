# k-Medians

The **k-Medians** algorithm is a **clustering** technique used in **unsupervised learning**. It is a partitioning method of cluster analysis that aims to partition _n_ observations into _k_ clusters based on their median values. Unlike k-Means, which uses the mean value of observations, k-Medians uses the median value of observations to define the center of a cluster. This algorithm is useful in situations where the mean value is not a good representation of the cluster center, such as in the presence of outliers or when dealing with non-numerical data.

{% embed url="https://youtu.be/EZQNfd6aOeQ?si=e91dVnNlcI3Jq8cj" %}

## k-Medians: Introduction

| Domains          | Learning Methods | Type       |
| ---------------- | ---------------- | ---------- |
| Machine Learning | Unsupervised     | Clustering |

The k-Medians algorithm is a commonly used **clustering** technique in **unsupervised learning**. It is a partitioning method of cluster analysis which attempts to partition n observations into k clusters. The goal of this algorithm is to minimize the sum of distances between each observation and their respective median, as opposed to the average in the case of k-Means. Unlike k-Means, k-Medians is more robust to outliers and can handle non- numerical data.

The algorithm works by first randomly selecting k observations as the initial medians and then assigning each observation to the closest median. The median of each cluster is then updated by calculating the median of all the observations assigned to it. This process is iteratively repeated until convergence, which is defined as no further changes in assignment of observations to clusters.

K-Medians is widely used in a variety of fields, including image segmentation, bioinformatics, and social network analysis. Its ability to handle non- numerical data and outliers make it a popular choice for real-world data analysis.

Like many clustering algorithms, k-Medians requires the user to specify the number of clusters, k, before running the algorithm. Choosing the optimal value of k is often determined empirically or by using a validation metric such as the silhouette coefficient.

## k-Medians: Use Cases & Examples

The k-Medians algorithm is a type of clustering method used in unsupervised learning. It is a partitioning method of cluster analysis that attempts to partition n observations into k clusters based on their median values.

One use case of the k-Medians algorithm is in market segmentation. Companies can use this algorithm to segment their customers based on their buying behaviors and preferences. By clustering customers together based on their median spending habits, companies can tailor their marketing strategies and product offerings to each group.

Another example of the k-Medians algorithm is in image compression. By clustering the median colors of pixels together, the algorithm can reduce the number of colors used in an image without significantly affecting its quality. This can result in smaller file sizes and faster load times.

The k-Medians algorithm can also be used in anomaly detection. By clustering data points together based on their median values, any data points that fall outside of their respective clusters can be identified as anomalies. This can be useful in detecting fraudulent transactions or identifying errors in data.

## Getting Started

The k-Medians algorithm is a partitioning method of cluster analysis which attempts to partition n observations into k clusters. It is a type of clustering algorithm and falls under unsupervised learning.

To get started with implementing the k-Medians algorithm, we can use Python and common machine learning libraries like NumPy, PyTorch, and scikit-learn.

```
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin

class KMedians:
    def __init__(self, k=2, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        
    def fit(self, X):
        m, n = X.shape
        
        # randomly initialize centroids
        self.centroids = np.zeros((self.k, n))
        for i in range(self.k):
            self.centroids[i] = X[np.random.choice(range(m))]
        
        for _ in range(self.max_iters):
            # assign each data point to closest centroid
            clusters = [[] for _ in range(self.k)]
            for x in X:
                distances = pairwise_distances_argmin(x.reshape(1,-1), self.centroids)
                clusters[distances[0]].append(x)
            
            # update centroids to median of cluster
            for i in range(self.k):
                if clusters[i]:
                    self.centroids[i] = np.median(clusters[i], axis=0)

```

## FAQs

### What is k-Medians?

k-Medians is a partitioning method of cluster analysis which attempts to partition n observations into k clusters. It is similar to k-Means algorithm, but instead of calculating the mean, it calculates the median for each cluster.

### What is the type of algorithm k-Medians?

k-Medians is a clustering algorithm.

### What is the learning method used in k-Medians?

The learning method used in k-Medians is unsupervised learning.

### What is the difference between k-Means and k-Medians?

The main difference between k-Means and k-Medians is the way they calculate the center of each cluster. k-Means uses the mean, while k-Medians uses the median. This makes k-Medians more robust to outliers, but it also makes it slower than k-Means.

### What are the applications of k-Medians?

k-Medians can be used in a variety of applications such as market segmentation, image segmentation, and data compression.

## k-Medians: ELI5

Imagine you have a bunch of marbles that you need to sort into different groups based on their color. You don't know how many different colors there are, but you have a bunch of empty jars ready to collect them. The k-Medians algorithm works kind of like this, but with numbers instead of marbles.

k-Medians is a type of clustering algorithm that tries to group similar data points together. It works by randomly choosing k number of centroids, or jar locations, and assigning each data point to the closest centroid. Then, it finds the median value of each group and assigns that as the new centroid. The algorithm keeps doing this until the centroids stop moving.

So, let's say you have a bunch of numbers that you want to group together in a meaningful way. The k-Medians algorithm will help you find groups based on their median values. It uses unsupervised learning, so you don't need to tell it what the groups are beforehand.

Think of it like a game of "guess the average" - the algorithm keeps guessing and refining until it gets as close as possible to the true average.

Ultimately, k-Medians helps you make sense of large amounts of data by grouping together similar values in a way that's easy to interpret and understand. [K Medians](https://serp.ai/k-medians/)
