# K-Means Clustering on Iris Dataset (Elbow Method)

## Objective

Use the Iris flower dataset from `sklearn` to:

- Perform K-Means clustering
- Use only **petal length** and **petal width** features
- Apply preprocessing (scaling)
- Use the Elbow Method to determine the optimal number of clusters
- Visualize the clusters

---

## Dataset

Dataset used: `sklearn.datasets.load_iris`

- Total samples: 150
- Selected features:
  - Petal Length (cm)
  - Petal Width (cm)
- Dropped:
  - Sepal Length
  - Sepal Width

---

## Preprocessing

MinMax Scaling was applied using:

`MinMaxScaler()`

Reason:
- K-Means is distance-based.
- Scaling ensures both features contribute equally.

---

## Elbow Method

- Calculated SSE (Sum of Squared Errors) for K values from 1 to 10.
- Plotted K vs SSE.
- Observed elbow point at:

👉 **K = 3**

This indicates optimal number of clusters is 3.

---

## Clustering

- Applied KMeans with `n_clusters=3`
- Assigned cluster labels to dataset
- Visualized clusters using scatter plot

---

## Results

- Optimal K found: **3**
- Clusters visually separated based on petal features
- Scaling improved clustering consistency

---

## Conclusion

Using petal length and width features, K-Means successfully grouped the Iris dataset into 3 clusters.

The Elbow Method helped determine the optimal value of K.

Feature scaling is important in distance-based algorithms like K-Means.