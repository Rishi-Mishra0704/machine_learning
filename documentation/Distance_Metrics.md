# Types of Distance Metrics in Machine Learning

In machine learning, distance metrics are used to measure the similarity or dissimilarity between data points. Here are some common distance metrics:

---

## 1. **Euclidean Distance**

Euclidean distance measures the straight-line distance between two points in space.

### Formula:
`d(X, Y) = sqrt((x1 - y1)^2 + (x2 - y2)^2)`

### Example:

For two points:
- Point X = (1, 2)
- Point Y = (4, 6)

**Euclidean Distance:**

`d(X, Y) = sqrt((1 - 4)^2 + (2 - 6)^2) = sqrt(9 + 16) = sqrt(25) = 5`

---

## 2. **Manhattan Distance**

Manhattan distance (or "city block distance") calculates the total sum of absolute differences between the coordinates.

### Formula:
`d(X, Y) = |x1 - y1| + |x2 - y2|`

### Example:

For two points:
- Point X = (1, 2)
- Point Y = (4, 6)

**Manhattan Distance:**

`d(X, Y) = |1 - 4| + |2 - 6| = 3 + 4 = 7`

---

## 3. **Cosine Similarity**

Cosine similarity measures the cosine of the angle between two vectors. It is often used for comparing text data.

### Formula:
`Cosine Similarity(X, Y) = (X . Y) / (||X|| * ||Y||)`

### Example:

For two vectors:
- X = [1, 2]
- Y = [4, 6]

**Cosine Similarity:**

`(X . Y) = 1*4 + 2*6 = 4 + 12 = 16`

`||X|| = sqrt(1^2 + 2^2) = sqrt(5)`

`||Y|| = sqrt(4^2 + 6^2) = sqrt(52)`

`Cosine Similarity(X, Y) = 16 / (sqrt(5) * sqrt(52)) ≈ 0.996`

---

## 4. **Hamming Distance**

Hamming distance is used to measure the difference between two binary strings by counting the number of differing positions.

### Formula:
`d(X, Y) = count of differing positions`

### Example:

For two binary strings:
- X = 101010
- Y = 100110

**Hamming Distance:**

`d(X, Y) = 2` (positions 3 and 5 differ)

---

## 5. **Jaccard Index**

The Jaccard Index measures the similarity between two sets by dividing the size of their intersection by the size of their union.

### Formula:
`Jaccard Index(X, Y) = |X ∩ Y| / |X ∪ Y|`

### Example:

For two sets:
- X = {1, 2, 3}
- Y = {2, 3, 4}

**Jaccard Index:**

`Intersection(X, Y) = {2, 3}`  
`Union(X, Y) = {1, 2, 3, 4}`

`Jaccard Index(X, Y) = 2 / 4 = 0.5`

---

## 6. **Mahalanobis Distance**

Mahalanobis distance takes into account the covariance of the data and is useful for data with correlations.

### Formula:
`d(X, Y) = sqrt((X - Y)ᵀ * S⁻¹ * (X - Y))`

**Note:** This metric is more complex and requires the covariance matrix (S).

---

## Choosing the Right Metric

- **Euclidean Distance**: Good for continuous data when you want the straight-line distance.
- **Manhattan Distance**: Useful for grid-like data.
- **Cosine Similarity**: Best for text data or when comparing vectors where the magnitude doesn't matter.
- **Hamming Distance**: For binary or categorical data.
- **Jaccard Index**: Used for comparing sets.
- **Mahalanobis Distance**: Useful when you need to account for correlations in multivariate data.

---

### Key Considerations

- **Normalization**: Scale your data appropriately before using Euclidean or Manhattan distances.
- **Dimensionality**: For high-dimensional data, consider Cosine Similarity or Mahalanobis Distance.
- **Computational Complexity**: Some metrics, like Mahalanobis, are computationally expensive.

---
