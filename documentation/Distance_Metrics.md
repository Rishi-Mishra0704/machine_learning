# Types of Distance Metrics in Machine Learning

In machine learning, **distance metrics** are used to measure the similarity or dissimilarity between data points. Here are some common distance metrics:

---

## 1. **Euclidean Distance**

**Formula:**

\[ d(\mathbf{X}, \mathbf{Y}) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2} \]

**Example:**

For two points:  
- \( \mathbf{X} = (1, 2) \)  
- \( \mathbf{Y} = (4, 6) \)

Euclidean Distance:  
\[ d(\mathbf{X}, \mathbf{Y}) = \sqrt{(1-4)^2 + (2-6)^2} = \sqrt{9 + 16} = \sqrt{25} = 5 \]

---

## 2. **Manhattan Distance**

**Formula:**

\[ d(\mathbf{X}, \mathbf{Y}) = \sum_{i=1}^n |x_i - y_i| \]

**Example:**

For two points:  
- \( \mathbf{X} = (1, 2) \)  
- \( \mathbf{Y} = (4, 6) \)

Manhattan Distance:  
\[ d(\mathbf{X}, \mathbf{Y}) = |1-4| + |2-6| = 3 + 4 = 7 \]

---

## 3. **Cosine Similarity**

**Formula:**

\[ \text{Cosine Similarity}(\mathbf{X}, \mathbf{Y}) = \frac{\mathbf{X} \cdot \mathbf{Y}}{\|\mathbf{X}\| \|\mathbf{Y}\|} \]

**Example:**

For two vectors:  
- \( \mathbf{X} = [1, 2] \)  
- \( \mathbf{Y} = [4, 6] \)

Cosine Similarity:  
\[ \frac{1 \times 4 + 2 \times 6}{\sqrt{1^2 + 2^2} \times \sqrt{4^2 + 6^2}} = \frac{4 + 12}{\sqrt{5} \times \sqrt{52}} = \frac{16}{\sqrt{260}} \approx 0.996 \]

---

## 4. **Hamming Distance**

**Formula:**

\[ d(\mathbf{X}, \mathbf{Y}) = \sum_{i=1}^n \mathbb{1}(x_i \neq y_i) \]

**Example:**

For two binary strings:  
- \( \mathbf{X} = 101010 \)  
- \( \mathbf{Y} = 100110 \)

Hamming Distance:  
\[ d(\mathbf{X}, \mathbf{Y}) = 2 \quad (\text{positions 3 and 5 are different}) \]

---

## 5. **Jaccard Index**

**Formula:**

\[ \text{Jaccard Index}(\mathbf{X}, \mathbf{Y}) = \frac{|X \cap Y|}{|X \cup Y|} \]

**Example:**

For two sets:  
- \( X = \{1, 2, 3\} \)  
- \( Y = \{2, 3, 4\} \)

Jaccard Index:  
\[ \frac{|\{2, 3\}|}{|\{1, 2, 3, 4\}|} = \frac{2}{4} = 0.5 \]

---

## 6. **Mahalanobis Distance**

**Formula:**

\[ d(\mathbf{X}, \mathbf{Y}) = \sqrt{(\mathbf{X} - \mathbf{Y})^T \mathbf{S}^{-1} (\mathbf{X} - \mathbf{Y})} \]

**Example:**

This metric requires the covariance matrix \( \mathbf{S} \), which depends on the distribution of data. It’s more complex and used for multivariate data.

---

## Choosing the Right Metric

- **Euclidean Distance** is great for continuous data when you want the straight-line distance between points.
- **Manhattan Distance** is useful when you're dealing with grid-like data.
- **Cosine Similarity** works well when you're dealing with text data or vectors where the orientation matters but not the magnitude.
- **Hamming Distance** is designed for binary or categorical data.
- **Jaccard Index** is often used for comparing sets.
- **Mahalanobis Distance** is helpful when you have correlated variables and need to consider their covariance.

---

### Key Considerations

- **Normalization**: Ensure your data is scaled appropriately, especially with Euclidean distance, which is sensitive to large differences in scale.
- **Dimensionality**: For high-dimensional data, distance metrics like **Mahalanobis** or **Cosine Similarity** might work better.
- **Computational Complexity**: Some distance metrics, like **Mahalanobis**, can be computationally expensive for large datasets.

---
