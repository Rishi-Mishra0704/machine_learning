# K-Means Clustering Project Notes

## Overview
- Objective: Use K-Means Clustering to classify universities into two groups (Private and Public).
- **Key Note:** Although labels are available for evaluation, K-Means Clustering is an unsupervised learning algorithm, typically used when labels are not available.

## Dataset Information
- Total Observations: 777
- Features:
  - **Private**: Indicates if the university is private (Yes) or public (No).
  - **Apps**: Number of applications received.
  - **Accept**: Number of applications accepted.
  - **Enroll**: Number of new students enrolled.
  - **Top10perc**: Percentage of new students from the top 10% of their high school class.
  - **Top25perc**: Percentage of new students from the top 25% of their high school class.
  - **F.Undergrad**: Number of full-time undergraduates.
  - **P.Undergrad**: Number of part-time undergraduates.
  - **Outstate**: Out-of-state tuition.
  - **Room.Board**: Room and board costs.
  - **Books**: Estimated book costs.
  - **Personal**: Estimated personal spending.
  - **PhD**: Percentage of faculty with PhDs.
  - **Terminal**: Percentage of faculty with terminal degrees.
  - **S.F.Ratio**: Student-to-faculty ratio.
  - **perc.alumni**: Percentage of alumni who donate.
  - **Expend**: Instructional expenditure per student.
  - **Grad.Rate**: Graduation rate.

## Exploratory Data Analysis (EDA)
1. **Visualizations:**
   - Scatterplots:
     - `Grad.Rate` vs. `Room.Board` (colored by `Private`).
     - `Outstate` vs. `F.Undergrad` (colored by `Private`).
   - Histograms:
     - `Outstate` distribution by `Private`.
     - `Grad.Rate` distribution by `Private`.
   
2. **Data Cleaning:**
   - Found a university with `Grad.Rate` > 100% (Cazenovia College).
   - Fixed the data by capping `Grad.Rate` at 100%.

## K-Means Clustering
- **Model Setup:**
  - Clustering into 2 clusters.
  - Input: All features except `Private`.

- **Cluster Centers:**
  - Observed the centers to understand clustering behavior.

## Evaluation
1. **Adding Cluster Labels:**
   - Converted `Private` to binary values (1 for "Yes", 0 for "No").
   - Created a new column `Cluster` in the dataset for evaluation.

2. **Performance Metrics:**
   - Confusion Matrix: Compared predicted clusters with actual labels.
   - Classification Report: Evaluated precision, recall, and accuracy.

## Observations
- K-Means successfully grouped universities based on features, despite no prior knowledge of labels.
- Limitations:
  - Clustering purely based on features might not align perfectly with real-world labels.
  - Results are dependent on feature scaling and initialization.

## Conclusion
- K-Means is a powerful tool for clustering unlabelled data.
- Future Steps:
  - Explore feature importance.
  - Experiment with different clustering algorithms for comparison.
