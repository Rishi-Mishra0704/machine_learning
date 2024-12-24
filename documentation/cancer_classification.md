# Cancer Classification with Neural Networks

## Objective
The goal of this project is to build a neural network model using TensorFlow/Keras to classify tumors as benign (0) or malignant (1) based on various diagnostic features of breast cancer.

## Dataset
The dataset `cancer_classification_nn.csv` contains 31 features, including one target column `benign_0__mal_1`, which indicates whether the tumor is benign or malignant.

### Features
- **Mean Values**: Statistics derived from images, e.g., `mean radius`, `mean texture`, `mean perimeter`.
- **Error Metrics**: Deviations such as `radius error`, `texture error`.
- **Worst Case Values**: E.g., `worst radius`, `worst perimeter`.
- **Target Variable**: `benign_0__mal_1` (0: benign, 1: malignant).

## Data Preprocessing
### Steps Taken:
1. **Data Inspection**:
   - Used `.head()` and `.describe()` to inspect and summarize the dataset.
2. **Exploratory Data Analysis**:
   - Visualized class distribution using Seaborn's `countplot`.
   - Analyzed feature correlations using `.corr()` and bar plots.
3. **Feature-Target Split**:
   - Features (`X`) and target (`y`) were separated.
4. **Train-Test Split**:
   - Data was split into training (75%) and testing (25%) sets using `train_test_split`.
5. **Feature Scaling**:
   - Scaled features to a range of [0,1] using `MinMaxScaler`.

## Model Architecture
### Initial Model:
- **Layers**:
  - Input Layer: 30 neurons, `relu` activation.
  - Hidden Layer: 15 neurons, `relu` activation.
  - Output Layer: 1 neuron, `sigmoid` activation.
- **Loss Function**: Binary Crossentropy.
- **Optimizer**: Adam.
- **Epochs**: 600.

### Improvements:
1. **Early Stopping**:
   - Monitored validation loss to prevent overfitting.
   - Set patience to 25 epochs.
2. **Dropout Regularization**:
   - Added dropout layers (50% rate) to mitigate overfitting.

## Model Training
1. **Training without Early Stopping**:
   - Observed overfitting after prolonged training.
2. **With Early Stopping**:
   - Improved generalization by halting training when validation loss plateaued.
3. **With Dropout**:
   - Further reduced overfitting.

## Evaluation
1. **Loss Visualization**:
   - Plotted training and validation loss for each model version.
2. **Prediction**:
   - Converted predictions to binary values using threshold 0.5.
3. **Performance Metrics**:
   - Generated classification report (Precision, Recall, F1-Score).
   - Created confusion matrix.

## Results
- Achieved satisfactory classification accuracy with minimal overfitting.
- Early stopping and dropout layers significantly improved model performance.

## Next Steps
- Experiment with more complex architectures (e.g., additional hidden layers).
- Perform hyperparameter tuning for optimal results.
- Investigate other regularization techniques or preprocessing methods.
