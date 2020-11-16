# Cannabis Classifications
This project is aimed at accurately classifying strains of cannabis by race (indica, sativa, or hybrid) using a classification model. 

### Data Sources
The Strain and Kushy APIs were called to obtain information on over 5k strains of cannabis. WikiLeaf was webscraped to obtain an addition 2k strains. 

### Features Used
THC & CBD content
Taste
Smell
Effects

### EDA 
Class imbalances (51% hybrids, 30% indicas, 19% sativas) were rectified using SMOTE to even the number of data points across each race.

### Models Used 
Dummy Baseline Model
Classification models with all features
Classification models with hand selected features
Classification models with PCA prinicpal component variables
Grid Search

Modeling
Baseline Dummy Model
Accuracy: 32.18% Average F-1 Score: 0.74

Baseline Matrix

Logistic Regression Model
57.85% Accuracy: 68.6

Logistic Matrix

Best KNN Model (K=19)
This model improved accuracy from the baseline, but was not the best model. Accuracy: 66.86% Average F-1 Score: 0.6

Best KNN Matrix

Best Random Forest
Accuracy: 68.31% Average F-1 Score: 0.6

Best Random Forest

Scalar Vector Machine
Accuracy: 69.48% Average F-1 Score: 0.63

SVM mtx
