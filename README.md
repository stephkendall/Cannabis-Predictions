# Cannabis-Predictions

This project was aimed at classifying strains of cannabis by their race (indica, sativa, or hybrid) using various attributes associated to the plant. Data regarding race, taste, and effects was gathered from The Strain API, along with webscraped information from WikiLeaf for the thc/cbd contents, and more strains and their thc contents were accessed from CSVs originally from the Kushy API. After running various models and balancing the classes using SMOTE, I was to achieve an accuracy of 70.4% in classification with Logistic Regression. 


## EDA 
After cleaning null values and matching strain information across the datasets using Fuzzy Wuzzy, I had severe class imbalance with 811 hybrid strains, 482 indica strains, and only 290 sativa strains. My continuous variables of THC and CBD contents displayed similarities across the three classes. In future applications of this project, the use of TF-IDF through Natural Language Processing would be useful in using the tastes and effects (categorical variables) for accurate classifications. 
 


## Models Used
Dummy Classifier as Baseline
Logistic Regression
KNN
Grid Search
