# Dry Bean Classification
A simple classification example using 4 machine learning algorithms and neutral networks.

## Dataset Info
For the task, a dry bean dataset was acquired from the UCI Machine Learning Repository. It contains 13611 samples, with a total of 16 features; 12 dimensions and 4 shape forms, for 7 different dry beans (Seker, Barbunya, Bombay, Cali, Dermosan, Horoz and Sira). More info about the dataset can be found [here](https://archive-beta.ics.uci.edu/dataset/602/dry+bean+dataset).

# Notebook Walkthrough
## 1. Data and its visualization
After fetching the data directly from the website, unzipping it, a dataframe was created. A historgam was then plotted, containing the counts of the 7 different bean classes, to show the degree of the class imbalance involved. Next, a series of 16 graphs were plotted for the 16 features. Each graph compared the density of the bean classes against the feature being explored.

## 2. Preprocessing
---
### Feature Selection
Out of the 16 features, 10 of the most relevant ones were to be selected for input to the model. This was done by evaluation of Anova F-scores for each feature, using the [SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest) and [f_classif](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif) functions in sklearn. A new dataframe was then made using these final features. 

---
### Label Transformation
Class labels of the 7 beans were encoded into integers (0-6). 
 
 ---
### Splitting and Scaling
The dataframe was split into training and testing datasets, 70:30 respectively. The labels of the test set were decoded back into the original bean labels for easy comparison with model output later onwards. [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) was applied to the input data of the train and test sets for reducing any dominance between features. 