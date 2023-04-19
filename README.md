# Dry Bean Classification
A simple classification example using 4 machine learning algorithms and neutral networks.

## Dataset Info
For the task, a dry bean dataset was acquired from the UCI Machine Learning Repository. It contains 13611 samples, with a total of 16 features; 12 dimensions and 4 shape forms, for 7 different dry beans (Seker, Barbunya, Bombay, Cali, Dermosan, Horoz and Sira). More info about the dataset can be found [here](https://archive-beta.ics.uci.edu/dataset/602/dry+bean+dataset).

# Notebook Walkthrough
## 1. Data and its visualization
After fetching the data directly from the website, unzipping it, a dataframe was created. A historgam was then plotted, containing the counts of the 7 different bean classes, to show the degree of the class imbalance involved. Next, a series of 16 graphs were plotted for the 16 features. Each graph compared the density of the bean classes against the feature being explored. 