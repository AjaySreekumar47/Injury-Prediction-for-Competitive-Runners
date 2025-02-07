# Injury-Prediction-for-Competitive-Runners

UPDATE 1:

PROBLEM STATEMENT:

The problem statement for this project is: while running is a great form of exercise, recreation and sport participation for adults – running under adverse conditions or with inadequate clothing or equipment can cause physical stress and a variety of injuries. Can we predict which factors exactly caused these injuries? Based on that, can we predict if a runner A is injured if they’re running in a specific set of conditions or factors? 

The original data came from DataverseNL in association with University of Groningen. 

Link: https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/UWU9PV 

Kaggle Version Link: https://www.kaggle.com/datasets/shashwatwork/injury-prediction-forcompetitive-runners/data 

The data set consists of a detailed training log from a Dutch high-level running team over a period of seven years (2012-2019). Also included were the middle and long-distance runners of the team, that is, those competing on distances between the 800 meters and the marathon. This design decision is motivated by the fact that these groups have strong endurance-based components in their training, making their training regimes comparable. The head coach of the team did not change during the years of data collection. The data set contains samples from 74 runners, of whom 27 are women and 47 are men. At the moment of data collection, they had been in the team for an average of 3.7 years. Most athletes competed on a national level, and some also on an international level. The study was conducted according to the requirements of the Declaration of Helsinki and was approved by the ethics committee of the second author’s institution. 

For data analysis and manipulation, we will use pandas, matplotlib as well as NumPy libraries in Python (PyCharm). As part of Machine Learning techniques or algorithms, we will test which of Logistic Regression, K-Nearest Neighbours or Random Forest Classifiers would give us the highest F1 scores, and correspondingly find the best possible parameters for the model – which should give us an accuracy score above 85%, so that we have a feasible model to use. 

UPDATE 2:

Data Preparation Plan: 

Data preparation plays a vital role in every machine learning endeavour, guaranteeing that the data is appropriately formatted for analysis and modelling purposes. The process begins with reading the dataset from a CSV file, followed by an exploration phase (EDA). During exploration, key steps include printing the first few rows and columns of the dataset to get a glimpse of its structure, examining data types, summarizing statistics, checking for missing values, and identifying unique values in categorical variables. We also attempt to find out any relationships between the variables we utilize in the dataset. 

Visualizing the correlation matrix helps to understand the relationships between predictor variables, shedding light on potential multicollinearity issues and guiding feature selection. Furthermore, visualizing the distribution of the target variable, 'injury', using count plots or pie charts provides insights into the class distribution and potential class imbalances. 

To ensure data quality, duplicate rows are identified and removed if present. Subsequently, the dataset is split into predictor variables (X) and the target variable (y), with the former representing the features used for prediction and the latter representing the outcome of interest. Normalization of predictor variables using techniques like ‘StandardScaler‘ may be employed to scale the features to a similar range, facilitating model convergence and performance. 

Finally, the dataset undergoes partitioning into training and testing subsets employing the 'train_test_split' function, commonly employing ratios like 80:20 or 70:30. The training set assumes the role of training machine learning models, whereas the testing set remains segregated for the purpose of evaluating model performance on unseen data, thereby scrutinizing its capacity for generalization. 

Predictor Variables (Features) and Response (Target Variable): 
The predictor variables, also known as features, encompass various metrics related to athlete training and performance, such as the number of sessions, maximum kilometres covered in a day, total kilometres logged, number of tough sessions, among others. These features serve as inputs to the machine learning models and are hypothesized to influence the likelihood of injury occurrence. 

On the other hand, the response variable, 'injury', serves as the target variable that the models aim to predict. It is a binary variable indicating whether an athlete suffered an injury during the observation period, with values of 1 representing the presence of an injury and 0 denoting its absence. Predicting this target variable is the primary objective of the machine learning task, as it enables proactive injury management and prevention strategies in sports and athletic training settings. 

Training and Testing Data Sets: 
The training data set comprises a subset of the original data reserved for model training, typically the majority portion, such as 80% of the data. This subset includes both predictor variables (X_train) and the corresponding target variable (y_train), allowing the models to learn patterns and relationships between features and the target. 

Conversely, the testing data set encompasses the residual portion of the initial data, typically accounting for approximately 20% of the dataset, and remains distinct from the training set. This subset comprises predictor variables (X_test) alongside their corresponding target variable (y_test), functioning as concealed data utilized to assess the performance of trained models. Through evaluating model performance on unseen data, we acquire valuable insights into the models' ability to generalize to novel instances, thereby instilling confidence in their predictive prowess. 

Machine Learning Techniques: 
The machine learning techniques employed in the project include logistic regression, decision trees, random forests, and support vector machines (SVM). Each technique offers unique advantages and is suitable for different types of data and problem domains. 

Logistic Regression: Logistic regression, a linear model tailored for binary classification tasks, estimates the likelihood that a specific instance pertains to a designated class. Despite its simplicity, logistic regression proves highly effective, providing both interpretability and straightforward implementation.  

Decision Trees: Decision trees, as non-linear models, divide the feature space into distinct regions, offering an intuitive and visually understandable approach. This segmentation process occurs iteratively, with the data being recursively split according to specific feature thresholds. Consequently, a hierarchical tree-like structure is formed, wherein each terminal node, or leaf, corresponds to a particular class label. 

Random Forests: Random Forests are powerful ensemble learning methods that leverage multiple decision trees to enhance predictive performance and mitigate overfitting. By introducing randomness in the tree-building process through the consideration of subsets of features and instances, Random Forests generate a diverse set of trees. These trees' predictions are combined to produce more robust and accurate predictions, making Random Forests a popular choice for various machine learning tasks. 

Support Vector Machines (SVM): SVM, a robust algorithm suitable for both classification and regression endeavours, strives to identify the optimal hyperplane within the feature space. This hyperplane effectively distinguishes instances belonging to distinct classes while simultaneously maximizing the margin between these classes. SVMs excel particularly in high-dimensional spaces and exhibit proficiency in capturing intricate relationships existing between features and the target variable. 

Each machine learning method involves training the model using a designated training dataset to adjust its parameters, while a separate testing dataset is employed to gauge how well the model performs. Various metrics, including accuracy, precision, recall, F1-score, and the confusion matrix, are computed to measure the efficacy of each model in forecasting injuries based on the provided predictors. The selection of a particular technique hinges on factors like the characteristics of the data, the desired level of interpretability of the model, and the balance between bias and variance.
