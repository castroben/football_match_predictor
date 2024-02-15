# football_match_predictor
1. Introduction 

Football, also called “soccer” in the U.S.A, is arguably the most popular sport in the world. Almost every country in the world has its own professional football league (some countries have multiple), which means there are thousands of football matches played every year. Even with such many records, the outcomes of football matches, historically, are very hard to predict. As any avid football fan can corroborate, there are multiple reasons for why predicting the outcomes of football matches is so hard.  

All such reasons are what make the sport so beloved by its gigantic fanbase. Worldwide, there are numerous sports betting and lottery economical sectors, which when aggregated, generate hundreds of billions of U.S dollars in revenue every year. Of course, businesses in these sectors benefit from the immense popularity of sports and the nonsensical behavior of its fans. However, the most successful businesses are ones that carefully craft and manipulate the odds of wagers on sporting events by analyzing the probability distributions of their outcomes.  

Numerous advances in computational tools, and the mass adoption of data-centered processes, have allowed the materialization of computational systems finely tuned for predicting the outcomes of sporting events. A computational system capable of predicting the outcome of football matches has the potential to generate millions of U.S dollars in revenue. 

2. Formulation 

As mentioned in the previous sections, the goal of this project is to develop a computational system capable of predicting the outcome of football matches. To formulate the problem most appropriately, two observations must be made. First, like in many other sports, there are three possible general outcomes for any match: “home team wins”, “visitor team wins” or “draw”. To establish a convention, from now on the outcome “home team wins” will be referenced as “home”, and the outcome “visitor team wins” will be referenced as “away”. Second, theoretically there is an infinite number of ways in which each outcome can come about. For example, a “home” outcome can come about a 1-0 result, a 2-0 result, a 2-1 result, a 3-0, etc. Based on these observations, the problem of predicting football match outcomes can be formulated as two separate data mining tasks: classification and regression. 

2.1 Classification 

One of the ways to formulate the problem of predicting football match outcomes is as a classification problem. Under this formulation the numerical results of a match in terms of goals are ignored, instead the results are summarized as a “home”, “away” or “draw” outcome. The input to the classification task is a feature vector representing the characteristics of a particular football match (the exact composition of the feature vector will be discussed in later sections). The output of the task is a single label “home”, “away” or “draw” that encapsulates the results of the match. 

2.2 Regression 

Another way to formulate the problem of predicting football match outcomes is as a regression problem. In contrast to the previous formulation, the numerical results in terms of goals are fundamental under this formulation. The input to the regression task is a feature vector representing the characteristics of an individual team, not of the entire match (again, the exact composition of the feature vector will be discussed in later sections). The output of the task is a single integer representing the predicted number of goals the team will score. Consequently, the exact outcome of any match can be predicted by predicting the number of goals each team will score. 

3. Datasets 

3.1 Data Source 

The original data source for this project can be found at https://www.kaggle.com/datasets/analystmasters/world-soccer-live-data-feed. This data source contains four different datasets and a description page which details the components of and differences between the datasets. According to the description page, every column or feature in the entire data collection can be placed under one of two categories: Current Situation and Head-to-Head History. As the names imply, features under Current Situation represent characteristics of a team based on its current standing at the moment the data was gathered. In contrast, the features under Head-to-Head History represent characteristics of a team based on its historic standing.  

More details on the difference between the two categories can be found in the description page. However, what is most important is that 4 different datasets were generated based on these two categories: Mode One dataset (contains both categories), Mode Two dataset (contains only Current Situation data), Mode Three dataset (contains only Head-to-Head History data), and Mode Four dataset (none of the categories are available). Because the Mode One dataset has the greatest number of features and the greatest number of entries, it was selected among the four datasets as the most ideal one.  


3.2 Data Preprocessing 

Before any data preprocessing, the Mode One dataset consists of 62 columns and 4,789 entries. Each entry in the dataset corresponds to a singular match and captures some statistics of both teams facing each other in 60 columns. The remaining two columns capture the result of the match in terms of goals scored by each team.  

This dataset was preprocessed in two stages. Firstly, the dataset was preprocessed under general terms to eliminate impurities and garbage values. Secondly, the result of the first stage was further preprocessed based on the usage of the data (classification or regression). A snapshot of the unprocessed dataset is provided below for clarity. 

3.2.1 General Data Preprocessing 

This was the first stage of data preprocessing. During this stage three main actions were performed on the dataset: removing redundant/unimpactful features, removing corrupted/invalid entries, and dividing the data set. To determine which features were minimally impactful or redundant a simple exploratory data analysis was conducted. 

 A total of 24 features were determined to have unimpactful or redundant information and dropped from the dataset. Any entry that contained empty or null data was dropped from the dataset. Along with this, entries whose “Number of Head-to-Head Matches” value was less than or equal to 6 was dropped. Entries whose “Jumps” value was greater than 3 was also dropped (this value corresponds to the largest gap in years between any two of the past 6 matches). 

 Finally, the entire dataset was divided into 7 categories based on countries. The following are the resulting data subsets with their respective shapes: England (348, 35), Spain (162, 35), Germany (163, 35), Italy (193, 35), France (175, 35), World - every league except the previous five (2225, 35), All - the union of all the categories (3266, 35). This division was done because the English, Spanish, German, Italian and French leagues are considered the best in the world, and therefore, potentially statistically different than other leagues. As you can observe, some data subsets have big deficiencies in terms of datapoints for training.  

3.2.2 Preprocessing for Classification and Regression 

The datasets resulting from the previous stage were modified to generate two collections of data subsets: one to be used for classification purposes and one for regression purposes. For the classification collection, the last two columns (“Team 1 Goals” and “Teams 2 Goals”) were removed and summarized by a single label reading “Home”, “Away” or “Draw”. No columns were added or deleted besides these. 

For the regression collection each data entry, which was a combination of both teams’ statistics, was divided into two. This essentially created regression data subsets with half the number of features but twice the number entries. Finally, a single binary feature was added to each resulting data entry. This new feature indicates whether the team is playing at “Home” or playing “Away”. 


4. Algorithm 

The problem of predicting the outcome of football matches was formulated as two separate data mining tasks. Therefore, two algorithms needed to be selected, one for classification and one for regression. The process of selecting a suitable algorithm was similar for each task and is described in more detail below. 

4.1 Classification Algorithm 

Based on the material covered in class and my own research, five different classifiers were preselected for the task of generating a “Home”, “Away” or “Draw” label. The five classifiers preselected were: Decision Tree, Support Vector, Gaussian Naïve Bayes, Random Forest, and Ridge. The classifiers mentioned were implemented using the open source sklearn python library. For the preselection process the classifier hyperparameters were not tuned in any way. Rather, the classifiers were executed with minimal tuning to analyze the rudimentary behavior of the algorithms.  

The classifiers were executed with each country category data subset, but since the “All” category was the largest its results provided the best insights. Based on these results the Decision Tree Classifier and the Support Vector Classifier were initially discarded. The Decision Tree Classifier was discarded due to its comparatively low performance. Even though the Support Vector Classifier had the second highest performance, it never predicted “Draw” as an outcome, so its performance results were trivial.  

The Naïve Bayes Classifier was later discarded because after adding cross validation, its performance decreased. In contrast, when cross validation was added to the Random Forest Classifier, its performance increased and rivaled that of the Ridge Classifier with Cross Validation. After further deliberation and analysis, the Random Forest Classifier with cross-validation was selected as the algorithm to be experimented on. 

4.2 Regression Algorithm 

Even though regression algorithms were not explicitly covered in class, the intuition behind regression is akin to the intuition behind classification.  Because of this, developing a model that predicts the number of goals scored by a team based on the team’s season statistics was similar to developing a model that predicts a label for a match’s outcome based on both teams’ season statistics.  

The five regressors preselected for this task were: Lasso, Ridge, Elastic-Net and Random Forest. The regressors mentioned were implemented using the open source sklearn python library. For the preselection process the classifier hyperparameters were not tuned in any way. Rather, the classifiers were executed with minimal tuning to analyze the rudimentary behavior of the algorithms. 

The regressors were executed with each country category data subset, but since the “All” category was the largest its results provided the best insights. The resulting mean-squared-error results for this category were: Random Forest 1.5120972443355787, Lasso 1.537159887439279, Ridge 1.460546971699954, Elastic-Net 1.504756808485736. Based on these results, the Ridge regressor was selected to be experimented on. 


5. Experiment 

The parameters for the regression and classification algorithms were not selected manually. Rather, the sklearn class GridSearchCV was used to find the best parameter combinations, and as the name suggests, perform cross validation on the algorithms. 

5.1 Classification Experiment 

The following parameter grids were used for the Random Forest and Ridge classifiers, respectively. 

    param_grid = { 

                'n_estimators': [10, 50, 100, 200],   

                'criterion': ['gini', 'entropy'],  

                'max_features': ['auto', 'sqrt', 'log2'], 

                'bootstrap': [True, False] 

            }   

n_estimators = number of trees created. criterion = function used to measure the quality of a split.                    max_features = number of features to consider when looking for a best split.                                                                bootsrap = indicator of bootsrap samples used when building trees. 

param_grid = { 

        'alpha' : [1.0, 2.0, 5.0, 10.0], 

        'solver': ['auto', 'svd', 'saga'], 

        'random_state': [0, 5, 10] 

    } 

alpha = regularization strength. 									                                 solver = solver to use in computational routines (svd – Singular Value Decomposition, saga – Stochastic Average Gradient Descent ).                                                                                                                                                                              random_state = used by ‘saga’ to shuffle data. 

The algorithms were executed with all country category data subsets. However, only the results for the “All” category are reported here. To evaluate the performance of the classifiers, a full classification report was generated using sklearn.metrics.classification_report  method. The results below show the best parameter combinations and the performance metrics for the Random Forest and Ridge classifiers, respectively. 


Best parameters: {'bootstrap': True, 'criterion': 'entropy', 'max_features': 'sqrt', 'n_estimators': 200} 

.    | Precision | Recall | F1-score | Support 
 |  |  |  | 
Away |    0.45   |  0.44  |   0.45   |   243   
 |  |  |  | 
Draw |    0.30   |  0.08  |   0.12   |   210   
 |  |  |  | 
Home |    0.53   |  0.76  |   0.62   |   364   

Accuracy: 49%  


Best parameters: {'alpha': 10.0, 'fit_intercept': True, 'random_state': 0, 'solver': 'saga'} 

.    | Precision | Recall | F1-score | Support 
:---: | :---: | :---: | :---: | :---:
Away |    0.48   |  0.51  |   0.49   |   243  
:---: | :---: | :---: | :---: | :---:
Draw |    0.50   |  0.02  |   0.05   |   210  
:---: | :---: | :---: | :---: | :---:
Home |    0.54   |  0.82  |   0.65   |   364   

Accuracy: 52% 


As was mentioned in the introduction of this report, predicting the outcome of football matches is very difficult, especially if no in-game statistics are considered. The data that was used for predicting the outcome of a game contain no information about the game itself. Rather, the data contains information about current season standings of each team, and historical results between the two teams. The lack of data insights is a possible explanation for why the prediction accuracy of the algorithms are not particularly high. The highest accuracy was achieved with for the “World” category (all leagues expect English, Spanish, German, Italian and French) with an average of 58%. This value is significantly larger than any that of any other category, which confirms that the five country leagues mentioned previously are unique. 

5.2 Regression Experiment 

The parameter grid used for the Ridge regressor was identical to the grid used for the Ridge classifier. The algorithms were executed with all country category data subsets. However, only the results for the “All” category are reported here. To evaluate the performance of the regressor, the predicted values were rounded, and the mean-squared-error between the rounded predictions and the true values was computed. The following are the results of the regressor: 

Best parameters: {'alpha': 10.0, 'random_state': 10, 'solver': 'saga'} 

Mean squared error: 0.9222290263319045 

As you can observe, after tuning the parameters of the Ridge regression algorithm, the mean squared error decreased by 36%. Generally speaking, predicting the outcome of football matches using regression as explained in section 2.2 is more effective than using classification as explained in section 2.1. The mean squared error reported above could be acceptable since, on average, the number of goals predicted is less than 1 goal away from the true number. One explanation for the higher effectiveness of regression is the type of data available. The data used to classify and regress contains information about the season statistics of teams. This information has important insights on a team’s goals statistics such as total goals scored in the season, total goals received in the season, average number of goals at half-time, average number of goals at full-time. Therefore, even though the data does not contain in-game statistics, it contains information useful for regression.  


Conclusion 

Even though the original data was preprocessed in a slightly different way for classification and for regression, the type of data available was the same for both data mining tasks. As discussed in the previous section, the type of data available was better suited for regression than for classification. Different algorithms were implemented both for regression and classification, and the algorithms were executed with 7 different datasets. As hypothesized in section 3.2.1, the experiment results provide evidence that the English, Spanish, German, Italian and French leagues are statistically different from other leagues in the world. Finally, the results both for classification and regression are evidence that predicting the outcome of football matches has been and continues being a difficult task. Any football fanatic can validate this statement. Its unpredictability makes football exciting and loved by millions of people around the world. 
