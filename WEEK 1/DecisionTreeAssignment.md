# Identify risky bank loans using decision tree #

This post intends to finish the first assignment of course “Machine Learning for Data Analysis” on coursera. We are assigned to perform a decision tree analysis to test nonlinear relationships among a series of explanatory variables and a binary, categorical response variable.

In this assignment, I used finicial data which contains information on loans obtained from a credit agency in Germany. This data can be downloaded from UCI Machine Learning Data Repository on https://archive.ics.uci.edu/ml. There are total 1000 observations and 17 features in this data. The target is **default** which is a binary variable: 'yes' and 'no', meaning whether the loan went into default. The explanatory variables consist of the following 16 components: &quot;checking_balance&quot;, &quot;months_loan_duration&quot;, &quot;credit_history&quot;,       &quot;purpose&quot;, &quot;amount&quot;, &quot;savings_balance&quot;, &quot;employment_duration&quot;, &quot;percent_of_income&quot;, &quot;years_at_residence&quot;, &quot;age&quot;, &quot;other_credit&quot;, &quot;housing&quot;, &quot;existing_loans_count&quot;, &quot;job&quot;, &quot;dependents&quot;,  &quot;phone&quot;. The details of each features can be seen in Section 3 `str(credit)`.

To identify the risky bank loans, we build a decision tree model using different programming languages such as SAS, Python and R. To get the dataset, please go to https://github.com/debinqiu/ML_Course/issues and download *credit.txt* and convert to *credit.csv* file.

##1. Fit decision tree in SAS##
The decision tree is conducted by PROC HPSPLIT in SAS. To build the decision tree on training and testing data, we first randomly shuffle the original data and select the first 700 observations as training data and the rest as testing data. 
The SAS code is as follows.
```
TITLE 'Import credit.csv data';
FILENAME CSV "/home/debinqiu0/Practice/credit.csv" TERMSTR = CRLF;
PROC IMPORT DATAFILE = CSV OUT = credit DBMS = CSV REPLACE;
RUN;

PROC PRINT DATA = credit(OBS = 10); 
RUN;

TITLE 'Create training and testing data respectively by randomly shuffling strategy';
PROC SQL;
CREATE TABLE credit AS
SELECT * FROM credit
ORDER BY ranuni(0)
;
RUN;

TITLE 'Training data with 700 observations';
DATA credit_train;
SET credit;
IF _N_ <= 700 THEN OUTPUT;
RUN;

TITLE 'Testing data with 300 observations';
DATA credit_test;
SET credit;
IF _N_ > 700 THEN OUTPUT;
RUN;

ODS GRAPHICS ON;

PROC HPSPLIT DATA = credit_train SEED = 123;
TITLE 'Decision tree for credit training data';
CLASS checking_balance credit_history purpose savings_balance 
	  employment_duration other_credit housing job phone default;
MODEL default(event = 'yes') = checking_balance months_loan_duration  
		credit_history	purpose amount savings_balance employment_duration
	   	percent_of_income years_at_residence age other_credit 
	   	housing existing_loans_count job dependents phone default;
GROW ENTROPY;
PRUNE COSTCOMPLEXITY;
CODE FILE = '/home/debinqiu0/Practice/dt_credit.sas';
RUN;

TITLE 'Predictions on credit testing data';
DATA credit_pred(KEEP = Actual Predicted);
SET credit_test END = EOF;
%INCLUDE "/home/debinqiu0/Practice/dt_credit.sas";
Actual = default;
Predicted = (P_defaultyes >= 0.5);
run;

TITLE "Confusion Matrix Based on Cutoff Value of 0.5";
PROC FREQ DATA = credit_pred;
TABLES Actual*Predicted /norow nocol nopct;
RUN;
```

![cost_comp_sas](https://cloud.githubusercontent.com/assets/16762941/12804239/5286cfb8-cabf-11e5-9aee-8a490e5bbf1a.png)

The trend of cost complexity analysis shows that the smallest average ASE (0.176) obtains at cost complexity parameter = 0.0068. Let's look at the graph of fitted tree as follows. We can see that the most four important features are checking_balance, month_loan_duration, credit_history and savings_balance. To interpret the tree, we see that if the checking_balance is greater than 200 DM or unknown, 318 samples are classified as 'no' and 87.74% of them are truly 'no' in the training data. Otherwise, if the checking_balance is smaller or equal to 200 DM, and if months_loan_duration is less than 21.68 and if the credit_history is perfect or very good, 21 samples are classified as 'yes' with 71.43% accurate rate. We can interpret others in the same fasion.

![tree_sas](https://cloud.githubusercontent.com/assets/16762941/12804299/fdf6fecc-cabf-11e5-956f-23c8575640e3.png)

Finally, let's check the accuracy of the fitted decision tree on testing data. The confusion matrix gives us the accuracy of 59% ((145 + 32)/300) which is somewhat low. However, the result can be improved by using random forest or gradient boosting that will be covered in the latter section.

![conf_mat_sas](https://cloud.githubusercontent.com/assets/16762941/12804393/da77c714-cac0-11e5-85e4-71659664e8a7.png)


## 2. Fit decision tree in Python ##
Python `sklearn` package provides numerous functions to perform machine learning methods, including decision tree. We now give the Python code to fit the decision tree for bank loans data. 

```python
import pandas as pd
import os
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder
os.chdir('/Users/Deman/Desktop/Dropbox/ML')

credit = pd.read_csv("credit.csv")

credit = credit.dropna()
targets = LabelEncoder().fit_transform(credit['default'])

predictors = credit.ix[:,credit.columns != 'default']

# Recode categorical variables as numeric variables
predictors.dtypes
for i in range(0,len(predictors.dtypes)):
    if predictors.dtypes[i] != 'int64':
        predictors[predictors.columns[i]] = LabelEncoder().fit_transform(predictors[predictors.columns[i]])

pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size=.4)

#Build model on training data
classifier = DecisionTreeClassifier().fit(pred_train,tar_train)
predictions = classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)

#Displaying the decision tree
from sklearn import tree
#from StringIO import StringIO
from io import StringIO
#from StringIO import StringIO 
from IPython.display import Image
out = StringIO()
tree.export_graphviz(classifier, out_file=out)
import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())
```
Since Python does not provide pruing on the decision tree, the classification accuracy (69%) may be higher than that from SAS. Also, it results in a huge tree shown in the following graph. Without pruning, the tree is more likely to overfit the data.

```python
>>> sklearn.metrics.confusion_matrix(tar_test,predictions)
Out[2]: 
array([[220,  66],
       [ 58,  56]])

>>> sklearn.metrics.accuracy_score(tar_test, predictions)
Out[3]: 0.68999999999999995
```

![tree_python](https://cloud.githubusercontent.com/assets/16762941/12804810/397d63c4-cac4-11e5-8a9e-259b8f45391a.png)

## 3. Fit decision tree in R ##
We finally build a decision tree in R using `rpart` package. In fact, there are several other packages such as `tree`, `C5.0` to fit such model. Here we only use `rpart` package for simplicity. The R code is as follows.
```r
> credit <- read.csv("credit.csv")
> str(credit)
'data.frame':	1000 obs. of  17 variables:
 $ checking_balance    : Factor w/ 4 levels "< 0 DM","> 200 DM",..: 1 3 4 1 1 4 4 3 4 3 ...
 $ months_loan_duration: int  6 48 12 42 24 36 24 36 12 30 ...
 $ credit_history      : Factor w/ 5 levels "critical","good",..: 1 2 1 2 4 2 2 2 2 1 ...
 $ purpose             : Factor w/ 6 levels "business","car",..: 5 5 4 5 2 4 5 2 5 2 ...
 $ amount              : int  1169 5951 2096 7882 4870 9055 2835 6948 3059 5234 ...
 $ savings_balance     : Factor w/ 5 levels "< 100 DM","> 1000 DM",..: 5 1 1 1 1 5 4 1 2 1 ...
 $ employment_duration : Factor w/ 5 levels "< 1 year","> 7 years",..: 2 3 4 4 3 3 2 3 4 5 ...
 $ percent_of_income   : int  4 2 2 2 3 2 3 2 2 4 ...
 $ years_at_residence  : int  4 2 3 4 4 4 4 2 4 2 ...
 $ age                 : int  67 22 49 45 53 35 53 35 61 28 ...
 $ other_credit        : Factor w/ 3 levels "bank","none",..: 2 2 2 2 2 2 2 2 2 2 ...
 $ housing             : Factor w/ 3 levels "other","own",..: 2 2 2 1 1 1 2 3 2 2 ...
 $ existing_loans_count: int  2 1 1 1 2 1 1 1 1 2 ...
 $ job                 : Factor w/ 4 levels "management","skilled",..: 2 2 4 2 2 4 2 1 4 1 ...
 $ dependents          : int  1 1 2 2 2 2 1 1 1 1 ...
 $ phone               : Factor w/ 2 levels "no","yes": 2 1 1 1 1 2 1 2 1 1 ...
 $ default             : Factor w/ 2 levels "no","yes": 1 2 1 1 2 1 1 1 1 2 ...
> #Split into training and testing sets
> set.seed(123)
> train_sample <- sample(1000,700)
> credit_train <- credit[train_sample,]
> credit_test <- credit[-train_sample,]
> X_train <- credit_train[-c(which(colnames(credit) %in% 'default'))]
> X_test <- credit_test[-c(which(colnames(credit) %in% 'default'))]
> 
> # Build model on training data
> library(rpart)
> credit_model <- rpart(default~.,data = credit_train)
> 
> # Make predictions on testing data
> credit_pred <- predict(credit_model,X_test, type = 'class')
> # accuracy
> sum(diag(table(credit_test$default,credit_pred)))/300
[1] 0.74
> 
> # Displaying the decision tree
> library(rpart.plot)
> rpart.plot(credit_model,under = TRUE,faclen = 3,extra = 106)
```
The fitted model gives an accuracy of 74% on testing data with 300 observations, which is higher than those obtained from SAS and Python. Also, we also have the following graph of tree, which is simpler than that from Python but a bit more complex than that from SAS. We can also see now the first four important features are checking_balance, month_loan_duration, employment_duration and credit_history which are slightly different from those in SAS.

![tree_r](https://cloud.githubusercontent.com/assets/16762941/12804930/77c86ede-cac5-11e5-97eb-8dea1aa194f3.png)

