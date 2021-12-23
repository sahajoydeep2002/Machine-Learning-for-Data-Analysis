#Predict a baseball player’s salary#

In this assignment, we want to predict a baseball player's salary using linear regression, especially penalized linear regression such as Lasso regression, since there are 19 explanatory variables in the **Hitters** dataset which contains 322 observations. The **explanatory variables** in this dataset are listed as follows. 
- AtBat: Number of times at bat in 1986
- Hits : Number of hits in 1986
- HmRun: Number of home runs in 1986
- Runs : Number of runs in 1986
- RBI  : Number of runs batted in in 1986
- Walks: Number of walks in 1986
- Years: Number of years in the major leagues
- CAtBat: Number of times at bat during his career
- CHits: Number of hits during his career
- CHmRun: Number of home runs during his career
- CRuns: Number of runs during his career
- CRBI: Number of runs batted in during his career
- CWalks: Number of walks during his career
- League: A factor with levels A and N indicating player's league at the end of 1986
- Division: A factor with levels E and W indicating player's division at the end of 1986
- PutOuts: Number of put outs in 1986
- Assists: Number of assists in 1986
- Errors: Number of errors in 1986
- NewLeague: A factor with levels A and N indicating player's league at the beginning of 1987

The **response** is the player's salary.
- Salary: 1987 annual salary on opening day in thousands of dollars

To fit a linear regression, we first remove the 59 missing values in salary and transform the categorical variables into numeric variables.
```
TITLE 'Import Hitters.csv data';
FILENAME CSV "/home/debinqiu0/Practice/Hitters.csv" TERMSTR = CRLF;
PROC IMPORT DATAFILE = CSV OUT = Hitters DBMS = CSV REPLACE;
RUN;

TITLE 'Data cleaning and transformation';
DATA Hitters_New;
SET Hitters;
IF League = 'A' THEN League1 = 1;
IF League = 'N' THEN League1 = 2;
IF NewLeague = 'A' THEN NewLeague1 = 1;
IF NewLeague = 'N' THEN NewLeague1 = 2;
IF Division = 'E' THEN Division1 = 1;
IF Division = 'W' THEN Division1 = 2;
IF Salary = 'NA' THEN DELETE;
Salary1 = INPUT(Salary, comma6.);
DROP League NewLeague Division Salary;
RUN;
```
Secondly, we fit a classic linear regression to check whether we have to use a more complex fitting methodology (penalized linear regression) or not. The parameter estimates from classic linear regression below show that of 19 explanatory variables, 13 were not relevant to the salary, so it is appropriate to perform the variable selection. 
```
TITLE 'Run a classic linear regression';
PROC REG DATA = Hitters_New;
MODEL Salary1 = AtBat Hits HmRun Runs RBI Walks Years 
               CAtBat CHits CHmRun CRuns CRBI CWalks 
               League1 Division1 PutOuts Assists Errors NewLeague1;
RUN;
```
![linearregression](https://cloud.githubusercontent.com/assets/16762941/13198902/0edfdc92-d7e3-11e5-81a4-8dab903e7e30.png)

Now we fit a Lasso regression on the Hitters data. We first split the entire dataset into training and testing data with 70% and 30% of observations, respectively. 
```
TITLE 'Split into training and testing data';
PROC SURVEYSELECT DATA = Hitters_New OUT = traintest SEED = 123
SAMPRATE = 0.7 METHOD = SRS OUTALL;
RUN;
```
![traintest](https://cloud.githubusercontent.com/assets/16762941/13198903/0ee22ca4-d7e3-11e5-9551-eaaf80b6bd10.png)

To fit the Lasso regression, we use the GLMSELECT procedure in SAS as follows. The least angle regression algorithm with k=10 fold cross validation was used to estimate the lasso regression model in the training set, and the model was validated using the test set. The change in the cross validation average (mean) squared error at each step was used to identify the best subset of predictor variables.
```
ODS GRAPHICS ON;
TITLE 'Run a Lasso regression';
PROC GLMSELECT DATA = traintest PLOTS = ALL SEED = 123;
PARTITION ROLE = SELECTED(train = '1' test = '0');
MODEL Salary1 = AtBat Hits HmRun Runs RBI Walks Years 
               CAtBat CHits CHmRun CRuns CRBI CWalks 
               League1 Division1 PutOuts Assists Errors 
               NewLeague1/SELECTION = LAR(CHOOSE = CV STOP = NONE)
               CVMETHOD = RANDOM(10);
RUN;
```
The 'Lar Selection Summary' table shows that we obtain the optimal model at step 14 with biggest reduction in residual sum of squares [with biggest CV PRESS]. In this selected model, of 19 explanatory variables, 14 are retained and the rest don't contribute to predict the salary. The selected variables are CRBI, Hits, Walks, PutOuts, CHits, Division1 (Division), Assists, CWalks, HmRun, Errors, AtBat, NewLeague1 (NewLeague), Years and Runs. 
![lasso1](https://cloud.githubusercontent.com/assets/16762941/13198897/0eda92aa-d7e3-11e5-9fd2-a2230732de1d.png)

Now let us look at the changes of coefficients when explanatory variables are added sequentially. The graph shows that at step 14, Hits, Walks, Runs and AtBat were most strongly associated with salary. Of them, AtBat were negatively associated with salary and the others were positively associated with salary. These 14 variables accounted for 65.2% of the variance in the salary response variable.  
![lasso2](https://cloud.githubusercontent.com/assets/16762941/13198898/0edacd92-d7e3-11e5-8d20-804262571895.png)

We can also see from the graph of trend of reduction of ASE for both training and testing data that the prediction accuracy is rather stable for the two datasets.

![lasso4](https://cloud.githubusercontent.com/assets/16762941/13198900/0edbb2fc-d7e3-11e5-8ece-a9083bf3268a.png)

The fitting information of the selected model are shown as follows.

![lasso5](https://cloud.githubusercontent.com/assets/16762941/13198901/0edbf546-d7e3-11e5-98eb-0618af68c95c.png)

Therefore, the salary can be predicted by the linear equation as follows.

Salary = 140.133 - 1.1452\*AtBat + 5.2491\*Hits - 1.7184\*HmRun - 0.5985\*Runs + 5.3258\*Walks - 6.7554\*Years + 0.2302\*CHits + 0.8002\*CRBI - 0.5555\*CWalks - 105.8088\*Division +0.2639\*PutOuts - 0.048385\*Assists - 0.9563\*Errors - 21.4788\*NewLeague.
