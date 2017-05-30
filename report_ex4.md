
# Assignment 4 - Machine Learning with Python

### Lihi Verchik - 308089333 , Aviram Adiri - 302991468

In this assignment we participated in a comatition of Loan Prediction.

### First step: Reading and Organizing the data.

We started with Reading the dataset in a dataframe using Pandas, and looked at the top rows for general idea.



```python
%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib
    


```python
import pandas as pd
import numpy as np
import matplotlib as plt
```


```python
df_train = pd.read_csv("./sources/train.csv") 
df_test = pd.read_csv("./sources/test.csv") 
```


```python
df_train.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Loan_ID</th>
      <th>Gender</th>
      <th>Married</th>
      <th>Dependents</th>
      <th>Education</th>
      <th>Self_Employed</th>
      <th>ApplicantIncome</th>
      <th>CoapplicantIncome</th>
      <th>LoanAmount</th>
      <th>Loan_Amount_Term</th>
      <th>Credit_History</th>
      <th>Property_Area</th>
      <th>Loan_Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LP001002</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>5849</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LP001003</td>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4583</td>
      <td>1508.0</td>
      <td>128.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LP001005</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>3000</td>
      <td>0.0</td>
      <td>66.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LP001006</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>2583</td>
      <td>2358.0</td>
      <td>120.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LP001008</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>6000</td>
      <td>0.0</td>
      <td>141.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LP001011</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>5417</td>
      <td>4196.0</td>
      <td>267.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6</th>
      <td>LP001013</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>2333</td>
      <td>1516.0</td>
      <td>95.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LP001014</td>
      <td>Male</td>
      <td>Yes</td>
      <td>3+</td>
      <td>Graduate</td>
      <td>No</td>
      <td>3036</td>
      <td>2504.0</td>
      <td>158.0</td>
      <td>360.0</td>
      <td>0.0</td>
      <td>Semiurban</td>
      <td>N</td>
    </tr>
    <tr>
      <th>8</th>
      <td>LP001018</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4006</td>
      <td>1526.0</td>
      <td>168.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LP001020</td>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>12841</td>
      <td>10968.0</td>
      <td>349.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Semiurban</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>



Then, we got the summary of numerical variables, and plot the histogram of ApplicantIncome.


```python
df_train.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ApplicantIncome</th>
      <th>CoapplicantIncome</th>
      <th>LoanAmount</th>
      <th>Loan_Amount_Term</th>
      <th>Credit_History</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>614.000000</td>
      <td>614.000000</td>
      <td>592.000000</td>
      <td>600.00000</td>
      <td>564.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5403.459283</td>
      <td>1621.245798</td>
      <td>146.412162</td>
      <td>342.00000</td>
      <td>0.842199</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6109.041673</td>
      <td>2926.248369</td>
      <td>85.587325</td>
      <td>65.12041</td>
      <td>0.364878</td>
    </tr>
    <tr>
      <th>min</th>
      <td>150.000000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>12.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2877.500000</td>
      <td>0.000000</td>
      <td>100.000000</td>
      <td>360.00000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3812.500000</td>
      <td>1188.500000</td>
      <td>128.000000</td>
      <td>360.00000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5795.000000</td>
      <td>2297.250000</td>
      <td>168.000000</td>
      <td>360.00000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>81000.000000</td>
      <td>41667.000000</td>
      <td>700.000000</td>
      <td>480.00000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_train['ApplicantIncome'].hist(bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22ed305bb00>




![png](/images/output_8_1.png)



```python
# plot the histogram of ApplicantIncome using box plot to better understand the distributions
df_train.boxplot(column='ApplicantIncome')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22ed31457f0>




![png](/images/output_9_1.png)



```python
# the box plot confirms the presence of a lot of outliers/extreme values
# It can be driven by the fact that we are looking at people with different education levels
# box plot and segregate by education:
df_train.boxplot(column='ApplicantIncome', by = 'Education')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22ed322a1d0>




![png](/images/output_10_1.png)



```python
# we can see there is no substantial difference between the mean income of graduate and non-graduates. 
# But there are a higher number of graduates with very high incomes
```


```python
#Plot the histogram of LoanAmount
df_train['LoanAmount'].hist(bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22ed334def0>




![png](/images/output_12_1.png)



```python
# plot the histogram of LoanAmount using box plot to better understand the distributions
df_train.boxplot(column='LoanAmount')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22ed3359cc0>




![png](/images/output_13_1.png)



```python
# There are some extreme values here too
# Both ApplicantIncome and LoanAmount require some amount of data munging
```

### Second step: completing missing values.
Now, after getting and understanding the data, we should complete missing values in the data.
The exact parts of the completing steps you can easily find as comments in the code below.


```python
# Get the number of missing values in each column 
df_train.apply(lambda x: sum(x.isnull()),axis=0) 
df_test.apply(lambda x: sum(x.isnull()),axis=0) 
```




    Loan_ID               0
    Gender               11
    Married               0
    Dependents           10
    Education             0
    Self_Employed        23
    ApplicantIncome       0
    CoapplicantIncome     0
    LoanAmount            5
    Loan_Amount_Term      6
    Credit_History       29
    Property_Area         0
    dtype: int64




```python
# For missing data in Self_Employed, check which is the common value
df_train['Self_Employed'].value_counts()
```




    No     500
    Yes     82
    Name: Self_Employed, dtype: int64




```python
# About 86% values are "No" for "Self_Employed" 
# We'll fill all missing data of Self_Employed with "No" since it's safe
df_train['Self_Employed'].fillna('No',inplace=True)
df_test['Self_Employed'].fillna('No',inplace=True)
```


```python
# Create a Pivot table, which provides the median values for all the groups of unique values of Self_Employed and Education
table = df_train.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
table
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Education</th>
      <th>Graduate</th>
      <th>Not Graduate</th>
    </tr>
    <tr>
      <th>Self_Employed</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No</th>
      <td>130.0</td>
      <td>113.0</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>157.5</td>
      <td>130.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Define a function which returns the values of these cells
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]
```


```python
# Replace missing cells using by applying the function
df_train['LoanAmount'].fillna(df_train[df_train['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
df_test['LoanAmount'].fillna(df_test[df_test['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
```


```python
# Get the number of missing values in each column 
df_train.apply(lambda x: sum(x.isnull()),axis=0) 
df_test.apply(lambda x: sum(x.isnull()),axis=0) 
```




    Loan_ID               0
    Gender               11
    Married               0
    Dependents           10
    Education             0
    Self_Employed         0
    ApplicantIncome       0
    CoapplicantIncome     0
    LoanAmount            0
    Loan_Amount_Term      6
    Credit_History       29
    Property_Area         0
    dtype: int64




```python
# We still have to fill the missing data of Gender, Married, 
# Dependents, Loan_Amount_Term, and Credit_History 
```


```python
# Since the extreme values of LoanAmount are practically possible
# instead of treating them as outliers, log transformation to nullify their effectv
df_train['LoanAmount_log'] = np.log(df_train['LoanAmount'])
df_train['LoanAmount_log'].hist(bins=20)
df_test['LoanAmount_log'] = np.log(df_test['LoanAmount'])
df_test['LoanAmount_log'].hist(bins=20)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22ed3516be0>




![png](/images/output_24_1.png)



```python
# Combine both incomes as total income and take a log transformation of the same
df_train['TotalIncome'] = df_train['ApplicantIncome'] + df_train['CoapplicantIncome']
df_train['TotalIncome_log'] = np.log(df_train['TotalIncome'])
df_train['LoanAmount_log'].hist(bins=20) 

df_test['TotalIncome'] = df_test['ApplicantIncome'] + df_test['CoapplicantIncome']
df_test['TotalIncome_log'] = np.log(df_test['TotalIncome'])
df_test['LoanAmount_log'].hist(bins=20) 
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22ed3639278>




![png](/images/output_25_1.png)



```python
# Fill missing data for Loan_Amount_Term with 360
# Fill missing data for Credit_History with 1
df_train['Loan_Amount_Term'].fillna(360, inplace=True)
df_train['Credit_History'].fillna(1, inplace=True)

df_test['Loan_Amount_Term'].fillna(360, inplace=True)
df_test['Credit_History'].fillna(1, inplace=True)
```


```python
# Check the common values for Dependents
df_train['Dependents'].value_counts()
```




    0     345
    1     102
    2     101
    3+     51
    Name: Dependents, dtype: int64




```python
# Fill missing data for Dependents with 0 since it's the most common so it's safe
df_train['Dependents'].fillna(0, inplace=True)
df_test['Dependents'].fillna(0, inplace=True)
```


```python
# Check the common values for Married
df_train['Married'].value_counts()
```




    Yes    398
    No     213
    Name: Married, dtype: int64




```python
# Fill missing data for Married with Yes since it's the most common so it's safe
df_train['Married'].fillna('Yes', inplace=True)
df_test['Married'].fillna('Yes', inplace=True)
```


```python
# Check the common values for Gender
df_train['Gender'].value_counts()
```




    Male      489
    Female    112
    Name: Gender, dtype: int64




```python
# Fill missing data for Gender with Male since it's the most common so it's safe
df_train['Gender'].fillna('Male', inplace=True)
df_test['Gender'].fillna('Male', inplace=True)
```


```python
# Get the number of missing values in each column 
df_train.apply(lambda x: sum(x.isnull()),axis=0) 
df_test.apply(lambda x: sum(x.isnull()),axis=0) 
```




    Loan_ID              0
    Gender               0
    Married              0
    Dependents           0
    Education            0
    Self_Employed        0
    ApplicantIncome      0
    CoapplicantIncome    0
    LoanAmount           0
    Loan_Amount_Term     0
    Credit_History       0
    Property_Area        0
    LoanAmount_log       0
    TotalIncome          0
    TotalIncome_log      0
    dtype: int64



### Third step: Training the Model following the train data.
Now, after all the information is completed, we can start with the Training part.
before that, let's change all inputs to be numeric (for sklearn).


```python
# Now we can start building a predictive model
# Using sklearn requires all inputs to be numeric - change all inputs to be numeric
from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    df_train[i] = le.fit_transform(df_train[i].astype(str))
    
from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
le = LabelEncoder()
for i in var_mod:
    df_test[i] = le.fit_transform(df_test[i].astype(str))
```


```python
df_train.dtypes
df_test.dtypes
```




    Loan_ID               object
    Gender                 int64
    Married                int64
    Dependents             int64
    Education              int64
    Self_Employed          int64
    ApplicantIncome        int64
    CoapplicantIncome      int64
    LoanAmount           float64
    Loan_Amount_Term     float64
    Credit_History       float64
    Property_Area          int64
    LoanAmount_log       float64
    TotalIncome            int64
    TotalIncome_log      float64
    dtype: object




```python
# Define a generic classification function, which takes a model as input 
# and determines the Accuracy and Cross-Validation scores

#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome]) 
```

### First model: Gradient Boosting model.


```python
# Make Gradient Boosting model with 'Credit_History'
from sklearn.ensemble import GradientBoostingClassifier

outcome_var = 'Loan_Status'
model1 = GradientBoostingClassifier()
predictor_var1 = [ 'Dependents', 'Education', 'Credit_History', 'TotalIncome_log']

classification_model(model1, df_train, predictor_var1, outcome_var)

# Make test set predictions
test_preds1 = model1.predict(df_test[predictor_var1])

# Create a submission for Analytics Vidhya
submission1 = pd.DataFrame({"Loan_ID":df_test["Loan_ID"], "Loan_Status":test_preds1})

submission1['Loan_Status'] = submission1['Loan_Status'].map({1: 'Y', 0: 'N'})

# Save submission to CSV
submission1.to_csv('submission1.csv',sep=',', index=False)

```

    Accuracy : 86.319%
    Cross-Validation Score : 78.669%
    

#### Result:
After checking several options for predicators, the best result that we found what with these predicators:

- Dependents
- Education
- Credit_History
- TotalIncome_log
 
 the result we got in the compatition was 0.77778.
 
 ![png](/images/score_GradientBoostingClassifier.PNG)
 
 The result is attached under the name: "results\GradientBoostingClassifier.csv"
 
 The score for this result is attached under the name: "score_GradientBoostingClassifier.PNG"

### Second model:  K-Neighbors model.


```python
# Make K-Neighbors Classifier model with 'Credit_History'
from sklearn.neighbors import KNeighborsClassifier

model2 = KNeighborsClassifier()
predictor_var2 = ['Gender', 'Married', 'Dependents', 'Education',
                  'Self_Employed', 'Credit_History','LoanAmount_log','TotalIncome_log']

classification_model(model2, df_train, predictor_var2, outcome_var)

# Make test set predictions
test_preds2 = model2.predict(df_test[predictor_var2])

# Create a submission for Analytics Vidhya
submission2 = pd.DataFrame({"Loan_ID":df_test["Loan_ID"],
                           "Loan_Status":test_preds2})

submission2['Loan_Status'] = submission2['Loan_Status'].map({1: 'Y', 0: 'N'})

# Save submission to CSV
submission2.to_csv('submission2.csv',sep=',', index=False)
```

    Accuracy : 81.433%
    Cross-Validation Score : 75.727%
    

#### Result:
After checking several options for predicators, the best result that we found what with these predicators:

- Gender
- Married
- Dependents
- Education
- Self_Employed
- Credit_History
- LoanAmount_log
- TotalIncome_log
 
 the result we got in the compatition was 0.7708.
 
  ![png](/images/score_kNeighborsClassifier.PNG)
 
 The result is attached under the name: "results\kNeighborsClassifier.csv"
 
 The score for this result is attached under the name: "score_kNeighborsClassifier.PNG"

## Summarize
The best result that we got was with Gradient Boosting Classifier.

  ![png](/images/leader_board.PNG)
 
The place at the leadership board is attached under: "leader_board.PNG"
