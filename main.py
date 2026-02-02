import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# loading data to pandas DataFrame
loan_dataset = pd.read_csv('dataset.csv')

type(loan_dataset)

# printing the first 5 rows of the dataframe
loan_dataset.head()

# number of rows and columns 
loan_dataset.shape

# statistical measures
loan_dataset.describe()

# number of missing values in each column
loan_dataset.isnull().sum()

# dropping the missing values
loan_dataset = loan_dataset.dropna()

# number of missing values in each column
loan_dataset.isnull().sum()

# label incoding
loan_dataset.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)

# printing the first 5 rows of the dataframe
loan_dataset.head()

# Dependent Column Values
loan_dataset['Dependents'].value_counts()

# replacing the value of 3+ to 4
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)

# Dependent Column Values
loan_dataset['Dependents'].value_counts()

# education & loan Status
sns.countplot(x='Education', hue='Loan_Status',data=loan_dataset)

# matrial & loan Status
sns.countplot(x='Married', hue='Loan_Status',data=loan_dataset)

# convert categorical columns to numerical values
loan_dataset.replace({"Married": {'No': 0, 'Yes':1}, "Gender": {'Male':1, 'Female':0}, "Self_Employed": {'No':0, 'Yes':1},
                      "Property_Area": {'Rural':0, 'Semiurban':1, 'Urban':2}, "Education": {'Graduate':1, 'Not Graduate':0}}, inplace=True)

# printing the first 5 rows of the dataframe
loan_dataset.head()

# separating the data and labels
x = loan_dataset.drop(columns=['Loan_ID','Loan_Status'], axis=1)
y = loan_dataset['Loan_Status']

print(x)

print(y)

scaler = StandardScaler()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=2)

print(x.shape, x_train.shape, x_test.shape)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

classifier = svm.SVC(kernel='linear')

# training the support Vector Machine model
classifier.fit(x_train, y_train)

# accuarcy score on training data
x_training_prediction = classifier.predict(x_train)
training_data_accuarcy = accuracy_score(x_training_prediction, y_train)
print(training_data_accuarcy)

# accuarcy score on test data
x_test_prediction = classifier.predict(x_test)
test_data_accuarcy = accuracy_score(x_test_prediction, y_test)
print(test_data_accuarcy)

gender = input("What is your Gender? ").lower()
gender = int(gender.replace('female','0').replace('male','1'))
married = input("Are you married? ").lower()
married = int(married.replace('no','0').replace('yes','1'))
if married == 1:
    depender = int(input("How many people you are dependent? "))
    if depender > 3:
        depender = 4
else:
    depender = 0
education = input("Are you Graduated? ").lower()
education = int(education.replace('yes','1').replace('no','0'))
Self_Employed = input("Are you Self Employed? ").lower()
Self_Employed = int(Self_Employed.replace('no','0').replace('yes','1'))
ApplicantIncome = int(input("How much Applicant Income? "))
CoapplicantIncome = int(input("How much Coapplicant Income? "))
LoanAmount = int(input("How much Loan Amount do you need? "))
Loan_Amount_Term = int(input("What is the Loan Amount Term? "))
Credit_History = int(input("What is the Credit History? (1 --> Good , 0 --> Have a Problems) "))
Property_Area = input("What is your Poperty Area? (Rural, Semiurban, Urban) ").lower()
Property_Area = int(Property_Area.replace('rural','0').replace('semiurban','1').replace('urban','2'))

input_data = np.array([[ 
    gender, married, depender, education, Self_Employed,
    ApplicantIncome, CoapplicantIncome, LoanAmount,
    Loan_Amount_Term, Credit_History, Property_Area
]])

input_data = scaler.transform(input_data)

predict = classifier.predict(input_data)

if predict[0] == 1:
    print("Your Loan Application is Accepted")
else:
    print("Your Loan Application is Rejected")