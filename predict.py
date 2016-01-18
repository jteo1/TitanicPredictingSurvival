import pandas as  pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

#Extracts name "Title", ie. Mr, Ms, Capt, Sir, etc
def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return "na"

#to be applied on each row to group titles into similar overarching categories
def bucket_titles(x):
    title = x["Title"]
    if title in ["Major", "Capt", "Col"]:
        return "Military"
    elif title in ["Don", "Jonkheer", "Rev", "Master", "Sir"] or title is "Dr" and x["Sex"] is "male":
        return "HighMale"
    elif title in ["Lady", "Mlle", "the Countess"] or title is "Dr" and x["Sex"] is "female":
        return "HighFemale"
    elif title in ["Mr"]:
        return "RegMale"
    elif title in ["Ms", "Mrs", "Miss", "Mme"]:
        return "RegFemale"
    else:
        return "Unknown"

#extract and modify features in the given dataframe to train to Logistic Regression
def modify_dataframe(df):
    #df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    #df["Fare"] = df["Fare"]/df["Numfamily"]
    #df["Embarked"] = df["Embarked"].map({"S" : 0, "C" : 1, "Q" : 2}).astype(int)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Sex"] = df["Sex"].map({"male" : 0, "female" : 1}).astype(int)
    df = df.dropna(subset = ["Embarked"], axis = 0) #only drop rows where there is a NaN in embarked column
    df["Numfamily"] = df["SibSp"] + df["Parch"] + 1

    #get titles from names, bucketize titles into 6 main groups, then create the dummy variable columns
    df["Title"] = df["Name"].map(lambda x: get_title(x))
    df["Title"] = df.apply(bucket_titles, axis = 1)
    df["Title"] = df["Title"].map({"HighFemale" : 5, "RegFemale" : 4, "HighMale" : 3, "Military" : 2, "RegMale" : 1, "Unknown" : 0}).astype(int)
    df = pd.concat([df, pd.get_dummies(df['Title']).rename(columns=lambda x: 'Title_' + str(x))], axis=1)

    return df

#TODO: add plot visualizations of features
def visualize_dataframe(df):
    # specifies the parameters of our graphs
    fig = plt.figure(figsize=(18,6), dpi=100)
    plt.show()

#-------------------------------_END HELPER FUNCTIONS----------------------------
#--------------------------------------------------------------------------------

#read training data into dataframe, ignore first row since its the header row
train = pd.read_csv('train.csv', header = 0)
train = modify_dataframe(train)
#visualize_dataframe(train)

#the meaningful features that I decided on
predictor_columns = ["Pclass", "Sex", "Age", "Numfamily", "Title_0", "Title_1", "Title_2", "Title_3", "Title_4"]

#Fit the training data to a Logistic Regression using predictor_columns to predict survival
predictor = LogisticRegression()
predictor.fit(train[predictor_columns], train["Survived"])
#print pd.DataFrame(zip(predictor_columns, np.transpose(predictor.coef_))) #print coefficients

#use test data to create prediction output
test_input = pd.read_csv("test.csv", header = 0)
test_input = modify_dataframe(test_input)
predictions = predictor.predict(test_input[predictor_columns])

#write predictions for submissions with 2-tuple rows of PassengerId and Survived
prediction_output = pd.DataFrame({"PassengerId" : test_input["PassengerId"],
                                  "Survived" : predictions
                                 })
#output to csv file while omitting the indexes
prediction_output.to_csv("prediction_output.csv", index = False)
