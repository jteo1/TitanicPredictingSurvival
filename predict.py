import pandas as  pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None #prevent from warning about modifying copy of a DataFrame

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

def visualize_dataframe(df):
    #setup a 2x3 grid of plots
    fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize=(20,10))

    #Total survival numbers
    ax1 = df.Survived.value_counts(ascending = True).plot(kind = "bar", alpha = 0.5, ax = axes[0,0], color = ['g', 'darkblue'])
    ax1.set_xticklabels(["Survived","Died"], rotation=0)
    ax1.set_title("Total Number of Survivors")

    #Survival by male vs female
    df_gender = pd.DataFrame(columns = ["Survived", "Died"])
    df_gender["Survived"] = df.Sex[df.Survived == 1].value_counts(sort = False)
    df_gender["Died"] = df.Sex[df.Survived == 0].value_counts(sort = False)
    ax2 = df_gender.plot(kind = "bar", alpha = 0.5, ax = axes[0,1], color = ['g', 'darkblue'])
    ax2.set_title("Survival by Sex")
    ax2.set_xticklabels(["Male","Female"], rotation=0)

    #Survival by passenger class
    df_class = pd.DataFrame(columns = ["Survived", "Died"])
    df_class["Survived"] = df.Pclass[df.Survived == 1].value_counts(sort = False)
    df_class["Died"] = df.Pclass[df.Survived == 0].value_counts(sort = False)
    ax3 = df_class.plot(kind = "bar", alpha = 0.5, rot = 1, ax = axes[0,2], color = ['g', 'darkblue'])
    ax3.set_title("Survival by Class")

    #Survival by Age (groups of 10 years)
    filter_values = [0, 10, 20, 30, 40, 50, 60, 70]
    df_age = pd.DataFrame(columns = ["Survived", "Died"])
    df_age["Survived"] = df.Age[df.Survived == 1].value_counts(sort = False, bins = filter_values)
    df_age["Died"] = df.Age[df.Survived == 0].value_counts(sort = False, bins = filter_values)
    ax4 = df_age.plot(kind = "bar", alpha = 0.5, rot = 1, ax = axes[1,0], color = ['g', 'darkblue'])
    ax4.set_title("Survival by Age")
    ax4.set_xticklabels(["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70"], rotation=0)

    #survival by number of family members
    df_family = pd.DataFrame(columns = ["Survived", "Died"])
    df_family["Survived"] = df.Numfamily[df.Survived == 1].value_counts(sort = False)
    df_family["Died"] = df.Numfamily[df.Survived == 0].value_counts(sort = False)
    ax5 = df_family.plot(kind = "bar", alpha = 0.5, rot = 1, ax = axes[1,1], color = ['g', 'darkblue'])
    ax5.set_title("Survival by Number of Family")

    #Survival by title
    df_title = pd.DataFrame(columns = ["Survived", "Died"])
    df_title["Survived"] = df.Title[df.Survived == 1].value_counts(sort = False)
    df_title["Died"] = df.Title[df.Survived == 0].value_counts(sort = False)
    ax6 = df_title.plot(kind = "bar", alpha = 0.5, ax = axes[1,2], color = ['g', 'darkblue'])
    plt.title("Survival by Name Title")
    ax6.set_xticklabels(["Unknown", "RegMale", "Military", "HighMale", "RegFemale", "HighFemale"], rotation=45)

    fig.savefig('survival_by_feature.jpg')

#-------------------------------_END HELPER FUNCTIONS----------------------------
#--------------------------------------------------------------------------------

#read training data into dataframe, ignore first row since its the header row
train = pd.read_csv('train.csv', header = 0)
train = modify_dataframe(train)
visualize_dataframe(train)

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
