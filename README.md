#Kaggle Titanic Competition: Machine Learning from Disaster

#Overview
In this program, we predict the survival of Titanic passengers based on a database of information provided by Kaggle, a  data science site that hosts machine learning competitions. As a popular starting competition for most beginners, it provides highly structured data with many introductory tutorials; however, beyond the tutorials users may create more advanced models based on complex data modeling techniques. I decided to employ the Logistic Regression technique as opposed to the recommended Random Forest, since my current understanding of predictive models is thus far limited to simpler regression models. In any case, it was found that the Logistic Regression performed almost as well as a Random Forest does, thereby proving that simpler models can be just as effective as the more complicated ones. <br> <br>
The link to the competition may be found here: https://www.kaggle.com/c/titanic

#Usage
Usage of the program is simply a matter of calling python on the predict.py script. It uses Kaggle's train.csv to train the Logistic Regression model, and a test.csv that witholds the survival status of each passenger to test the model. The output is a .csv file that contains a passengerId and the predicted result of that passenger, which is used to tentatively score the model in a public leaderboard, and a .jpg file containing the survival by each feature I chose to be significant.

#Notes about the program
This project served as an introduction into machine learning and Python's pandas and sklearn packages, and as such I decided to go with a simpler predictive model. However, as found throughout the forums, the Logistic Regression is capable of performing just as well (if not better) compared to the Random Forest. It was actually easy to swap in a Random Forest algorithm due to the simplicity of the sklearn package in Python; instead of calling predictor = LogisticRegression(), I could use predictor = RandomForestClassifier(n_estimators = 100) with no effect on the other code to employ the different model. Surprisingly, the Random Forest performed worse anyways, so I stuck to the simpler model since I had a better understanding of it. 
<br><br>
The first step was to determine which features were meaningful from the given data, and what new features we could engineer from all the data. The attributes given in the training data were: <br>
Survival, PassengerClass, Name, Sex, #Sibling/Spouses, #Parents/Children, ticketNum, Fare, Cabin, EmbarkLocation 
<br><br>
From intuition, I decided that the important features to be found are:
<ul>
  <li>Gender/Age. The Titanic tragedy was widely known for saving women and children first, and upon examination of the data it was indeed found that a high proportion of women survived compared to men.</li>
  <li>Passenger Class. Higher classes were given priority to the lifeboats, and I decided that the fare was redundant given this information. Since the fare is proportional to the Class anyway.</li>
  <li>Number of Family Aboard. The less family a passenger had aboard, the more likely I assume they were to survive; conversely, if a passsenger had many family members, they may spend crucial time saving other loved ones and staying together. To track this feature, I added the results of Sibling/Spouses and Parents/Children together to create a new feature column.</li>
  <li>Title of a person's name. Surprisingly, I was able to extract important information from the name of a passenger. This is because the "title" of a passenger is also included, ie. Mr, Mrs, Dr, Major, Capt, Don, Master, etc. I decided to group these titles into 6 main categories, being Military, Prestigious Female, Regular Female, Prestigious Male, Regular Male, and an Unknown title group. I used dummy variables to indicate which of the 6 groups a passenger was in (denoted by a 1 or a 0), while ommitting a one of them to avoid the dummy variable trap. </li>
</ul> 

We can affirm these intuitions by plotting these features vs. the number of survivals/deaths for each. This generated the following figure: <br>
![alt tag](https://github.com/jteo1/TitanicPredictingSurvival/blob/master/survival_by_feature.jpg)
Indeed we find these intuitions to be relatively correct. The proportion of female survivals is much higher than that of the men, and the 1st and 2nd class passengers had a substantially higher survival rate than the 3rd class passengers. Also, we can see that children (age 0-10) were the only bucket to have more survivors that deaths. Interestingly, young to middle age passengers (age 20-30) appear to have the lowest survival rate, likely since they are composed of people who are too old to be considered children but too young to have high enough prestige to be considered priority passengers. From the survival by family graph, we find that small families of 2, 3, and 4 survived better than larger families (excluding familes of "1", which represents passengers traveling alone). In the last graph, we see that prestigious and regular titled women had very high survival rates, with regular males having astounding low survival rates.

Using Python's sklearn package, I was able to find the coefficients of the logistic regression as well. They were given to be (with the titles having labels that I manually appended): <br>
```
0     Pclass     [-1.0282680664] <br>
1        Sex     [1.62831294051] <br>
2        Age  [-0.0227492356786] <br>
3  Numfamily   [-0.283363260057] <br>
4    Title_0  [-0.0837148184562] (Unknown)<br>
5    Title_1   [-0.572020559236] (Regular Male)<br>
6    Title_2  [0.00278960533722] (Military)<br>
7    Title_3     [1.49157579596] (High Male)<br>
8    Title_4    [0.848931989972](Regular Female)<br>
```

With Title_5 (High Female) ommitted to avoid the aforementioned dummy variable trap. We can see that the Sex attribute had a strong effect on the result after mapping Male to the value 0 and Female to the value 1. Also, the Pclass had a strong inverse relationship, since the order of passenger class goes from 1st class, 2nd class, 3rd class. From Title_1, we see that the Regular males were likely to die, given a negative correlation. High Males and Regular Females had strong positive correlations, with High Female presumably have a strong positive correlation if it were included as well. Perhaps the military titles didn't have a high correlation due to time spent saving others first rather than all saving themselves.
<br><br>
In the end, I decided to ignore the ticket#, and EmbarkLocation columns, since intuitively these results results wouldn't have any bearing on their outcome. Unfortunately, I had to ignore the Cabin column, since one would presume that certain cabins were closer to the lifeboats and would provide some advantage to nearer passengers. However, around only 200/900 rows contained the cabin information.
<br><br>



#Results
The result on the test data for my model was 0.78947 on the public leaderboards. This result was actually quite decent; the top 5-10% scores were around 0.81, with score in the 0.9s presumed to be from cheaters that either found the true values or used larger datasets online to train their model. Also, given the presence of many highly experienced users on this website, I am thus satisfied with its performance. In any case, the current score is a result of the public leaderboard using their given test set; in the future, the private leaderboard score will contain a larger test set that provides a final official score when the competition concludes.
