import pandas as pd


#reading files survey_2016.csv and survey_2014.csv
data_2016 = pd.read_csv("C:/Users/Ghajaanan Jeyakumara/Desktop/mental health/survey_2016.csv")
data_2014 = pd.read_csv("C:/Users/Ghajaanan Jeyakumara/Desktop/mental health/survey_2014.csv")

#prints the number of rows and column in data_2016 and data_2014 dataframes
#print(data_2016.shape)
#print(data_2014.shape)

#prints the common columns in data_2016 and data_2014
#print(data_2014.columns&data_2016.columns)

#prints the number of common columns in data_2016 and data_2014
#print(len(data_2016.columns&data_2014.columns))


#joins the two dataframes using 'inner join' i.e. using the columns present in both data frames
#ignore_index=True is used to provide a sequential index i.e from 0.... 2693 rather than 0....1432 and 0....1259
#stores it in a new data frame

result = pd.concat([data_2014,data_2016],ignore_index=True,join='inner')



#converts the data frame in to a csv file for further use 
#result.to_csv("C:/Users/Ghajaanan Jeyakumara/Desktop/mental health/result.csv")


#frame = pd.read_csv("C:/Users/Ghajaanan Jeyakumara/Desktop/mental health/result.csv")
#print(result.head())
#removing the column which will not be needed for our analysis
#by default axis=0 i.e. rows , axis=1 is columns

result = result.drop(['country','state'],axis=1)

#prints the number of rows and columns in result
#print(result.shape)

#iterates through each column and checks for null values, if there are null values, provides the sum of null values in each column
#print(pd.isnull(result).sum())

#create lists of features by data type
intFeatures=['age']
strFeatures=['gender','country','self_employed','family_history','treatment',
             'work_interfere','num_employees','work_remote','tech_company','benefits',
             'care_options','employee_wellness_ program','seek_help','anonymity_protected',
             'leave','mental_health_consequences','physical_health_consequences','coworkers',
             'supervisors','men_health_interview','phy_health_interview','mental_vs_physical',
             'observed_consequence']

#default values to substitute for missing values
intDefault=0
strDefault='NaN'


#filling NA vlues i.e.missing values with default values
for feature in result:
    if feature in intFeatures:
        result[feature]=result[feature].fillna(intDefault)
    elif feature in strFeatures:
        result[feature]=result[feature].fillna(strDefault)
    else:
        print("feature %s not found",feature)
        
        
#substitute missing age with median
result['age'].fillna(result['age'].median(),inplace=True)

#substitute age column's median for age values <18 and >80 (cuz it doesn't make sense)
array = pd.Series(result['age'])
array[array<18] = result['age'].median()
result['age'] = array
array = pd.Series(result['age'])
array[array>80] = result['age'].median()
result['age'] = array 

#creating a new column to categorise age_range
result['age_range'] = pd.cut(result['age'], [0,20,30,65,80], labels=["0-20", "21-30", "31-65","65-80"], include_lowest=True)

#changing all gender values to lowercase for ease of replacement
gender = result['gender'].str.lower()

#selecting unique entries
gender = gender.unique()
#print(gender)
#print(len(gender))

#categorise gender
male_list=['m','male','male-ish','maile','something kinda male?','cis male','mal','male (cis)','make','guy (-ish) ^_^','male ','man','msle','mail','malr','cis man',
      'ostensibly male, unsure what that really means','male.','nb masculine','sex is male','dude',"i'm a man why didn't you make this a drop down question. you should of asked sex? and i would of answered yes please. seriously how much text can this take? ",'m|','cisdude']
female_list=['female','cis female','f','woman','femake','female ','cis-female/femme','female (cis)','femail','i identify as female.','female assigned at birth ','fm','cis female ','female or multi-gender femme','female/woman','cisgender female','fem','female (props for making this a freeform field, though)',' female','cis-woman','female-bodied; no feelings about gender']
lgbtq_list=['trans-female','queer/she/they','non-binary','fluid','all','genderqueer','androgyne','agender','male leaning androgynous','trans woman','neuter','female (trans)','queer','bigender','transitioned, m2f','genderfluid (born female)','other/transfeminine','androgynous','male 9:1 female, roughly','genderfluid','genderqueer woman','mtf','male/genderqueer','nonbinary','unicorn','male (trans, ftm)','genderflux demi-girl','transgender woman']
other_list=['nah','enby','a little about you','p','nan','other','none of your business','human','afab']
    
# iterrows() function helps loop through each row of a dataframe
#we are generalising gender for ease
for (row, col) in result.iterrows():

    if str.lower(col.gender) in male_list:
        result['gender'].replace(to_replace=col.gender, value='male', inplace=True)

    if str.lower(col.gender) in female_list:
        result['gender'].replace(to_replace=col.gender, value='female', inplace=True)

    if str.lower(col.gender) in lgbtq_list:
        result['gender'].replace(to_replace=col.gender, value='lgbtq', inplace=True)
    if str.lower(col.gender) in other_list:
        result['gender'].replace(to_replace=col.gender,value='other', inplace=True)

#to check if gender values are generalised
#print("gender:",result['gender'].unique())


#since self_employed is very less, replace NaN values with No
result['self_employed'] = result['self_employed'].replace(strDefault, 'No')
#then replcae all 1 and 0 to yes and no
result['self_employed']=result['self_employed'].replace((0,1),('No','Yes'))


#NaN values in work_interfere are replaced with Don't know
result['work_interfere'] = result['work_interfere'].replace(strDefault,'Don\'t know')

#in work_remote, substituting sometimes and always to yes, never with no so that data_20154 and data_2016 dataframes have common values
result['work_remote'] = result['work_remote'].replace(('Sometimes','Always'),'Yes')
result['work_remote'] = result['work_remote'].replace(('Never'),'No')

#replcae all 1 and 0 to yes and no in tech_company
result['tech_company']=result['tech_company'].replace((0.0,1.0),('No','Yes'))

#replace Don't know with I don't know and Not eligible for coverage with No since they mean the same thing
result['benefits'] = result['benefits'].replace(('Don\'t know'),'I don\'t know')
result['benefits'] = result['benefits'].replace('Not eligible for coverage / N/A','No')

#replace I am not sure to Not sure since they mean the same thing
result['care_options'] = result['care_options'].replace(('I am not sure'),'Not sure')

#replace Don't know with I don't know since they mean the same thing
result['employee_wellness_ program'] = result['employee_wellness_ program'].replace(('Don\'t know'),'I don\'t know')

result['seek_help'] = result['seek_help'].replace(('Don\'t know'),'I don\'t know')

result['anonymity_protected'] = result['anonymity_protected'].replace(('Don\'t know'),'I don\'t know')

result['leave'] = result['leave'].replace(('Don\'t know'),'I don\'t know')

result['mental_vs_physical'] = result['mental_vs_physical'].replace('Don\'t know','I don\'t know')

#even after rectifying anomalies, there were NaN values present
#dropping rows with NaN values ... there were NaN values only in 7 columns. So chose 1 out of 7 and removed rows
no_nan=result[result['observed_consequence']!='NaN']
#print(no_nan.shape)

#saving the file cleaned data frame as a csv file named "clean_data.csv"
no_nan.to_csv("C:/Users/Ghajaanan Jeyakumara/Desktop/mental health/clean_data.csv")
