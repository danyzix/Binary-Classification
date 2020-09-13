import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler

def predict_likelihood(df, label):
    # input data as a dataframe
    # label = label column name in the data to predict
        
    # encode the label and one-hot code categorical variables
    label_encoder = LabelEncoder()
    df[label] = label_encoder.fit_transform(df[label])
    df = pd.get_dummies(df, drop_first=True)
    
    features = ['age', 'euribor3m', 'campaign', 'housing_yes', 'pdays', 'loan_yes',
       'poutcome_success', 'cons.price.idx', 'marital_married',
       'education_high.school', 'education_university.degree', 'previous',
       'job_technician', 'marital_single', 'day_of_week_mon',
       'day_of_week_wed', 'day_of_week_tue', 'day_of_week_thu',
       'contact_telephone', 'default_unknown', 'job_blue-collar',
       'education_professional.course', 'education_basic.9y', 'job_management',
       'poutcome_nonexistent', 'job_services', 'job_retired',
       'education_unknown', 'education_basic.6y', 'job_self-employed',
       'job_entrepreneur', 'job_unemployed', 'month_oct', 'job_student',
       'job_housemaid', 'month_mar', 'month_may']
    
    # Lets split the data in training and test sets with stratification to keep the proportion of y classes even
    stratifier = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=10)
    for train_index, test_index in stratifier.split(df, df['y']):
        train_data = df.loc[train_index]
        test_data = df.loc[test_index]
       
    # feature select and scale the data
    ytrain = train_data[label]
    ytest = test_data[label]
    sc = StandardScaler()
    xtrain = sc.fit_transform(train_data.loc[:, features])
    xtest = sc.transform(test_data.loc[:, features])
    
    # over-sample training data
    os = RandomOverSampler(random_state=10)
    xtrain, ytrain = os.fit_sample(xtrain, ytrain)
    
    # train and predict
    clf = XGBClassifier(booster='gbtree', random_state=10)
    clf.fit(xtrain, ytrain)
    y_pred = clf.predict(xtest)
    y_prob = clf.predict_proba(xtest)[:, 1]
    return y_prob
    
if __name__ == '__main__':
    df = pd.read_csv('../../Data/bank-direct-marketing.csv', sep=';')
    print(predict_likelihood(df, 'y'))
   

      
