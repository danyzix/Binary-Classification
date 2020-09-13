import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek

class Classification:
    '''grid searches among classifires,
       compares the accuracies of the classifiers,
       plots the ROC curves
    '''    
    def __init__(self, xtrain, ytrain, xtest, ytest):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.scaler = StandardScaler()
        
    def update_features(self, features):
        self.xtrain = xtrain.loc[:, features]
        self.xtest = xtest.loc[:, features]
        
    def resample(self, sampling_type):
        os = RandomOverSampler(random_state=10)
        nm = NearMiss()
        smote = SMOTETomek(random_state=10)
        if sampling_type == 'over':
            self.xtrain, self.ytrain = os.fit_sample(self.xtrain, self.ytrain)
        elif sampling_type == 'under':
            self.xtrain, self.ytrain = nm.fit_sample(self.xtrain, self.ytrain)
        else:
            self.xtrain, self.ytrain = smote.fit_sample(self.xtrain, self.ytrain)
    
    def train_model(self, classifier, score):
        pipeline = Pipeline(steps = [('scaler', self.scaler), ('classifier', self.classifiers[classifier])])
        gs = GridSearchCV(pipeline, param_grid=self.parameters[classifier], n_jobs=-1, scoring=score)
        gs.fit(self.xtrain, self.ytrain)    
        return gs
    
    def grid_search(self, classifiers, parameters, score='accuracy'):
        self.classifiers = classifiers
        self.parameters = parameters
        result = {}
        pbar = tqdm(total=len(self.classifiers))
        for classifier in self.classifiers:
            model = self.train_model(classifier, score)
            result.update({classifier: {'model': model,
                                         'train_score': model.score(self.xtrain, self.ytrain),
                                         'crossValidaton_score': model.best_score_ ,
                                         'test_score': model.score(self.xtest, self.ytest),
                                         'refit_time': model.refit_time_
                                         }})
            pbar.update(1)
        pbar.refresh()
        self.result = result
        return result
    
    def model_comparison(self):
        model_comparison = pd.DataFrame()
        for a in self.result:
            model_comparison = model_comparison.append(pd.Series([a, self.result[a]['train_score'], 
                               self.result[a]['crossValidaton_score'], self.result[a]['test_score'], 
                               self.result[a]['refit_time']]), ignore_index=True)
        model_comparison.columns = ['model', 'train_score', 'crossVal_mean_score', 'test_score', 'refit_time']
        model_comparison = model_comparison.round(3)
        return model_comparison  
    
    def plot_rocs(self):
        # Comparison of each of the best models of the classifiers based on AUC
        plt.figure(figsize=(8, 6))
        for classifier in self.classifiers:
            ypred_prob = self.result[classifier]['model'].best_estimator_.predict_proba(self.xtest)
            fpr, tpr, thresh = roc_curve(self.ytest, ypred_prob[:, 1])
            auc = round(roc_auc_score(self.ytest,ypred_prob[:, 1]), 3)
            plt.plot(fpr, tpr, label=f'{classifier}, auc={auc}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate', fontsize=15)
        plt.ylabel('True Positive Rate', fontsize=15)
        plt.legend(fontsize=14)
        plt.show()  
        
    def predict(self, model):
        # baseline model selection based on the table above
        model = self.result[model]['model'].best_estimator_
        # predict on the test data
        ypred = model.predict(self.xtest)
        return ypred
        
    def print_report(self, ypred):
        # print the confusion matrix
        print(f'Confusion Matrix:\n{confusion_matrix(self.ytest, ypred)}')
        # print the classification report
        print(f'Report:\n{classification_report(self.ytest, ypred)}')