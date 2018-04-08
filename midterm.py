import numpy as np
import xgboost as xgb
import pandas as pd
from ml_metrics import quadratic_weighted_kappa
from sklearn.metrics import confusion_matrix

# load data
print('Loading data...')
train = pd.read_csv("data/training.csv")
test = pd.read_csv("data/testing.csv")
print('Done.')
print(np.shape(train), np.shape(test))

# data cleaning
train_test = train.append(test)
train_test.fillna(-999, inplace=True)  # replace na with -999

train_test['Response'] = train_test['Response'].astype(int)

# simple feature engineering
train_test['Product_Info_2_c'] = train_test.Product_Info_2.str[0]
train_test['Product_Info_2_n'] = train_test.Product_Info_2.str[1]
train_test['BMI_Age'] = train_test['BMI'] * train_test['Ins_Age']
med_keyword_columns = train_test.columns[train_test.columns.str.startswith('Medical_Keyword_')]
train_test['Med_Keywords_Count'] = train_test[med_keyword_columns].sum(axis=1)

# factorize categorical variables after adding new feature
train_test['Product_Info_2'] = pd.factorize(train_test['Product_Info_2'])[0]
train_test['Product_Info_2_c'] = pd.factorize(train_test['Product_Info_2_c'])[0]
train_test['Product_Info_2_n'] = pd.factorize(train_test['Product_Info_2_n'])[0]

# split train and test set
train = train_test[train_test['Response'] > 0].copy()
test = train_test[train_test['Response'] < 1].copy()

# build model
print('Building model...')
train_matrix = xgb.DMatrix(train.drop(['Id','Response'], axis=1), train['Response'].values, missing=-999)
test_matrix = xgb.DMatrix(test.drop(['Id','Response'], axis=1), label=test['Response'].values, missing=-999)

param = {'max_depth':7, 'eta':0.05, 'silent':1, 'objective':'reg:linear','subsample':0.85,'colsample_bytree':0.3,'min_child_weight':360 }

gbt = xgb.train(param, train_matrix, 720)

print('Done.')
# get preds
train_preds = gbt.predict(train_matrix, ntree_limit=gbt.best_iteration)
print('Train score is:', quadratic_weighted_kappa(np.round(np.clip(train_preds, 1, 8)), train['Response']))

cm = confusion_matrix(train['Response'].values, np.round(np.clip(train_preds, 1, 8)).astype(int))
print("Confusion matrix: \n%s" % cm)
test_preds = gbt.predict(test_matrix, ntree_limit=gbt.best_iteration)

# result without optimization
out = pd.DataFrame({"Id": test['Id'].values, "Response": np.round(np.clip(test_preds, 1, 8)).astype(int)})
out = out.set_index('Id')
out.to_csv('xgb_submission.csv')
