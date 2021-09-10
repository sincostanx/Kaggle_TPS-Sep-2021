import argparse
import os
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import torch
import pytorch_tabnet
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

def splitTrainVal(train, trainp, test, fold, target):
    train_indices = train[train.Set!=fold].index
    valid_indices = train[train.Set==fold].index
    
    if not trainp.empty:
        trainp_indices = trainp[trainp.Set!=fold].index
        x_train = np.concatenate([train[features].values[train_indices], trainp[features].values[trainp_indices]], axis=0)
        y_train = np.concatenate([train[target].values[train_indices], trainp[target].values[trainp_indices]], axis=0)
    else:
        x_train = train[features].values[train_indices]
        y_train = train[target].values[train_indices]
    
    x_val = train[features].values[valid_indices]
    y_val = train[target].values[valid_indices]
    
    x_train = np.asarray(x_train, dtype=np.int64)
    y_train = np.asarray(y_train, dtype=np.int64)
    x_val = np.asarray(x_val, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)
    
    x_test = test[features].to_numpy(dtype=np.int64)        
    
    print("X train shape: ", x_train.shape)
    print("X validation shape: ", x_val.shape)
    print("X test shape: ", x_test.shape)
    print("Y train shape: ", y_train.shape)
    print("Y validation shape: ", y_val.shape)
    print("number of zero label in train data: ", np.sum(y_train)/float(x_train.shape[0]))
    print("number of zero label in validation data: ", np.sum(y_val)/float(x_val.shape[0]))
    
    return x_train, y_train, x_val, y_val, x_test, valid_indices

def getPretrainerParam(cat_idxs, cat_dims):
    return 

def getTabnetParam(cat_idxs, cat_dims):
    trainer_param = dict(
                    cat_idxs=cat_idxs,
                    cat_dims=cat_dims,
                    cat_emb_dim=2,
                    n_d = 16,
                    n_a = 16,
                    gamma = 1.3,
                    n_steps = 3,
                    lambda_sparse = 1e-3,
                    optimizer_fn=torch.optim.AdamW,
                    optimizer_params=dict(lr=1e-3),
                    mask_type='entmax' # "sparsemax"
                )
    
    tabnet_param = dict(
                   cat_idxs=cat_idxs,
                   cat_dims=cat_dims,
                   cat_emb_dim=2,
                   n_d = 16,
                   n_a = 16,
                   gamma = 1.3,
                   n_steps = 3,
                   lambda_sparse = 1e-3,
                   optimizer_fn=torch.optim.AdamW,
                   optimizer_params=dict(lr=1e-3, weight_decay = 1e-2),
                   scheduler_params={"step_size":5, # how to use learning rate scheduler
                                     "gamma":0.9},
                   scheduler_fn=torch.optim.lr_scheduler.StepLR,
                   mask_type='entmax' # "sparsemax"
               )
    return trainer_param, tabnet_param

def trainTabnet(train, test, args, trainp = pd.DataFrame()):
    cat_idxs = [i for i in range(len(features))]
    cat_dims = [4096 for i in range(len(features))]
    
    train_pred = np.zeros((train.shape[0], 1))
    test_pred = np.zeros((test.shape[0], args.fold))
    
    for i in range(args.fold):
        print("################# START FOLD ", i+1, "#################")
        x_train, y_train, x_val, y_val, x_test, valid_indices = splitTrainVal(train, trainp, test, i, target)
        
        saving_path_name = "Tabnet" + str(i) + ".zip" if trainp.empty else "Tabnet-pseudo" + str(i) + ".zip"
        
        if not args.load_model:
            trainer_param, tabnet_param = getTabnetParam(cat_idxs, cat_dims)
            
            if args.tabnet_pretrainer:
                unsupervised_model = TabNetPretrainer(**trainer_param)
                unsupervised_model.fit(
                    train[x_train],
                    eval_set=[x_val],
                    pretraining_ratio=0.8,
                    batch_size=1024, virtual_batch_size=128,
                    max_epochs = 100,
                )
                saving_path_name = "Pretrainer" + str(i)
                unsupervised_model.save_model(saving_path_name)
            else: unsupervised_model = None
            
            model = TabNetClassifier(**tabnet_param)
            model.fit(
                x_train,y_train,
                eval_set=[(x_train, y_train), (x_val, y_val)],
                eval_name=['train', 'valid'],
                eval_metric=['auc'],
                max_epochs=50 , patience=5,
                loss_fn=torch.nn.functional.cross_entropy,
                batch_size=16192, virtual_batch_size=1024,
                from_unsupervised=unsupervised_model,
                weights=1,
                drop_last=False
            )
            model.save_model(saving_path_name)
        
        else:
            model = TabNetClassifier()
            model.load_model(saving_path_name)
        
        train_pred[valid_indices, :] = model.predict_proba(x_val)[:, 1].reshape(-1, 1)
        test_pred[:, i] = model.predict_proba(x_test)[:, 1]
    
    test_pred_mean = np.mean(test_pred, axis=1).reshape(-1, 1)
    print(test_pred)
    print(test_pred_mean)
    print("Total AUC score : ", roc_auc_score(train[target].to_numpy(), train_pred))
    
    return train_pred, test_pred_mean

def getXGBParam():
    return {
              'tree_method' : 'gpu_hist', 
              'learning_rate' : 0.01,
              'n_estimators' : 50000,
              'colsample_bytree' : 0.3,
              'subsample' : 0.75,
              'reg_alpha' : 19,
              'reg_lambda' : 19,
              'max_depth' : 5, 
              'predictor' : 'gpu_predictor'
          }

def trainXGB(train, test, args, trainp = pd.DataFrame()):
    train_pred = np.zeros((train.shape[0], 1))
    test_pred = np.zeros((test.shape[0], args.fold))
    
    for i in range(args.fold):
        
        print("################# START FOLD ", i+1, "#################")
        
        x_train, y_train, x_val, y_val, x_test, valid_indices = splitTrainVal(train, trainp, test, i, target) 
        
        saving_path_name = "xgb" + str(i) + ".json" if trainp.empty else "xgb-pseudo" + str(i) + ".json"
        
        if not args.load_model:
            xgb_params = getXGBParam()
            model_xgb = XGBClassifier(**xgb_params)
            model_xgb.fit(x_train, y_train,
                    eval_set=[(x_val, y_val)],
                    early_stopping_rounds=200,
                    verbose=500, eval_metric = 'auc')
            model_xgb.save_model(saving_path_name)
        else:
            model_xgb = XGBClassifier()
            model_xgb.load_model(saving_path_name)
        
        train_pred[valid_indices, :] = model_xgb.predict_proba(x_val, ntree_limit=model_xgb.best_ntree_limit)[:, 1].reshape(-1,1)
        test_pred[:, i] = model_xgb.predict_proba(x_test, ntree_limit=model_xgb.best_ntree_limit)[:, 1]
        
    test_pred_mean = np.mean(test_pred, axis=1).reshape(-1, 1)
    print(test_pred)
    print(test_pred_mean)
    print("Total AUC score : ", roc_auc_score(train[target].to_numpy(), train_pred))
    
    return train_pred, test_pred_mean

def execute(args):
    
    np.random.seed(0)
    
    ###################################### Read dataset #########################################
    train = pd.read_csv(args.train_path, index_col=False)
    test = pd.read_csv(args.test_path, index_col=False)
    train["Set"] = np.random.choice(args.fold, p=[1./args.fold]*args.fold, size=(train.shape[0],))

    ###################################### Feature Engineering & Handle missing data #########################################
    """
    This method will cause data leakage and degrade model's generalization capability. However, that is not a problem
    for Kaggle competitions since all testing data are provided already.
    
    In practical, these two lines should be used instead:
    
    train[column_X] = pipe.fit_transform(train[column_X])
    test[column_X] = pipe.transform(test[column_X])
    """
    
    train['n_missing'] = train.isna().sum(axis=1)
    test['n_missing'] = test.isna().sum(axis=1)

    column_X = [col for col in train.columns if col not in ['claim', 'Set', 'n_missing']]
    
    pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median',missing_values=np.nan)),
            ("scaler", StandardScaler())
            ])  
    pipe.fit(pd.concat([train[column_X], test[column_X]]))
    train[column_X] = pipe.transform(train[column_X])
    test[column_X] = pipe.transform(test[column_X])
    
    #Perform discretization to enable categorical embedding in TabNet only
    train_transform = train.copy()
    test_transform = test.copy()
    
    discretizer = KBinsDiscretizer(n_bins=4096, encode='ordinal',strategy='uniform')
    discretizer.fit(pd.concat([train_transform[column_X], test_transform[column_X]]))
    train_transform[column_X] = discretizer.transform(train_transform[column_X])
    test_transform[column_X] = discretizer.transform(test_transform[column_X])

    ###################################### Train/Eval models #########################################
    
    
    global target, unused_feat, features
    
    target = 'claim'
    unused_feat = ['Set']    
    features = [col for col in train.columns if col not in unused_feat+[target]]
    
    tabnet_train, tabnet_test = trainTabnet(train_transform, test_transform, args)
    
    if not args.stack:
        test_pred = np.squeeze(tabnet_test)
    else:
        xgb_train, xgb_test = trainXGB(train, test, args)
        
        stack_x_train = np.concatenate((train[features].to_numpy(), tabnet_train, xgb_train), axis = 1)
        stack_x_test = np.concatenate((test[features].to_numpy(), tabnet_test, xgb_test), axis = 1)
        
        stk = StratifiedKFold(n_splits = args.fold)
        
        test_pred = 0
        fold = 1
        total_auc = 0
        
        y = train[target].to_numpy()
        
        for train_index, valid_index in stk.split(stack_x_train, y):
            x_train, y_train = stack_x_train[train_index], y[train_index]
            x_valid, y_valid = stack_x_train[valid_index], y[valid_index]
            
            lr = LogisticRegression(n_jobs = -1, random_state = 0, C = 1000, max_iter = 1000)
            lr.fit(x_train, y_train)
            
            valid_pred = lr.predict_proba(x_valid)[:, 1]
            test_pred += (1./args.fold)*lr.predict_proba(stack_x_test)[:, 1]
            auc = roc_auc_score(y_valid, valid_pred)
            total_auc += (1./args.fold)*auc
            print('Fold', fold, 'AUC :', auc)
            fold += 1
            
        print('Total AUC score :', total_auc)
    
    submission = pd.read_csv('sample_solution.csv')
    submission['claim'] = test_pred
    submission.to_csv('submission.csv', index = 0)
    
    ###################################### Train/Eval w/ Pseudolabels #########################################
    
    if args.pseudolabel:
        
        test_pred = np.where(test_pred < 0.5, 0, 1)
        print(test_pred)
        fold_label = np.random.choice(args.fold, p=[1./args.fold]*args.fold, size=(test.shape[0],))
        
        trainp = test.copy()
        trainp['claim'] = test_pred
        trainp['Set'] = fold_label
        trainp = trainp[train.columns.tolist()]
        
        trainp_transform = test_transform.copy()
        trainp_transform['claim'] = test_pred
        trainp_transform['Set'] = fold_label
        trainp_transform = trainp_transform[train_transform.columns.tolist()]
        
        tabnet_train, tabnet_test = trainTabnet(train_transform, test_transform, args, trainp_transform)
    
        if not args.stack:
            test_pred = np.squeeze(tabnet_test)
        else:
            xgb_train, xgb_test = trainXGB(train, test, args, trainp)
            
            stack_x_train = np.concatenate((train[features].to_numpy(), tabnet_train, xgb_train), axis = 1)
            stack_x_test = np.concatenate((test[features].to_numpy(), tabnet_test, xgb_test), axis = 1)
            
            stk = StratifiedKFold(n_splits = args.fold)
            
            test_pred = 0
            fold = 1
            total_auc = 0
            
            y = train[target].to_numpy()
            
            for train_index, valid_index in stk.split(stack_x_train, y):
                x_train, y_train = stack_x_train[train_index], y[train_index]
                x_valid, y_valid = stack_x_train[valid_index], y[valid_index]
                
                lr = LogisticRegression(n_jobs = -1, random_state = 0, C = 1000, max_iter = 1000)
                lr.fit(x_train, y_train)
                
                valid_pred = lr.predict_proba(x_valid)[:, 1]
                test_pred += (1./args.fold)*lr.predict_proba(stack_x_test)[:, 1]
                auc = roc_auc_score(y_valid, valid_pred)
                total_auc += (1./args.fold)*auc
                print('Fold', fold, 'AUC :', auc)
                fold += 1
                
            print('Total AUC score :', total_auc)
    
    ###################################### Write predictions #########################################
    submission = pd.read_csv('sample_solution.csv')
    submission['claim'] = test_pred
    submission.to_csv('submission-pseudolabel.csv', index = 0)

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Training script. Default values of all arguments are recommended for reproducibility",
        fromfile_prefix_chars="@",
        conflict_handler="resolve",
    )
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument("--train_path", default="train.csv", type=str, help="path to training data")
    parser.add_argument("--test_path", default="test.csv", type=str, help="path to testing data")
    parser.add_argument("--fold", default=5, type=int, help="number of fold")
    parser.add_argument("--tabnet_pretrainer", default=False, help="if set, will use pretrainer for tabnet", action="store_true")
    parser.add_argument("--load_model", default=False, help="if set, will skip training", action="store_true")
    parser.add_argument("--stack", default=True, help="if set, will use stack generalization with XGBoost and Logistic Regression", action="store_true")
    parser.add_argument("--pseudolabel", default=False, help="if set, will use pseudolabeling", action="store_true")
    
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = "@" + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()
        
    execute(args)