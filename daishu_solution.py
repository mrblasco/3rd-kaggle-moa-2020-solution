import pandas as pd
import numpy as np
import warnings
import gc, os
from time import time
import datetime, random

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import QuantileTransformer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler
from torch.nn.modules.loss import _WeightedLoss

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import argparse
import json
import utils
import logging

# CNN
import model.daishu 
from model.daishu import SmoothBCEwLogits
from model.daishu import GBN
from model.daishu import GLU
from model.daishu import Model
from model.daishu import FeatureTransformer
from model.daishu import AttentionTransformer
from model.daishu import TabNet
from model.daishu import Self_Attention
from model.daishu import DecisionStep
from model.daishu import Attention_dnn
from model.daishu import Dnn


# Functions -------------------------------------------------------

args = argparse.ArgumentParser()
args.add_argument('--input_dir', default='./data/from_kaggle'
                , help='Directory containing dataset')
args.add_argument('--model_dir', default='./experiments/base_model'
                  , help='Directory containing params.json')

def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 

def Metric(labels, preds):
    labels = np.array(labels)
    preds = np.array(preds)
    metric = 0
    for i in range(labels.shape[1]):
        metric += (-np.mean(labels[:, i]*np.log(np.maximum(preds[:, i], 1e-15))+(1-labels[:, i])*np.log(np.maximum(1-preds[:, i], 1e-15))))
    return metric/labels.shape[1]

def Feature(df):
    transformers = {}
    for col in (genes+cells):
        transformer = QuantileTransformer(n_quantiles=100
                                        , random_state=0
                                        , output_distribution='normal')
        transformer.fit(df[:train.shape[0]][col].values.reshape(-1, 1))
        df[col] = transformer.transform(df[col].values.reshape(-1, 1)).reshape(1, -1)[0]
        transformers[col] = transformer
    gene_pca  = PCA(n_components = ncompo_genes, random_state = 42).fit(df[genes])
    pca_genes = gene_pca.transform(df[genes])
    cell_pca  = PCA(n_components = ncompo_cells, random_state = 42).fit(df[cells])
    pca_cells = cell_pca.transform(df[cells])
    pca_genes = pd.DataFrame(pca_genes, columns = [f"pca_g-{i}" for i in range(ncompo_genes)])
    pca_cells = pd.DataFrame(pca_cells, columns = [f"pca_c-{i}" for i in range(ncompo_cells)])
    df = pd.concat([df, pca_genes, pca_cells], axis = 1)
    for col in ['cp_time', 'cp_dose']:
        tmp = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, tmp], axis=1)
        df.drop([col], axis=1, inplace=True)
    return df, transformers, gene_pca, cell_pca

def Ctl_augment(train, target, train_nonscored):
    aug_trains = []
    aug_targets = []
    for _ in range(3):
        train1 = train.copy()
        target1 = target.copy()
        nonscore_target1 = train_nonscored.copy()
        ctl1 = ctl_train.sample(train1.shape[0], replace=True).reset_index(drop=True)
        ctl2 = ctl_train.sample(train1.shape[0], replace=True).reset_index(drop=True)
        ctl3 = ctl_train.sample(train1.shape[0], replace=True).reset_index(drop=True)
        ctl4 = ctl_train.sample(train1.shape[0], replace=True).reset_index(drop=True)
        mask_index1 = list(np.random.choice(ctl3.index.tolist(), int(ctl3.shape[0]*0.4), replace=False))
        ctl3.loc[mask_index1, genes+cells] = 0.0
        ctl4.loc[mask_index1, genes+cells] = 0.0
        ctl5 = ctl_train.sample(train1.shape[0], replace=True).reset_index(drop=True)
        ctl6 = ctl_train.sample(train1.shape[0], replace=True).reset_index(drop=True)
        mask_index2 = list(np.random.choice(list(set(ctl5.index)-set(mask_index1)), int(ctl5.shape[0]*0.3), replace=False))
        ctl5.loc[mask_index1+mask_index2, genes+cells] = 0.0
        ctl6.loc[mask_index1+mask_index2, genes+cells] = 0.0
        train1[genes+cells] = train1[genes+cells].values + ctl1[genes+cells].values - ctl2[genes+cells].values \
                              + ctl3[genes+cells].values - ctl4[genes+cells].values + ctl5[genes+cells].values - ctl6[genes+cells].values# * np.random.rand(train1.shape[0]).reshape(-1, 1)
        aug_train = train1.merge(target1, how='left', on='sig_id')
        aug_train = aug_train.merge(nonscore_target1, how='left', on='sig_id')
        aug_trains.append(aug_train[['cp_time', 'cp_dose']+genes+cells])
        aug_targets.append(aug_train[targets+nonscored_targets])
    df = pd.concat(aug_trains).reset_index(drop=True)
    target = pd.concat(aug_targets).reset_index(drop=True)
    for col in (genes+cells):
        df[col] = transformers[col].transform(df[col].values.reshape(-1, 1)).reshape(1, -1)[0]
    pca_genes = gene_pca.transform(df[genes])
    pca_cells = cell_pca.transform(df[cells])
    pca_genes = pd.DataFrame(pca_genes, columns = [f"pca_g-{i}" for i in range(ncompo_genes)])
    pca_cells = pd.DataFrame(pca_cells, columns = [f"pca_c-{i}" for i in range(ncompo_cells)])
    df = pd.concat([df, pca_genes, pca_cells], axis = 1)
    for col in ['cp_time', 'cp_dose']:
        tmp = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, tmp], axis=1)
        df.drop([col], axis=1, inplace=True)
    xs = df[train_cols].values
    ys = target[targets].values
    ys1 = target[nonscored_targets].values
    return xs, ys, ys1

def train_and_predict(features, sub, aug, mn, folds, seed):
    
    # Initialize arrays
    oof = train[['sig_id']]
    for t in targets:
        oof[t] = 0.0
    preds = []
    
    # Split test in batches 
    test_X = test[features].values
    test_data_loader = DataLoader(dataset=TensorDataset(torch.Tensor(test_X))
                                , batch_size = 128
                                , shuffle = False)
    
    # Run k fold 
    eval_train_loss = 0
    for fold, (trn_ind, val_ind) in enumerate(MultilabelStratifiedKFold(n_splits = folds, shuffle = True, random_state=seed)\
                                              .split(train, train_target[targets])):
        train_X = train.loc[trn_ind, features].values
        eval_train_X = train_X.copy()
        train_Y = train_target.loc[trn_ind, targets].values
        train_Y1 = train_nonscored.loc[trn_ind, nonscored_targets].values
        eval_train_Y  = train_Y.copy()
        eval_train_Y1 = train_Y1.copy()
        if aug:
            aug_X, aug_Y, aug_Y1 = Ctl_augment(ori_train.loc[trn_ind], train_target.loc[trn_ind], train_nonscored.loc[trn_ind])
            train_X = np.concatenate([train_X, aug_X], axis=0)
            train_Y = np.concatenate([train_Y, aug_Y], axis=0)
            train_Y1 = np.concatenate([train_Y1, aug_Y1], axis=0)
            del aug_X, aug_Y, aug_Y1
        valid_X = train.loc[val_ind, features].values
        valid_Y = train_target.loc[val_ind, targets].values
        
        eval_train_data_loader = DataLoader(dataset=TensorDataset(torch.Tensor(eval_train_X)
                                                                , torch.Tensor(eval_train_Y))
                                            , batch_size=128
                                            , shuffle=False
                                            , drop_last=False)
        train_data_loader = DataLoader(dataset=TensorDataset(torch.Tensor(train_X)
                                                            , torch.Tensor(train_Y)
                                                            , torch.Tensor(train_Y1))
                                      , batch_size=128
                                      , shuffle=True
                                      , drop_last=True)
        valid_data_loader = DataLoader(dataset=TensorDataset(torch.Tensor(valid_X)
                                                            , torch.Tensor(valid_Y))
                                      , batch_size=1024
                                      , shuffle=False)
        
        if mn == 'tabnet':
            model = TabNet(len(features), len(targets), len(nonscored_targets), n_d=128, n_a=256, n_shared=1, n_ind=1, n_steps=3, relax=2., vbs=128)
            optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=1e-3, weight_decay=1.00e-5/5, eps=1e-5)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                              max_lr=1/90.0/3, epochs=EPOCHS, steps_per_epoch=len(train_data_loader))
        elif mn == 'attention_dnn':
            model = Attention_dnn(len(features), len(targets), len(nonscored_targets), 256, 1500, 2, 0.3)
            optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=1e-3, weight_decay=1.00e-5/4.75, eps=1e-5)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                              max_lr=1/90.0/4, epochs=EPOCHS, steps_per_epoch=len(train_data_loader))
        else:
            model = Dnn(len(features), len(targets), len(nonscored_targets), 1500)
            optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=1e-3, weight_decay=1.00e-5/6, eps=1e-5)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                              max_lr=1/90.0/3.5*3, epochs=EPOCHS, steps_per_epoch=len(train_data_loader))
        model.to(device)
        
        loss_fn = nn.BCEWithLogitsLoss()
        loss_tr = SmoothBCEwLogits(smoothing = 0.001)
        best_valid_metric = 1e9
        not_improve_epochs = 0
        
        for epoch in range(EPOCHS):
            if epoch > 0 and aug:
                # Augmentation at every epochs 
                aug_X, aug_Y, aug_Y1 = Ctl_augment(ori_train.loc[trn_ind], train_target.loc[trn_ind], train_nonscored.loc[trn_ind])
                train_X = np.concatenate([eval_train_X, aug_X], axis=0)
                train_Y = np.concatenate([eval_train_Y, aug_Y], axis=0)
                train_Y1 = np.concatenate([eval_train_Y1, aug_Y1], axis=0)
                train_data_loader = DataLoader(dataset=TensorDataset(torch.Tensor(train_X)
                                                                    , torch.Tensor(train_Y)
                                                                    , torch.Tensor(train_Y1))
                                              , batch_size=128
                                              , shuffle=True
                                              , drop_last=True)
                del aug_X, aug_Y, aug_Y1
            # change batch size 
            if epoch > 19:
                train_data_loader = DataLoader(dataset=TensorDataset(torch.Tensor(train_X), torch.Tensor(train_Y), torch.Tensor(train_Y1))
                                              , batch_size=512, shuffle=True, drop_last=True)
            
            # train
            train_loss = 0.0
            train_num = 0
            for data in (train_data_loader):
                optimizer.zero_grad()
                x, y, y1 = [d.to(device) for d in data]
                outputs, outputs1 = model(x)
                loss1 = loss_tr(outputs, y)
                loss2 = loss_fn(outputs1, y1)
                loss = loss1*0.5 + loss2*0.5
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_num += x.shape[0]
                train_loss += (loss1.item()*x.shape[0])
            
            train_loss /= train_num
            
            # eval out-of-fold 
            model.eval()
            valid_loss = 0.0
            valid_num = 0
            valid_preds = []
            for data in (valid_data_loader):
                x, y = [d.to(device) for d in data]
                outputs, _ = model(x)
                valid_preds.extend(list(outputs.sigmoid().cpu().detach().numpy()))
                loss = loss_fn(outputs, y)
                valid_num += x.shape[0]
                valid_loss += (loss.item()*x.shape[0])
            valid_loss /= valid_num
            valid_mean = np.mean(valid_preds)
            
            # predictions on test dataset 
            t_preds = []
            for data in (test_data_loader):
                x = data[0].to(device)
                with torch.no_grad():
                    outputs, _ = model(x)
                t_preds.extend(list(outputs.sigmoid().cpu().detach().numpy()))
            pred_mean = np.mean(t_preds)
            
            # checkpoints 
            if valid_loss < best_valid_metric:
                torch.save(model.state_dict(), os.path.join(args.model_dir, 'model_%s_seed%s_fold%s.ckpt'%(mn, seed, fold)))
                not_improve_epochs = 0
                best_valid_metric = valid_loss
                logging.info('[epoch %s] lr: %.6f, train_loss: %.6f, valid_metric: %.6f, valid_mean:%.6f, pred_mean:%.6f'%(epoch, optimizer.param_groups[0]['lr'], train_loss, valid_loss, valid_mean, pred_mean))
            else:
                not_improve_epochs += 1
                logging.info('[epoch %s] lr: %.6f, train_loss: %.6f, valid_metric: %.6f, valid_mean:%.6f, pred_mean:%.6f, NIE +1 ---> %s'%(epoch, optimizer.param_groups[0]['lr'], train_loss, valid_loss, valid_mean, pred_mean, not_improve_epochs))
                if not_improve_epochs >= 50:
                    break
            model.train()
        
        # Load best model 
        state_dict = torch.load(os.path.join(args.model_dir, 'model_%s_seed%s_fold%s.ckpt'%(mn, seed, fold))
                              , torch.device("cuda" if torch.cuda.is_available() else "cpu") )
        model.load_state_dict(state_dict)
        model.eval()
        
        # Predictions in training
        train_preds = []
        for data in (eval_train_data_loader):
            x = data[0].to(device)
            with torch.no_grad():
                outputs, _ = model(x)
            train_preds.extend(list(outputs.sigmoid().cpu().detach().numpy()))
        train_loss = Metric(eval_train_Y, train_preds)
        eval_train_loss += train_loss
        print('eval_train_loss:', train_loss)
        
        # Predictions in out-of-fold 
        valid_preds = []
        for data in (valid_data_loader):
            x, y = [d.to(device) for d in data]
            with torch.no_grad():
                outputs, _ = model(x)
            valid_preds.extend(list(outputs.cpu().detach().numpy()))
        oof.loc[val_ind, targets] = 1 / (1+np.exp(-np.array(valid_preds)))
        
        # Predictions in testing 
        t_preds = []
        for data in (test_data_loader):
            x = data[0].to(device)
            with torch.no_grad():
                outputs, _ = model(x)
            t_preds.extend(list(outputs.sigmoid().cpu().detach().numpy()))
        print(np.mean(1 / (1+np.exp(-np.array(valid_preds)))), np.mean(t_preds))
        preds.append(t_preds)
        del train_X, train_Y, valid_X, valid_Y, train_data_loader, valid_data_loader
    
    # Format predictions to submission 
    sub[targets] = np.array(preds).mean(axis=0)
    print('eval_train_loss:', eval_train_loss/folds
                            , oof[targets].mean().mean()
                            , sub[targets].mean().mean())
    logging.info('valid_metric:%.6f'%Metric(train_target[targets].values
                                          , oof[targets].values))
    return oof, sub

# Main -------------------------------------------------------

args = args.parse_args()

# load parameters 
json_path = os.path.join(args.model_dir, 'params.json')   
assert os.path.isfile(json_path), "No json file found at {}".format(json_path)
params = utils.Params(json_path)

# set logger
utils.set_logger(os.path.join(args.model_dir, 'train.log'))

# Set hyperparameters
device        = ('cuda' if torch.cuda.is_available() else 'cpu')
ncompo_genes  = params.ncompo_genes #80
ncompo_cells  = params.ncompo_cells #10
EPOCHS        = params.num_epochs # 23
AUGMENT       = params.augmentation
SEEDS         = params.num_seeds

Seed_everything(seed=42)

files = ['%s/test_features_calibr.csv'%args.input_dir, 
         '%s/train_targets_scored.csv'%args.input_dir, 
         '%s/train_features.csv'%args.input_dir, 
         '%s/train_targets_nonscored.csv'%args.input_dir, 
         '%s/train_drug.csv'%args.input_dir, 
         '%s/sample_submission.csv'%args.input_dir]
 
logging.info("Loading the datasets from {}".format(args.input_dir))  

test            = pd.read_csv(files[0])
train_target    = pd.read_csv(files[1])
train           = pd.read_csv(files[2])
train_nonscored = pd.read_csv(files[3])
train_drug      = pd.read_csv(files[4])
#sub             = pd.read_csv(files[5])
sub             = test.copy(deep = True)

# keep datasets aligned by rows (axis = 0) 
(train_target, _)     = train_target.align(train, axis = 0, join = 'inner')
(train_nonscored, _)  = train_nonscored.align(train, axis = 0, join = 'inner')
(train_drug, _)       = train_drug.align(train, axis = 0, join = 'inner')

logging.info("Training dataset of size {} x {}".format(train.shape[0], train.shape[1]))
logging.info("Training targets of size {} x {}".format(train_target.shape[0], train_target.shape[1]))
logging.info("Testing dataset of size {} x {}".format(test.shape[0], test.shape[1]))

logging.info("Dropping selected features with low variability...")
drop_cols = ['g-513', 'g-370', 'g-707', 'g-300', 'g-130', 'g-375', 'g-161', 
       'g-191', 'g-376', 'g-176', 'g-477', 'g-719', 'g-449', 'g-204', 
       'g-595', 'g-310', 'g-276', 'g-399', 'g-438', 'g-537', 'g-582', 
       'g-608', 'g-56', 'g-579', 'g-45', 'g-252', 'g-12', 'g-343', 
       'g-737', 'g-571', 'g-555', 'g-506', 'g-299', 'g-715', 'g-239', 
       'g-654', 'g-746', 'g-436', 'g-650', 'g-326', 'g-630', 'g-465', 
       'g-487', 'g-290', 'g-714', 'g-452', 'g-227', 'g-170', 'g-520', 
       'g-467']+['g-54', 'g-87', 'g-111', 'g-184', 'g-237', 'g-302', 'g-305', 
       'g-313', 'g-348', 'g-399', 'g-450', 'g-453', 'g-461', 'g-490', 
       'g-497', 'g-550', 'g-555', 'g-584', 'g-592', 'g-682', 'g-692', 
       'g-707', 'g-748', 'g-751']
drop_cols = list(set(drop_cols))
train.drop(drop_cols, axis=1, inplace=True)
test.drop(drop_cols, axis=1, inplace=True)

genes = [col for col in train.columns if col.startswith("g-")]
cells = [col for col in train.columns if col.startswith("c-")]

features = genes + cells
targets = [col for col in train_target if col!='sig_id']
nonscored_targets = [col for col in train_nonscored if col!='sig_id']

# keep copy of original training dataset
ori_train = train.copy()

# controls in training and testing 
ctl_train = train.loc[train['cp_type']=='ctl_vehicle'].append(test.loc[test['cp_type']=='ctl_vehicle']).reset_index(drop=True)

logging.info("Processing features...")

tt = train.append(test).reset_index(drop=True)
tt, transformers, gene_pca, cell_pca = Feature(tt)
train = tt[:train.shape[0]]
test = tt[train.shape[0]:].reset_index(drop=True)

# remove controls in training
if 1:
    train_target    = train_target.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    train_nonscored = train_nonscored.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    train_drug      = train_drug.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    ori_train       = ori_train.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    train           = train.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)

train_cols = [col for col in train.columns if col not in ['sig_id', 'cp_type']]

model_names = ['attention_dnn', 'tabnet', 'dnn']

for model_name in model_names:
  logging.info("Model: {} ...".format(model_name))
  for seed in range(SEEDS):
    Seed_everything(seed)
    oof, sub = train_and_predict(features = train_cols
                                    , sub   = sub.copy()
                                    , aug   = AUGMENT
                                    , mn    = model_name
                                    , folds = params.num_folds
                                    , seed  = seed)
    
    valid_metric = Metric(train_target[targets].values, oof[targets].values)
    logging.info('oof mean:%.6f, sub mean:%.6f, valid metric:%.6f'%(oof[targets].mean().mean(), sub[targets].mean().mean(), valid_metric))
    
    sub.to_csv(os.path.join(args.model_dir, "preds_{}_seed{}.csv".format(mn, seed)), index=False)
    oof.to_csv(os.path.join(args.model_dir, "oof_{}_seed{}.csv".format(mn, seed)), index=False)
  
  logging.info("model {} done!".format(model_name))

logging.info("done!")


