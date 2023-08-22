#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch
from torch.utils.data import SequentialSampler, DataLoader
from torch.utils.data import Dataset
from torch import nn
import numpy as np
import os, random, gc

from torch import optim
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score
from scipy.special import softmax
from collections import defaultdict



def split_df(df, folds_mapping, fold_id:int = 0):
    folds = df["patient_id"].map(folds_mapping)

    df_train = df[folds != fold_id]
    df_train = df_train[~df_train["target"].isnull()].reset_index(drop=True)

    df_valid = df[folds == fold_id]
    df_valid = df_valid[~df_valid["target"].isnull()].reset_index(drop=True)
    
    return df_train, df_valid

def create_folds_mapping(df, n_folds=5, random_state=42):
    folds_df = pd.DataFrame({"patient_id":df["patient_id"].unique()})
    folds_df["fold"] = -1

    for i, (_, test_index) in enumerate(KFold(n_splits=n_folds, 
            shuffle=True, random_state=random_state).split(folds_df)):
        folds_df.loc[test_index, "fold"] = i
    folds_mapping = folds_df.set_index(["patient_id"])["fold"]
    return folds_mapping


def smape1p_ind(A, F):
    val = 200 * np.abs(F - A) / (np.abs(A+1) + np.abs(F+1))
    return val

def smape1p(A, F):
    return smape1p_ind(A, F).mean()

def smape1p_opt(x):
    #return np.median(x)
    tgts = np.arange(0, 61)
    #tgts = [smape(x, val) for val in np.arange(0, 61)]
    scores = [smape1p(x, val) for val in tgts]
    return tgts[np.argmin(scores)]

def split_df(df, folds_mapping, fold_id:int = 0):
    folds = df["patient_id"].map(folds_mapping)

    df_train = df[folds != fold_id]
    df_train = df_train[~df_train["target"].isnull()].reset_index(drop=True)

    df_valid = df[folds == fold_id]
    df_valid = df_valid[~df_valid["target"].isnull()].reset_index(drop=True)
    
    return df_train, df_valid

def create_folds_mapping(df, n_folds=5, random_state=42):
    folds_df = pd.DataFrame({"patient_id":df["patient_id"].unique()})
    folds_df["fold"] = -1

    for i, (_, test_index) in enumerate(KFold(n_splits=n_folds, 
            shuffle=True, random_state=random_state).split(folds_df)):
        folds_df.loc[test_index, "fold"] = i
    folds_mapping = folds_df.set_index(["patient_id"])["fold"]
    return folds_mapping

def run_single_fit(model, df_train, df_valid, fold_id, seed, probs):
    if probs:
        p = model.fit_predict_proba(df_train, df_valid)
        p = pd.DataFrame(p, columns=[f"prob_{i}" for i in range(p.shape[1])]).reset_index(drop=True)
        res = pd.DataFrame({"seed":seed, "fold": fold_id, \
            "patient_id":df_valid["patient_id"], "visit_month":df_valid["visit_month"], \
            "target_month":df_valid["target_month"], "target_i":df_valid["target_i"], \
            "target":df_valid["target"]}).reset_index(drop=True)
        return pd.concat([res, p], axis=1)
    else:
        p = model.fit_predict(df_train, df_valid)
        return pd.DataFrame({"seed":seed, "fold": fold_id, \
            "patient_id":df_valid["patient_id"], "visit_month":df_valid["visit_month"], \
            "target_month":df_valid["target_month"], "target_i":df_valid["target_i"], \
            "target":df_valid["target"], "preds":p})

class BaseModel:
    def fit(self, df_train):
        raise "NotImplemented"

    def predict(self, df_valid):
        raise "NotImplemented"

    def predict_proba(self, df_valid):
        raise "NotImplemented"

    def fit_predict(self, df_train, df_valid):
        self.fit(df_train)
        return self.predict(df_valid)

    def fit_predict_proba(self, df_train, df_valid):
        self.fit(df_train)
        return self.predict_proba(df_valid)

    def cv(self, sample, sup_sample=None, n_folds=5, random_state=42):
        folds_mapping = create_folds_mapping(sample, n_folds, random_state)
                
        res = None
        for fold_id in sorted(folds_mapping.unique()):
            df_train, df_valid = split_df(sample, folds_mapping, fold_id)
            if sup_sample is not None:
                df_train = pd.concat([df_train, sup_sample], axis=0)
            p = self.fit_predict(df_train, df_valid)
            delta = pd.DataFrame({"fold": fold_id,  \
                    "patient_id":df_valid["patient_id"], "visit_month":df_valid["visit_month"], \
                    "target_month":df_valid["target_month"], "target_i":df_valid["target_i"], \
                    "target":df_valid["target"], "preds":p})
            res = pd.concat([res, delta], axis=0)

        return res

    def cvx(self, sample, sup_sample=None, n_runs=1, n_folds=5, random_state=42, probs=False):
        np.random.seed(random_state)
        seeds = np.random.randint(0, 1e6, n_runs)

        run_args = []
        for seed in seeds:
            folds_mapping = create_folds_mapping(sample, n_folds, seed)
            for fold_id in sorted(folds_mapping.unique()):
                df_train, df_valid = split_df(sample, folds_mapping, fold_id)
                if sup_sample is not None:
                    df_train = pd.concat([df_train, sup_sample], axis=0)
                run_args.append(dict(
                    df_train = df_train,
                    df_valid = df_valid,
                    fold_id = fold_id,
                    seed = seed,
                    probs = probs
                ))

        res = Parallel(-1)(delayed(run_single_fit)(self, **args) for args in run_args)
        #res = [run_single_fit(self, **args) for args in run_args]
        return pd.concat(res, axis=0)

    def loo(self, sample, sup_sample=None, probs=False, sample2=None):
        if sample2 is None:
            sample2 = sample
        run_args = []
        for patient_id in sample["patient_id"].unique():
            df_train = sample[sample["patient_id"] != patient_id]
            df_valid = sample2[sample2["patient_id"] == patient_id]
            if sup_sample is not None:
                df_train = pd.concat([df_train, sup_sample], axis=0)
            run_args.append(dict(
                df_train = df_train,
                df_valid = df_valid,
                fold_id = None,
                seed = None,
                probs=probs
            ))

        res = Parallel(-1)(delayed(run_single_fit)(self, **args) for args in run_args)
        return pd.concat(res, axis=0)

def print_cvx_summary(res_df):
    scores = res_df.groupby(["seed", "fold"]).apply(lambda x: smape1p(x["target"], x["preds"])).values
    print("# ", len(scores), " runs")
    #print("# 05   :      ", np.quantile(scores, 0.05))
    #print("# 25   :   ", np.quantile(scores, 0.25))
    print("# mean :", scores.mean())
    #print("# 75   :   ", np.quantile(scores, 0.75))
    #print("# 95   :      ", np.quantile(scores, 0.95))
    print("# ovrl :", smape1p(res_df["target"], res_df["preds"]))

def print_loo_summary(res_df):
    scores = res_df.groupby(["patient_id"]).apply(lambda x: smape1p(x["target"], x["preds"])).values
    print("# ", len(scores), " runs")
    #print("# 05   :      ", np.quantile(scores, 0.05))
    #print("# 25   :   ", np.quantile(scores, 0.25))
    print("# mean :", scores.mean())
    #print("# 75   :   ", np.quantile(scores, 0.75))
    #print("# 95   :      ", np.quantile(scores, 0.95))
    print("# ovrl :", smape1p(res_df["target"], res_df["preds"]))

def single_smape1p(preds, tgt):
    # .tile(A, reps) => construct an array by repeating A the number of times given by reps.
    # reps => the number of repetitions of A along each axis.
    x = np.tile(np.arange(preds.shape[1]), (preds.shape[0], 1))    
    # x.shape => (64,87)
    
    # x =>
    # array([[ 0,  1,  2, ..., 84, 85, 86],
    #        [ 0,  1,  2, ..., 84, 85, 86],
    #        ...,
    #        [ 0,  1,  2, ..., 84, 85, 86],
    #        [ 0,  1,  2, ..., 84, 85, 86]])
    
    
    x = np.abs(x - tgt) / (2 + x + tgt)
    
    # x =>
    # [[0.         0.33333333 0.5        ... 0.97674419 0.97701149 0.97727273]
    #  [0.         0.33333333 0.5        ... 0.97674419 0.97701149 0.97727273]
    #  ...
    #  [0.         0.33333333 0.5        ... 0.97674419 0.97701149 0.97727273]
    #  [0.         0.33333333 0.5        ... 0.97674419 0.97701149 0.97727273]]    
    
    # x * preds).sum(axis=1).shape => (64,)
    
    return (x * preds).sum(axis=1)

def opt_smape1p(preds):
    '''
    Symmetric Mean Absolute Percentage Error (SMAPE).
    '''
    # preds.shape => (64, 87)
    
    # single_smape1p(preds, 0).reshape(-1,1).shape => (64, 1)
    
    x = np.hstack([single_smape1p(preds, i).reshape(-1,1) for i in range(preds.shape[1])])
    # x.shape => (64, 87)
    # x[0] => [0.61426332 0.52453866 0.4698005  0.43519453 0.41392353 0.40219079 ... 0.8018691  0.8037691  0.80563325]
    
    # x.argmin(axis=1).shape => (64,)
    
    # x.argmin(axis=1) => [6 6 6 6 6 7 6 6 7 8 7 7 5 6 5 5 ... 7 6 7 7 7 7 7 7 8 8 8 8 6 6 6 6]
    
    return x.argmin(axis=1)

class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug, mode="train"):
        self.cfg = cfg
        self.mode = mode
        self.df = df.copy()
        self.features = df[cfg.features].values
        if self.mode != "test":
            self.targets = df[self.cfg.target_column].values.astype(np.float32)
        else:
            self.targets = np.zeros(len(df))

    def __getitem__(self, idx):
        features = self.features[idx]
        targets = self.targets[idx]
        
        feature_dict = {
            "input": torch.tensor(features),
            "target_norm": torch.tensor(targets),
        }
        return feature_dict

    def __len__(self):
        return len(self.df)


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        self.n_classes = cfg.n_classes
        self.cnn = nn.Sequential(*([
            nn.Linear(len(self.cfg.features), cfg.n_hidden),
            nn.LeakyReLU(),
            ] +
            [
            nn.Linear(cfg.n_hidden, cfg.n_hidden),
            nn.LeakyReLU(),
            ] * self.cfg.n_layers)
        )

        self.head = nn.Sequential(
            nn.Linear(cfg.n_hidden, self.n_classes),
            nn.LeakyReLU(),
        )

    def forward(self, batch):
        input = batch["input"].float()
        y = batch["target_norm"]
        x = input
        x = self.cnn(x)
        preds = self.head(x).squeeze(-1)
        loss = (torch.abs(y - preds) / (torch.abs(0.01 + y) + torch.abs(0.01 + preds))).mean()
        return {"loss": loss, "preds": preds, "target_norm": y}


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_train_dataloader(train_ds, cfg, verbose):
    train_dataloader = DataLoader(
        train_ds,
        sampler=None,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=cfg.tr_collate_fn,
        drop_last=cfg.drop_last,
        worker_init_fn=worker_init_fn,
    )
    if verbose:
        print(f"train: dataset {len(train_ds)}, dataloader {len(train_dataloader)}")
    return train_dataloader


def get_val_dataloader(val_ds, cfg, verbose):
    sampler = SequentialSampler(val_ds)
    if cfg.batch_size_val is not None:
        batch_size = cfg.batch_size_val
    else:
        batch_size = cfg.batch_size
    val_dataloader = DataLoader(
        val_ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=cfg.val_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    if verbose:
        print(f"valid: dataset {len(val_ds)}, dataloader {len(val_dataloader)}")
    return val_dataloader


def get_scheduler(cfg, optimizer, total_steps):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup * (total_steps // cfg.batch_size),
        num_training_steps=cfg.epochs * (total_steps // cfg.batch_size),
    )
    return scheduler


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True #  True my_
    torch.backends.cudnn.benchmark = False # False my_

def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict

def run_eval(model, val_dataloader, cfg, pre="val", verbose=True):
    model.eval()
    torch.set_grad_enabled(False)
    val_data = defaultdict(list)
    if verbose:
        progress_bar = tqdm(val_dataloader)
    else:
        progress_bar = val_dataloader
    for data in progress_bar:
        batch = batch_to_device(data, cfg.device)
        if cfg.mixed_precision:
            with autocast():
                output = model(batch)
        else:
            output = model(batch)
        for key, val in output.items():
            val_data[key] += [output[key]]
    for key, val in output.items():
        value = val_data[key]
        if len(value[0].shape) == 0:
            val_data[key] = torch.stack(value)
        else:
            val_data[key] = torch.cat(value, dim=0)

    preds = val_data["preds"].cpu().numpy()
    if (pre == "val") and verbose:
        metric = smape1p(100*val_data["target_norm"].cpu().numpy(), 100*preds)
        print(f"{pre}_metric 1 ", metric)
        metric = smape1p(100*val_data["target_norm"].cpu().numpy(), np.round(100*preds))
        print(f"{pre}_metric 2 ", metric)
    
    return 100*preds


def run_train(cfg, train_df, val_df, test_df=None, verbose=True):
    os.makedirs(str(cfg.output_dir + "/"), exist_ok=True)

    if cfg.seed < 0:
        cfg.seed = np.random.randint(1_000_000)
    if verbose:
        print("seed", cfg.seed)
    set_seed(cfg.seed)

    train_dataset = CustomDataset(train_df, cfg, aug=None, mode="train")
    train_dataloader = get_train_dataloader(train_dataset, cfg, verbose)
    
    if val_df is not None:
        val_dataset = CustomDataset(val_df, cfg, aug=None, mode="val")
        val_dataloader = get_val_dataloader(val_dataset, cfg, verbose)

    if test_df is not None:
        test_dataset = CustomDataset(test_df, cfg, aug=None, mode="test")
        test_dataloader = get_val_dataloader(test_dataset, cfg, verbose)

    model = Net(cfg)
    model.to(cfg.device)

    total_steps = len(train_dataset)
    params = model.parameters()
    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=0)
    scheduler = get_scheduler(cfg, optimizer, total_steps)

    if cfg.mixed_precision:
        scaler = GradScaler()
    else:
        scaler = None

    cfg.curr_step = 0
    i = 0
    optimizer.zero_grad()
    l_acc = [2,3,4,6]
    for epoch in range(cfg.epochs):
        set_seed(cfg.seed + epoch)
        if verbose:
            print("EPOCH:", epoch)
            progress_bar = tqdm(range(len(train_dataloader)))
        else:
            progress_bar = range(len(train_dataloader))
        tr_it = iter(train_dataloader)
        losses = 0
        gc.collect()
        
        n_accumulate = l_acc[-1]
        #  np.roll; roll array elements along a given axis.
        l_acc = np.roll(l_acc, shift=1)        

        for itr in progress_bar:
            i += 1
            data = next(tr_it)
            model.train()
            torch.set_grad_enabled(True)
            batch = batch_to_device(data, cfg.device)            
            
            if cfg.mixed_precision:
                with autocast():
                    output_dict = model(batch)
            else:
                output_dict = model(batch)                
            
            loss = output_dict["loss"]
            losses += loss.item()
            if cfg.mixed_precision:
                scaler.scale(loss).backward()                                
                if (itr + 1) % n_accumulate == 0:
#                     # unscale the gradients before clipping.
#                     scaler.unscale_(optimizer)

#                     # clip the norm of the gradients to 10
#                     # this is to help prevent the "exploding gradients" problem.                
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()                
            else:
                loss.backward()
                if (itr + 1) % n_accumulate == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                
            if scheduler is not None:
                scheduler.step()
                
            #print("step/iter {} and total loss {}".format(itr+1, loss.item()))
        print("epoch {} and total loss {}".format(epoch+1, losses/itr+1))                
        if val_df is not None:
            if (epoch + 1) % cfg.eval_epochs == 0 or (epoch + 1) == cfg.epochs:
                run_eval(model, val_dataloader, cfg, pre="val", verbose=verbose)

    if test_df is not None:
        return run_eval(model, test_dataloader, cfg, pre="test", verbose=verbose)
    else:
        return model

def run_test(model, cfg, test_df):
    test_dataset = CustomDataset(test_df, cfg, aug=None, mode="test")
    test_dataloader = get_val_dataloader(test_dataset, cfg, verbose=False)
    return run_eval(model, test_dataloader, cfg, pre="test", verbose=False)



