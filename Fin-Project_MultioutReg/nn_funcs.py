"""
Classes and Helper Functions for Winton Project
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import torch.nn as nn
import torch.utils.data as data_utils

# Define Data Builder
class build_data(BaseEstimator, TransformerMixin):
    """
    Loads and Prepares dataset for pytorch
    WMAE weights are last two values in y"""
    
    def __init__(self, df, drop, split_size=0.33, rand=22391, batch=1, shuffle=True, pin=True, ts_only=True, wt=True):
        self.wt = wt
        self.rand = rand
        self.split_size = split_size
        self.batch = batch
        self.shuffle = shuffle
        self.pin = pin
        self.ts = ts_only
        
        if pin:
            print('If pin==True the data cannot be preloaded into GPU, cuda must be False when fitting')
        
        df = df.astype('float')
        
        ccols = [i for i in df.columns if 'Feature' in i]
        self.keep = [i for i in ccols if i not in drop]
        
        self.x = df.iloc[:,26:147] # time steps remove -1 and -2
        print(self.x.columns)
        self.x2 = df.loc[:,self.keep] # other features
        self.y = df.iloc[:,147:]
               
    def _na_fill(self,mode):
        for i in self.x2.columns:
            if i in mode:
                self.x2[i] = self.x2[i].fillna(value=self.x2[i].mode()[0])
            else:
                self.x2[i] = self.x2[i].fillna(value=self.x2[i].median())
                
        self.x = self.x.interpolate(method='linear', axis=1)
        self.x_fin = pd.concat([self.x2,self.x], axis=1)
        
    def _split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x_fin, self.y, test_size=self.split_size, random_state=self.rand)

        # Seperate Features and TS
        self.X_train_ts = X_train.iloc[:,25:147]
        self.X_test_ts = X_test.iloc[:,25:147]
        print(self.X_train_ts.columns)

        self.X_train_ft = X_train.iloc[:,:23]
        self.X_test_ft = X_test.iloc[:,:23]

        # Get Weights for MAE
        # Weights also included in loader, be sure to index when running
        self.test_wt, self.train_wt = np.asarray(y_test.iloc[:,-2:]), np.asarray(y_train.iloc[:,-2:])
        self.y_test, self.y_train = np.asarray(y_test.iloc[:,:-2]), np.asarray(y_train.iloc[:,:-2])
        
    def _scale(self,stsc,lab,dev=True):
        ctrans =  ColumnTransformer(
                    [('scale_all', StandardScaler(), stsc),
                     ('cats', OneHotEncoder(categories='auto'), lab)])
        
       # xtsc = StandardScaler()
        xtsc = QuantileTransformer(output_distribution='normal', random_state=self.rand)
       # ytsc = StandardScaler()
        ytsc = QuantileTransformer(output_distribution='normal', random_state=self.rand)
        mmx = MinMaxScaler(feature_range=(-1,1))
        mmy = MinMaxScaler(feature_range=(-1,1))
        #wtsc = StandardScaler(with_mean=False)
        
        self.X_train_ft = ctrans.fit_transform(self.X_train_ft)
        self.X_test_ft = ctrans.transform(self.X_test_ft)
        self.X_train_ts = xtsc.fit_transform(self.X_train_ts)
        self.X_test_ts = xtsc.transform(self.X_test_ts)
        
        self.X_train_ts = mmx.fit_transform(self.X_train_ts)
        self.X_test_ts = mmx.transform(self.X_test_ts)
        
        
        if self.ts:
            self.x_train = self.X_train_ts
            self.x_test = self.X_test_ts
        else:
            self.x_train = np.concatenate([self.X_train_ft, self.X_train_ts], axis=1)
            self.x_test = np.concatenate([self.X_test_ft, self.X_test_ts], axis=1)
        
       # self.train_wt = wtsc.fit_transform(self.train_wt)
       # self.test_wt = wtsc.transform(self.test_wt)
        
        self.y_train_sc = ytsc.fit_transform(self.y_train)
        self.y_test_sc = ytsc.transform(self.y_test)
        self.y_train_sc = mmy.fit_transform(self.y_train_sc)
        self.y_test_sc = mmy.transform(self.y_test_sc)
        
        if self.wt:
            self.y_train = np.concatenate([self.y_train,self.train_wt],axis=1)
            self.y_test = np.concatenate([self.y_test,self.test_wt],axis=1)
        
        self.xtrans_sc = xtsc
        self.xtrans_mm = mmx
        self.ytrans_mm = mmy
        self.ytrans_sc = ytsc
        self.ftrans = ctrans
        
    def fit(self, mode, stsc, lab, cuda=False):
        self._na_fill(mode)
        self._split()
        self._scale(stsc,lab)
        self.mode = mode
        
        torch_x_train, torch_y_train = torch.from_numpy(self.x_train).float(), torch.from_numpy(self.y_train_sc).float()
        torch_x_test, torch_y_test = torch.from_numpy(self.x_test).float(), torch.from_numpy(self.y_test_sc).float()

        m_ret = 119
        
        if cuda:
            torch_x_train, torch_x_test = torch_x_train.view(-1,1,m_ret).cuda(), torch_x_test.view(-1,1,m_ret).cuda()
            torch_y_train, torch_y_test = torch_y_train.view(-1,1,62).cuda(), torch_y_test.view(-1,1,62).cuda()
        else:
            torch_x_train, torch_x_test = torch_x_train.view(-1,1,m_ret), torch_x_test.view(-1,1,m_ret)
            torch_y_train, torch_y_test = torch_y_train.view(-1,1,62), torch_y_test.view(-1,1,62)
        
        train = data_utils.TensorDataset(torch_x_train, torch_y_train)
        test = data_utils.TensorDataset(torch_x_test, torch_y_test)

        
        train_loader = data_utils.DataLoader(train, batch_size=self.batch, shuffle=self.shuffle, pin_memory=self.pin)
        test_loader = data_utils.DataLoader(test, batch_size=self.batch, shuffle=self.shuffle, pin_memory=self.pin)
        
        return train_loader, test_loader
    
    def fit_sub(self, sub_df, ft=False):
        sub_df = sub_df.astype('float')
        sub_x = sub_df.iloc[:,26:147] # time steps
        print(sub_x.columns)
        sub_x2 = sub_df.loc[:,self.keep] # other features
        
        # Fill NA
        for i in sub_x2.columns:
            if i in self.mode:
                sub_x2[i] = sub_x2[i].fillna(value=sub_x2[i].mode()[0])
            else:
                sub_x2[i] = sub_x2[i].fillna(value=sub_x2[i].median())
                
        sub_x = sub_x.interpolate(method='linear', axis=1)
        sub_x = sub_x.iloc[:,2:] # remove ret -1 and -2
        print(sub_x.columns)
        # Scale
        
        sub_x = self.xtrans_sc.transform(sub_x)
        sub_x = self.xtrans_mm.transform(sub_x)
        sub_x2 = self.ftrans.transform(sub_x2)
        if ft:
            sub = np.concatenate([sub_x2, sub_x], axis=1)
        else:
            sub = sub_x
        
        # Make loader
        sub = torch.from_numpy(sub).float()
        sub_ds = data_utils.TensorDataset(sub)
        sub_loader = data_utils.DataLoader(sub_ds, batch_size=self.batch, shuffle=False, pin_memory=self.pin)
        
        return sub_loader
        
    def get_weights(self):
        self.train_wt = torch.from_numpy(self.train_wt).float()
        self.test_wt = torch.from_numpy(self.test_wt).float()
        return self.train_wt, self.test_wt
    
    def reverse_trans(self, x=False, y=False):
        if x is not False and y is not False:
            return self.xtrans_sc.inverse_transform(x), self.ytrans_sc.inverse_transform(y)
        elif x is not False:
            return self.xtrans_sc.inverse_transform(x)
        elif y is not False:
            y2 = self.ytrans_mm.inverse_transform(y)
            return self.ytrans_sc.inverse_transform(y2)

        
    def make_sub(self, sub, fn, path=r"C:\Users\rlagr\fin\winton\data\\"):
        sub = self.ytrans_mm.inverse_transform(sub)
        sub = self.ytrans_sc.inverse_transform(sub)
        sub = sub.reshape(-1,1)

        win = [i for i in range(1,120001)]
        step = [i for i in range(1,63)]
        rnames = [None] * sub.shape[0]

        ind = 0

        for i in win:
            for k in step:
                name = str(i) + '_' + str(k)
                rnames[ind] = name
                ind += 1

        s = pd.DataFrame(rnames)
        s.columns = ['Id']
        s['Predicted'] = sub
        path = path + fn

        s.to_csv(path, index=False)


#  Custom Loss Functions

class wmae_loss(torch.nn.Module):
    """Second weight for last 2 preds, first for rest"""
    def __init__(self):
        super(wmae_loss,self).__init__()
        
    def _wmae(self,pred, true, wts):
        n = true.shape[0] * true.shape[-1]
        intra = torch.sum(wts[:,:,0] * torch.abs(true[:,:,:-2] - pred[:,:,:-2]))
        daily = torch.sum(wts[:,:,1] * torch.abs(true[:,:,-2:] - pred[:,:,-2:]))
        return (intra + daily) / n
        
    def forward(self, pred, true, wt):
        return self._wmae(pred, true, wt)


# Define other custom loss function
class mod_mse(torch.nn.Module):
    """Punish predictions close to zero and not close to true
    Add standard MSE to prevent large inaccurate predictions
    Use l1 and l2 to weight each side"""
    def __init__(self, l1=1, l2=1, eps=0.00001):
        super(mod_mse, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.eps = eps
        
    def _mod(self, pred, true):
        pred = torch.add(pred, self.eps)
        return self.l1 * torch.mean(torch.abs(1 - (true/pred))) + self.l2 * torch.mean((pred-true)**2)
    
    def forward(self, pred, true):
        return self._mod(pred,true)


# Training and Prediction/Submission Functions
class train_model():
    """Class to train model
        To use WMAE pass weight to .fit()
        Assumes data is preloaded into GPU if using it"""
    def __init__(self, model, loss, opt, sched=None):
        self.model = model
        self.loss_f = loss
        self.sched = sched
        self.opt = opt
        self.losses = False
        
    def fit(self, train, test, num_epochs=10, plots=1, loss_print=500, batch=1, weights=None):
        if weights is not None:
            tr_weights = weights[0]
            ts_weights = weights[1]
        
        # Dict to store losses
        self.losses = {'train_loss' : [None]*num_epochs , 'eval_loss' : [None]*num_epochs}
        
        for epoch in range(num_epochs): # Iterate through train then test
            print('\nstarting epoch ', str(epoch + 1))
            iterations = 0
            iter_loss = 0.0

            self.model.train()
            for i, (x,y) in enumerate(train): # Iterate train data
                #with torch.autograd.detect_anomaly():

                self.opt.zero_grad()
                out = self.model(x)

                if weights is None:
                    loss = self.loss_f(out, y)
                else:
                    loss = self.loss_f(out, y, tr_weights[i:i+batch,:,:])

                iter_loss += loss.item()
                loss.backward()
                self.opt.step()
                if self.sched is not None:
                    self.sched.step()


                iterations += 1

                if i % loss_print == 0:
                    print('\titeration', str(i), '--', str(iter_loss/iterations))

            # Record Training Loss
            self.losses['train_loss'][epoch] = iter_loss / iterations

            # Test
            ev_loss = 0.0
            iterations = 0
            
            self.model.eval()
            for i, (x,y) in enumerate(test):
        
                out = self.model(x)

                if weights is None:
                    loss = self.loss_f(out, y)
                else:
                    loss = self.loss_f(out, y, ts_weights[i:i+batch,:,:])
                    
                ev_loss += loss.item()

                iterations += 1

            if epoch % plots == 0: # Output prediction vs true plot
                tst_plt = out.cpu().data.numpy()
                tr_plt = y.cpu().data.numpy()
                tst_plt = tst_plt.reshape(-1,62)
                tr_plt = tr_plt.reshape(-1,62)
                plt.figure(figsize=(13,7))
                plt.plot(tst_plt[0], label="Pred")
                plt.plot(tr_plt[0], label='True')
                plt.legend()
                plt.show()

            self.losses['eval_loss'][epoch] = ev_loss / iterations
            
            print('Epoch {}/{}, Training Loss: {:.5f}, Testing Loss: {:.5f}'
                   .format(epoch+1, num_epochs, self.losses['train_loss'][epoch], self.losses['eval_loss'][epoch]))
        
    def _losses(self, which='train_loss'):
        if not self.losses:
            print('Model has not been fit yet')
        else:
            if which == 'train_loss':
                return self.losses['train_loss']
            elif which == 'eval_loss':
                return self.losses['eval_loss']
            else:
                print('Pass either train_loss (default) or eval_loss')


def predict(model, dat, act=False, sub=False):
    """Outputs predicted data for a given model
    Can also Return True Values (act = True)
    Or for use in submission (sub = True)
    """

    model.eval()
    res = []
    y_out = []
    if sub:
        for i, x in enumerate(dat):
            x = x[0].view(-1,1,119) # Change if using ret -1 and -2
         #   x = x.cuda()
            out = model(x)
            out = out.cpu().data.numpy()
            res.append(out)
            
    else:
        for i, (x,y) in enumerate(dat):
            out = model(x)
            out = out.cpu().data.numpy()
            res.append(out)

            if act:
                true = y.cpu().data.numpy()
                y_out.append(true)
    
    res = np.array(res)
    if act:
        y_out = np.array(y_out)
        return res, y_out
    else:
        return res
    
def make_sub(sub_f, fn, data):
    """Creates submission file
    """
    sub = data.reverse_trans(y=sub_f)
    sub = sub.reshape(-1,1)
   # print(sub.shape)
    win = [i for i in range(1,120001)]
    step = [i for i in range(1,63)]
    rnames = [None] * sub.shape[0]

    ind = 0

    for i in win:
        for k in step:
            name = str(i) + '_' + str(k)
            rnames[ind] = name
            ind += 1

    s = pd.DataFrame(rnames)
    s.columns = ['Id']
    s['Predicted'] = sub
    #path = r"C:\Users\rlagr\fin\winton\data\\"
    path = r"C:\Users\RemyLagrois\Desktop\\"
    path = path + fn
    
    s.to_csv(path, index=False)