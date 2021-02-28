import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler

class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2619422201258426)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(0.2619422201258426)
        self.dense4 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.2619422201258426)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm4(x)
        x = self.dropout4(x)
        x = F.leaky_relu(self.dense4(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x

class resnetModel(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size,ispretrain=False):
        super(resnetModel, self).__init__()
        self.ispretrain=ispretrain
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))


        self.batch_norm2 = nn.BatchNorm1d(num_features+hidden_size)
        self.dropout2 = nn.Dropout(0.2619422201258426)
        self.dense2 = nn.utils.weight_norm(nn.Linear(num_features+hidden_size, hidden_size))
        self.batch_norm20 = nn.BatchNorm1d(hidden_size)
        self.dropout20 = nn.Dropout(0.2619422201258426)
        self.dense20 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))


        self.batch_norm3 = nn.BatchNorm1d(2*hidden_size)
        self.dropout3 = nn.Dropout(0.2619422201258426)
        self.dense3 = nn.utils.weight_norm(nn.Linear(2*hidden_size, hidden_size))
        self.batch_norm30 = nn.BatchNorm1d(hidden_size)
        self.dropout30 = nn.Dropout(0.2619422201258426)
        self.dense30 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))


        self.batch_norm4 = nn.BatchNorm1d(2*hidden_size)
        self.dropout4 = nn.Dropout(0.2619422201258426)
        if self.ispretrain:
          self.dense4 = nn.utils.weight_norm(nn.Linear(2*hidden_size, num_targets))
        else:
          self.dense5 = nn.utils.weight_norm(nn.Linear(2*hidden_size, num_targets))

    def forward(self, x):
        x1 = self.batch_norm1(x)
        x1 = F.leaky_relu(self.dense1(x1))
        x = torch.cat([x,x1],1)

        x2 = self.batch_norm2(x)
        x2 = self.dropout2(x2)
        x2 = F.leaky_relu(self.dense2(x2))
        x2 = self.batch_norm20(x2)
        x2 = self.dropout20(x2)
        x2 = F.leaky_relu(self.dense20(x2))
        x = torch.cat([x1,x2],1)

        x3 = self.batch_norm3(x)
        x3 = self.dropout3(x3)
        x3 = F.leaky_relu(self.dense3(x3))
        x3 = self.batch_norm30(x3)
        x3 = self.dropout30(x3)
        x3 = F.leaky_relu(self.dense30(x3))
        x3 = torch.cat([x2,x3],1)

        x3 = self.batch_norm4(x3)
        x3 = self.dropout4(x3)
        if self.ispretrain:
          x3 = self.dense4(x3)
        else:
          x3 = self.dense5(x3)
        return x3

class resnetsimpleModel(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size,ispretrain=False):
        super(resnetsimpleModel, self).__init__()
        self.ispretrain=ispretrain
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))


        self.batch_norm2 = nn.BatchNorm1d(num_features+hidden_size)
        self.dropout2 = nn.Dropout(0.2619422201258426)
        self.dense2 = nn.utils.weight_norm(nn.Linear(num_features+hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(2*hidden_size)
        self.dropout3 = nn.Dropout(0.2619422201258426)
        self.dense3 = nn.utils.weight_norm(nn.Linear(2*hidden_size, hidden_size))

        self.batch_norm4 = nn.BatchNorm1d(2*hidden_size)
        self.dropout4 = nn.Dropout(0.2619422201258426)
        if self.ispretrain:
          self.dense4 = nn.utils.weight_norm(nn.Linear(2*hidden_size, num_targets))
        else:
          self.dense5 = nn.utils.weight_norm(nn.Linear(2*hidden_size, num_targets))

    def forward(self, x):
        x1 = self.batch_norm1(x)
        x1 = F.leaky_relu(self.dense1(x1))
        x = torch.cat([x,x1],1)

        x2 = self.batch_norm2(x)
        x2 = self.dropout2(x2)
        x2 = F.leaky_relu(self.dense2(x2))
        x = torch.cat([x1,x2],1)

        x3 = self.batch_norm3(x)
        x3 = self.dropout3(x3)
        x3 = self.dense3(x3)
        x3 = torch.cat([x2,x3],1)

        x3 = self.batch_norm4(x3)
        x3 = self.dropout4(x3)
        if self.ispretrain:
          x3 = self.dense4(x3)
        else:
          x3 = self.dense5(x3)
        return x3

class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

class MoADataset:
    def __init__(self, features, targets,noise=0.1,val=0):
        self.features = features
        self.targets = targets
        self.noise = noise
        self.val = val

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        sample = self.features[idx, :].copy()

        if 0 and np.random.rand()<0.3 and not self.val:
            sample = self.swap_sample(sample)

        dct = {
            'x' : torch.tensor(sample, dtype=torch.float),
            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)
        }
        return dct

    def swap_sample(self,sample):
            #print(sample.shape)
            num_samples = self.features.shape[0]
            num_features = self.features.shape[1]
            if len(sample.shape) == 2:
                batch_size = sample.shape[0]
                random_row = np.random.randint(0, num_samples, size=batch_size)
                for i in range(batch_size):
                    random_col = np.random.rand(num_features) < self.noise
                    #print(random_col)
                    sample[i, random_col] = self.features[random_row[i], random_col]
            else:
                batch_size = 1

                random_row = np.random.randint(0, num_samples, size=batch_size)


                random_col = np.random.rand(num_features) < self.noise
                #print(random_col)
                #print(random_col)

                sample[ random_col] = self.features[random_row, random_col]

            return sample

class TestDataset:
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct
