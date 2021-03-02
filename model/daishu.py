import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler
from torch.nn.modules.loss import _WeightedLoss


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
        loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

class Model(nn.Module):
    def __init__(self, num_features, num_targets, num_targets2, hidden_size):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2619422201258426)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.2619422201258426)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))
        self.dense4 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets2))
    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        y = self.dense3(x)
        y1 = self.dense4(x)
        return y, y1

class GBN(nn.Module):
    def __init__(self, inp, vbs=128, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm1d(inp, momentum=momentum)
        self.vbs = vbs
    def forward(self, x):
        chunk = torch.chunk(x, max(1, x.size(0)//self.vbs), 0)
        res = [self.bn(y) for y in chunk ]
        return torch.cat(res, 0)

class GLU(nn.Module):
    def __init__(self, inp_dim, out_dim, fc=None, vbs=128):
        super().__init__()
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(inp_dim, out_dim*2)
        self.bn = GBN(out_dim*2, vbs=vbs)
        self.od = out_dim
        self.dropout = nn.Dropout(0.2619422201258426)
    def forward(self, x):
        x = self.dropout(self.bn(F.leaky_relu((self.fc(x)))))
        return x[:, :self.od]*torch.sigmoid(x[:, self.od:])

class FeatureTransformer(nn.Module):
    def __init__(self, inp_dim, out_dim, shared, n_ind, vbs=128):
        super().__init__()
        first = True
        self.shared = nn.ModuleList()
        if shared:
            self.shared.append(GLU(inp_dim, out_dim, shared[0], vbs=vbs))
            first= False
            for fc in shared[1:]:
                self.shared.append(GLU(out_dim, out_dim, fc, vbs=vbs))
        else:
            self.shared = None
        self.independ = nn.ModuleList()
        if first:
            if shared:
                self.independ.append(GLU(inp_dim, out_dim, vbs=vbs))
            else:
                self.independ.append(GLU(out_dim, out_dim, vbs=vbs))
        for x in range(first, n_ind):
            self.independ.append(GLU(out_dim, out_dim, vbs=vbs))
        self.scale = torch.sqrt(torch.tensor([.5], device=device))
        self.dropout = nn.Dropout(0.2619422201258426)
        self.bn = nn.BatchNorm1d(out_dim)
        self.fc = nn.Linear(inp_dim, out_dim)
    def forward(self, x):
        if self.shared:
            x = self.dropout(self.bn(F.leaky_relu(self.shared[0](x))))
            for glu in self.shared[1:]:
                glu_x = self.dropout(glu(x))
                x = torch.add(x, glu_x)
                x = x*self.scale
        else:
            x = self.dropout(self.bn(F.leaky_relu(self.fc(x))))
        for glu in self.independ:
            glu_x = self.dropout(glu(x))
            x = torch.add(x, glu_x)
            x = x*self.scale
        return x

class AttentionTransformer(nn.Module):
    def __init__(self, inp_dim, out_dim, relax, vbs=128):
        super().__init__()
        self.fc = nn.Linear(inp_dim, out_dim)
        self.bn = GBN(out_dim, vbs=vbs)
        self.r = torch.tensor([relax], device=device)
    def forward(self, a, priors):
        a = self.bn(self.fc(a))
        mask = torch.sigmoid(a*priors)
        priors =priors*(self.r-mask)
        return mask, priors

class DecisionStep(nn.Module):
    def __init__(self, inp_dim, n_d, n_a, shared, n_ind, relax, vbs=128):
        super().__init__()
        self.fea_tran = FeatureTransformer(inp_dim, n_d+n_a, shared, n_ind, vbs)
        self.atten_tran = AttentionTransformer(n_a, inp_dim, relax, vbs)
    def forward(self, x, a, priors):
        mask, priors = self.atten_tran(a, priors)
        loss = ((-1)*mask*torch.log(mask+1e-10)).mean()
        x = self.fea_tran(x*mask)#x*mask
        return x, loss, priors

class TabNet(nn.Module):
    def __init__(self, inp_dim, final_out_dim, final_out_dim2, n_d=64, n_a=64, n_shared=2, n_ind=2, n_steps=5, relax=1.2, vbs=128):
        super().__init__()
        if n_shared>0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim, 2*(n_d+n_a)))
            for x in range(n_shared-1):
                self.shared.append(nn.Linear(n_d+n_a, 2*(n_d+n_a)))
        else:
            self.shared=None
        self.first_step = FeatureTransformer(inp_dim, n_d+n_a, self.shared, n_ind)
        self.steps = nn.ModuleList()
        for x in range(n_steps-1):
            self.steps.append(DecisionStep(inp_dim, n_d, n_a, self.shared, n_ind, relax, vbs))
        self.fc = Model(n_d, final_out_dim, final_out_dim2, 1500) # add second targets 
        self.bn = nn.BatchNorm1d(inp_dim)
        self.n_d = n_d
    def forward(self, x):
        x = self.bn(x)
        x_a = self.first_step(x)[:, self.n_d:]
        loss = torch.zeros(1).to(device)
        out = torch.zeros(x.size(0), self.n_d).to(device)
        priors = torch.ones(x.shape).to(device)
        #loss = torch.zeros(1).to(x.device)
        #out = torch.zeros(x.size(0), self.n_d).to(x.device)
        #priors = torch.ones(x.shape).to(x.device)
        for step in self.steps:
            x_te, l, priors = step(x, x_a, priors)
            out += F.relu(x_te[:, :self.n_d])
            x_a = x_te[:, self.n_d:]
            loss += l
        return self.fc(out)

class Self_Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                    "The hidden size (%d) is not a multiple of the number of attention "
                    "heads (%d)" % (hidden_size, n_head)
                )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size)**0.5

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = context_layer
        return outputs

class Attention_dnn(nn.Module):
    def __init__(self, num_features, num_targets, num_targets2, hidden_size0, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size0))
        self.batch_norm1 = nn.BatchNorm1d(hidden_size0)
        self.dropout1 = nn.Dropout(0.2619422201258426)

        self.att_dense1 = nn.utils.weight_norm(nn.Linear(1, 64))

        self.self1 = Self_Attention(64, num_attention_heads, attention_probs_dropout_prob)

        self.batch_norm2 = nn.BatchNorm1d(hidden_size0)
        self.dropout2 = nn.Dropout(0.2619422201258426)

        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size0, hidden_size))
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.2619422201258426)

        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))
        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(0.2619422201258426)

        self.dense4 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))
        self.dense5 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets2))

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.dropout1(self.batch_norm1(F.leaky_relu(self.dense1(x))))
        ori_x = x
        x = x.view(x.shape[0], x.shape[1], 1)

        x = self.att_dense1(x)
        x = self.self1(x)
        x = torch.max(x, dim=-1)[0]
        x = x + ori_x#torch.cat([x, ori_x], dim=1)

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)

        x = F.leaky_relu(self.dense3(x))

        x = self.batch_norm4(x)
        x = self.dropout4(x)

        y = self.dense4(x)
        y1 = self.dense5(x)
        return y, y1

class Dnn(nn.Module):
    def __init__(self, num_features, num_targets, num_targets2, hidden_size):
        super(Dnn, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2619422201258426)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.2619422201258426)

        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(0.2619422201258426)

        self.dense4 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))
        self.dense5 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets2))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = F.leaky_relu(self.dense3(x))

        x = self.batch_norm4(x)
        x = self.dropout4(x)
        y = self.dense4(x)
        y1 = self.dense5(x)
        return y, y1
