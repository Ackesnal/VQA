# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch
import math


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

    
    
class MHAttRela(nn.Module):
    def __init__(self, __C):
        super(MHAttRela, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_r = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, r, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)
        
        r = self.linear_r(r).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, r, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, relation, mask):
        d_k = query.size(-1)

        scores = (torch.matmul(query, key.transpose(-2, -1)) + torch.matmul(relation, key.transpose(-2, -1))) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)
    
# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)
        
        self.relu = nn.ReLU()
        self.linear = nn.Linear(14, 100)
        
    def forward(self, x, y, x_mask, y_mask):
        
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask)
        ))
        
        t = self.dropout2(self.mhatt2(v=x, k=x, q=y, mask=x_mask))
        t = self.relu(self.linear(t.reshape(t.shape[0], t.shape[2], t.shape[1])))
        x = self.norm2(x + t.reshape(t.shape[0], t.shape[2], t.shape[1]))
        
        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


    
    
class COGA(nn.Module):
    def __init__(self, __C):
        super(COGA, self).__init__()

        self.mhattx2y = MHAttRela(__C)
        self.mhatty2x = MHAttRela(__C)
        self.mhattxself = MHAttRela(__C)
        self.mhattyself = MHAttRela(__C)
        self.ffnx = FFN(__C)
        self.ffny = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)
        
        self.dropout4 = nn.Dropout(__C.DROPOUT_R)
        self.norm4 = LayerNorm(__C.HIDDEN_SIZE)
        
        self.dropout5 = nn.Dropout(__C.DROPOUT_R)
        self.norm5 = LayerNorm(__C.HIDDEN_SIZE)
        
        self.dropout6 = nn.Dropout(__C.DROPOUT_R)
        self.norm6 = LayerNorm(__C.HIDDEN_SIZE)
        
        
    def forward(self, x, y, x_mask, y_mask, x_pos, y_pos):
        new_x = self.norm1(x + self.dropout1(self.mhattx2y(v=y, k=y, q=x, r=x_pos, mask=y_mask)))
        new_y = self.norm2(y + self.dropout2(self.mhatty2x(v=x, k=x, q=y, r=y_pos, mask=x_mask)))
        new_x = self.norm3(new_x + self.dropout3(self.mhattxself(v=new_x, k=new_x, q=new_x, r=x_pos, mask=x_mask)))
        new_y = self.norm4(new_y + self.dropout4(self.mhattxself(v=new_y, k=new_y, q=new_y, r=y_pos, mask=y_mask)))
        new_x = self.norm5(new_x + self.dropout5(self.ffnx(new_x)))
        new_y = self.norm6(new_y + self.dropout6(self.ffny(new_y)))
        
        return new_x, new_y
    
# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, __C):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])
        #self.enc_list = nn.ModuleList([COGA(__C) for _ in range(__C.LAYER)])
        
    def forward(self, y, x, y_mask, x_mask, y_pos, x_pos):
        # Get encoder last hidden vector
        for enc in self.enc_list:
            y = enc(y, y_mask)
        for dec in self.dec_list:
            x = enc(x, y, x_mask, y_mask)
            
        #for enc in self.enc_list:
        #    x, y = enc(x, y, x_mask, y_mask, x_pos, y_pos)

        return y, x
