# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.utils.make_mask import make_mask
from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm
from openvqa.models.rela.rela import MCA_ED
from openvqa.models.rela.adapter import Adapter
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size, postag_size):
        super(Net, self).__init__()
        self.__C = __C
    
        if __C.USE_BERT:
            self.pretrained_emb = pretrained_emb
            
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.HIDDEN_SIZE
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=20,
            embedding_dim=__C.HIDDEN_SIZE
        )
        self.postag_embedding = nn.Embedding(
            num_embeddings=postag_size,
            embedding_dim=__C.HIDDEN_SIZE
        )

        # Loading the GloVe embedding weights
        #if __C.USE_GLOVE:
        #    self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        
        #self.lstm = nn.LSTM(
        #    input_size=__C.WORD_EMBED_SIZE,
        #    hidden_size=__C.HIDDEN_SIZE,
        #    num_layers=1,
        #    batch_first=True
        #)
        

        self.adapter = Adapter(__C)

        self.backbone = MCA_ED(__C)

        # Flatten to vector
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        # Classification layers
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)
        
        # Tucker Decomposition For Bilinear Fusion
        self.linear1 = nn.Linear(__C.FLAT_OUT_SIZE, 1000)
        self.linear2 = nn.Linear(__C.FLAT_OUT_SIZE, 1000)
        self.bilinear = nn.Bilinear(1000,1000,1000)
        self.linear3 = nn.Linear(1000, __C.FLAT_OUT_SIZE)
        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        
        
    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix, ques_postag):
        # Pre-process Language Feature
        if self.__C.USE_BERT:
            lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
            position_embed = self.position_embedding(torch.arange(ques_ix.shape[1], device='cuda').repeat(ques_ix.shape[0], 1))
            lang_feat = self.pretrained_emb(ques_ix)
        else:
            lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
            lang_feat = self.embedding(ques_ix)
            #self.lstm.flatten_parameters()
            #lang_feat, _ = self.lstm(lang_feat)
            position_embed = self.position_embedding(torch.arange(ques_ix.shape[1], device='cuda').repeat(ques_ix.shape[0], 1))
            postag_embed = self.postag_embedding(ques_postag)
            lang_feat = lang_feat + position_embed + postag_embed
        img_feat, img_feat_mask, bbox_feat = self.adapter(frcn_feat, grid_feat, bbox_feat)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask,
            position_embed,
            bbox_feat
        )

        # Flatten to vector
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        # Classification layers
        proj_feat = self.TuckerFusion(lang_feat, img_feat)
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat
    
    def TuckerFusion(self, lang_feat, img_feat):
        lang_feat = self.dropout1(self.linear1(lang_feat))
        img_feat = self.dropout2(self.linear2(img_feat))
        fused_feat = self.dropout3(self.linear3(self.bilinear(lang_feat, img_feat)))
        return fused_feat
        
