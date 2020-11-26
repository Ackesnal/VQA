# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import torch.nn as nn
import torch
from openvqa.core.base_dataset import BaseAdapter
from openvqa.utils.make_mask import make_mask


class Adapter(BaseAdapter):
    def __init__(self, __C):
        super(Adapter, self).__init__(__C)
        self.__C = __C
        self.sim = nn.CosineSimilarity(dim = 1)

    def bbox_proc(self, bbox):
        area = (bbox[:, :, 2] - bbox[:, :, 0]) * (bbox[:, :, 3] - bbox[:, :, 1])
        return torch.cat((bbox, area.unsqueeze(2)), -1)

    def vqa_init(self, __C):
        imgfeat_linear_size = __C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][1]
        if __C.USE_BBOX_FEAT:
            self.bbox_linear = nn.Linear(5, __C.HIDDEN_SIZE)
            # imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
        self.frcn_linear = nn.Linear(imgfeat_linear_size, __C.HIDDEN_SIZE)

    def gqa_init(self, __C):
        imgfeat_linear_size = __C.FEAT_SIZE['gqa']['FRCN_FEAT_SIZE'][1]
        if __C.USE_BBOX_FEAT:
            self.bbox_linear = nn.Linear(5, __C.BBOXFEAT_EMB_SIZE)
            imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
        self.frcn_linear = nn.Linear(imgfeat_linear_size, __C.HIDDEN_SIZE)

        if __C.USE_AUX_FEAT:
            self.grid_linear = nn.Linear(__C.FEAT_SIZE['gqa']['GRID_FEAT_SIZE'][1], __C.HIDDEN_SIZE)


    def clevr_init(self, __C):
        self.grid_linear = nn.Linear(__C.FEAT_SIZE['clevr']['GRID_FEAT_SIZE'][1], __C.HIDDEN_SIZE)


    def vqa_forward(self, feat_dict):
        frcn_feat = feat_dict['FRCN_FEAT']
        bbox_feat = feat_dict['BBOX_FEAT']

        img_feat_mask = make_mask(frcn_feat)

        if self.__C.USE_BBOX_FEAT:
            bbox_feat = self.bbox_proc(bbox_feat)
            bbox_feat = self.bbox_linear(bbox_feat)
            sim_matrix = torch.zeros(bbox_feat.shape[0], bbox_feat.shape[1], bbox_feat.shape[1])
            for i in range(bbox_feat.shape[1]):
                for j in range(bbox_feat.shape[1]):
                    print(bbox_feat[:, i, :].shape)
                    sim_matrix[:, i, j] = self.sim(bbox_feat[:, i, :], bbox_feat[:, j, :])
            print(sim_matrix, sim_matrix.shape)
            #frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
        frcn_feat = self.frcn_linear(frcn_feat)
        img_feat = frcn_feat + bbox_feat
        
        return img_feat, img_feat_mask, bbox_feat


    def gqa_forward(self, feat_dict):
        frcn_feat = feat_dict['FRCN_FEAT']
        bbox_feat = feat_dict['BBOX_FEAT']
        grid_feat = feat_dict['GRID_FEAT']

        img_feat_mask = make_mask(frcn_feat)

        if self.__C.USE_BBOX_FEAT:
            bbox_feat = self.bbox_proc(bbox_feat)
            bbox_feat = self.bbox_linear(bbox_feat)
            frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
        img_feat = self.frcn_linear(frcn_feat)

        if self.__C.USE_AUX_FEAT:
            grid_feat_mask = make_mask(grid_feat)
            img_feat_mask = torch.cat((img_feat_mask, grid_feat_mask), dim=-1)
            grid_feat = self.grid_linear(grid_feat)
            img_feat = torch.cat((img_feat, grid_feat), dim=1)

        return img_feat, img_feat_mask


    def clevr_forward(self, feat_dict):
        grid_feat = feat_dict['GRID_FEAT']

        img_feat_mask = make_mask(grid_feat)
        img_feat = self.grid_linear(grid_feat)

        return img_feat, img_feat_mask



