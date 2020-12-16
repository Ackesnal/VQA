import torch.utils.data as data
import glob
import json

class VGDataset(data.Dataset):
    def __init__(self):
        with open("data/vgenome/scene_graphs.json") as fp:
            scene_graphs = json.load(fp)
        print(type(scene_graphs))

    def load_synonyms(self):
        # get the object synsets
        with open("object_synsets.json", "r") as fp:
            synsets = json.load(fp)
        self.object_sample_to_class = synsets
        class_to_sample = dict()
        for sample, cls in synsets.items():
            if cls not in class_to_sample:
                class_to_sample[cls] = []
            class_to_sample[cls].append(sample)
        self.object_class_to_sample = class_to_sample
        
        # get the relationship synsets
        with open("relationship_synsets.json", "r") as fp:
            synsets = json.load(fp)
        self.relationship_sample_to_class = synsets
        class_to_sample = dict()
        for sample, cls in synsets.items():
            if cls not in class_to_sample:
                class_to_sample[cls] = []
            class_to_sample[cls].append(sample)
        self.relationship_class_to_sample = class_to_sample
        
        # get the attribute synsets
        with open("attribute_synsets.json", "r") as fp:
            synsets = json.load(fp)
        self.attribute_sample_to_class = synsets
        class_to_sample = dict()
        for sample, cls in synsets.items():
            if cls not in class_to_sample:
                class_to_sample[cls] = []
            class_to_sample[cls].append(sample)
        self.attribute_class_to_sample = class_to_sample
    
    
    def load_image_path(self):
        self.image_path = glob.glob("../../murel.bootstrap.pytorch/data/vqa/vgenome/extract_rcnn/2018-04-27_bottom-up-attention_fixed_36/*.pth")
        print(self.image_path)
    
    
    def get_image_features(self,):
        raise NotImplementedError() 
    
    
    def load_ques_ans(self, idx):
        raise NotImplementedError()


    def load_img_feats(self, idx, iid):
        frcn_img = torch.load("./data/vgemone/"+idx+".pth")
        

    def __getitem__(self, idx):

        ques_ix_iter, ques_pos_iter, ans_iter, iid = self.load_ques_ans(idx)

        frcn_feat_iter, grid_feat_iter, bbox_feat_iter = self.load_img_feats(idx, iid)

        return \
            torch.from_numpy(frcn_feat_iter),\
            torch.from_numpy(grid_feat_iter),\
            torch.from_numpy(bbox_feat_iter),\
            torch.from_numpy(ques_ix_iter),\
            torch.from_numpy(ans_iter),\
            torch.from_numpy(ques_pos_iter)


    def __len__(self):
        return self.data_size

    def shuffle_list(self, list):
        random.shuffle(list)
v = VGDataset()
