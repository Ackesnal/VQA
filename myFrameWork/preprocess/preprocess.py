import torch.utils.data as data
import glob
import json

class VGDataset(data.Dataset):
    def __init__(self):
        self.load_synonyms()
        self.load_images()
        self.load_objects()
        self.load_attributes()
        self.load_relationships()
        
    def load_synonyms(self):
        # get the object synsets
        with open("./data/object_synsets.json", "r") as fp:
            synsets = json.load(fp)
        self.object_sample_to_class = synsets
        self.object_class_to_index = dict()
        for sample, cls in synsets.items():
            if cls not in self.object_class_to_index:
                self.object_class_to_index[cls] = len(self.object_class_to_index)
        
        # get the relationship synsets
        with open("./data/relationship_synsets.json", "r") as fp:
            synsets = json.load(fp)
        self.relationship_sample_to_class = synsets
        self.relationship_class_to_index = dict()
        for sample, cls in synsets.items():
            if cls not in self.relationship_class_to_index:
                self.relationship_class_to_index[cls] = len(self.relationship_class_to_index)
        
        # get the attribute synsets
        with open("./data/attribute_synsets.json", "r") as fp:
            synsets = json.load(fp)
        self.attribute_sample_to_class = synsets
        self.attribute_class_to_index = dict()
        for sample, cls in synsets.items():
            if cls not in self.attribute_class_to_index:
                self.attribute_class_to_index[cls] = len(self.attribute_class_to_index)
    
    def load_image_path(self):
        self.image_path = glob.glob("~/murel.bootstrap.pytorch/data/vqa/vgenome/extract_rcnn/2018-04-27_bottom-up-attention_fixed_36/*.pth")
        print(self.image_path)
    
    def load_objects(self):
        with open("./data/vgenome/objects.json") as fp:
            objects = json.load(fp)
        self.objects = dict()
        for img in objects:
            region_objects = img["objects"]
            for obj in region_objects:
                self.objects[obj["object_id"]] = {
                    "x" : obj["x"],
                    "y" : obj["y"],
                    "w" : obj["w"],
                    "h" : obj["h"],
                    "cls" : self.object_class_to_index[obj["synsets"][0]]
                }
                
    def load_relationships(self):
        with open("./data/vgenome/relationships.json") as fp:
            relationships = json.load(fp)
        self.relationships = dict()
        for relation in relationships:
            region_relationships = relation["relationships"]
            for rela in region_relationships:
                self.relationships[rela["relationship_id"]] = {
                    "objects" : [rela["subject"]["object_id"], rela["object"]["object_id"]],
                    "cls" : self.relationship_class_to_index[rela["synsets"][0]]
                }
        
    def load_attributes(self):
        with open("./data/vgenome/attributes.json") as fp:
            attributes = json.load(fp)
        self.attributes = dict()
        for attribute in attributes:
            region_attributes = attribute["attributes"]
            for attr in region_attributes:
                self.attributes[attr["object_id"]] = attr["attributes"]
        
    def load_images(self):    
        with open("./data/vgenome/scene_graphs.json") as fp:
            scenes = json.load(fp)
        self.scenes = dict()
        for scene in scenes:
            relationships = []
            for relation in scene["relationships"]:
                relationships.append(relation["relationship_id"])
            self.scenes[scene["image_id"]] = relationships
        
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
