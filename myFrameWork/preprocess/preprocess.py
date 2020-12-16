import torch.utils.data as data
import glob
import json
import time

class VGDataset(data.Dataset):
    def __init__(self):
        print("Initializing the dataset ...")
        print("Loading metadatas ...")
        print("---------------------------------------------")
        start_time = time.time()
        self.load_synonyms()
        self.load_images()
        self.load_objects()
        self.load_attributes()
        self.load_relationships()
        finish_time = time.time()
        print("---------------------------------------------")
        print("Done! %.6f seconds" % (finish_time - start_time))
        
    def load_synonyms(self):
        # get the object synsets
        print("Loading the object synsets ...")
        with open("./data/vgenome/object_synsets.json", "r") as fp:
            synsets = json.load(fp)
        self.object_sample_to_class = synsets
        self.object_class_to_index = dict()
        for sample, cls in synsets.items():
            if cls not in self.object_class_to_index:
                self.object_class_to_index[cls] = len(self.object_class_to_index)
        print("Finish! Totally " + str(len(self.object_class_to_index)) + " kinds of objects\n")
        
        # get the relationship synsets
        print("Loading the relationship synsets ...")
        with open("./data/vgenome/relationship_synsets.json", "r") as fp:
            synsets = json.load(fp)
        self.relationship_sample_to_class = synsets
        self.relationship_class_to_index = dict()
        for sample, cls in synsets.items():
            if cls not in self.relationship_class_to_index:
                self.relationship_class_to_index[cls] = len(self.relationship_class_to_index)
        print("Finish! Totally " + str(len(self.relationship_class_to_index)) + " kinds of relationships\n")
        
        # get the attribute synsets
        print("Loading the attribute synsets ...")
        with open("./data/vgenome/attribute_synsets.json", "r") as fp:
            synsets = json.load(fp)
        self.attribute_sample_to_class = synsets
        self.attribute_class_to_index = dict()
        for sample, cls in synsets.items():
            if cls not in self.attribute_class_to_index:
                self.attribute_class_to_index[cls] = len(self.attribute_class_to_index)
        print("Finish! Totally " + str(len(self.attribute_class_to_index)) + " kinds of attributes\n")
        
        
    def load_image_path(self):
        self.image_path = glob.glob("~/murel.bootstrap.pytorch/data/vqa/vgenome/extract_rcnn/2018-04-27_bottom-up-attention_fixed_36/*.pth")
        print(self.image_path)
    
    
    def load_objects(self):
        print("Loading all object metadata ...")
        with open("./data/vgenome/objects_vocab.txt") as fp:
            self.object_to_index = dict()
            self.index_to_object = dict()
            idx = 0
            for line in fp.readlines():
                obj = line.strip()
                print(obj)
                self.object_to_index[obj] = idx
                self.index_to_object[idx] = obj
                
        with open("./data/vgenome/objects.json") as fp:
            objects = json.load(fp)
        self.objects = dict()
        for img in objects:
            region_objects = img["objects"]
            for obj in region_objects:
                if len(obj["synsets"]) == 0:
                    print(obj)
                    continue
                self.objects[obj["object_id"]] = {
                    "x" : obj["x"],
                    "y" : obj["y"],
                    "w" : obj["w"],
                    "h" : obj["h"],
                    "cls" : self.object_class_to_index[obj["synsets"][0]]
                }
        print("Finish! Totally " + str(len(self.objects)) + " objects in " + str(len(self.scenes)) + " images\n")
        
        
    def load_relationships(self):
        print("Loading all relationship metadata ...")
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
        print("Finish! Totally " + str(len(self.relationships)) + " relationships in " + str(len(self.scenes)) + " images\n")
        
        
    def load_attributes(self):
        print("Loading all attribute metadata ...")
        with open("./data/vgenome/attributes.json") as fp:
            attributes = json.load(fp)
        self.attributes = dict()
        for attribute in attributes:
            region_attributes = attribute["attributes"]
            for attr in region_attributes:
                self.attributes[attr["object_id"]] = attr["attributes"]
        print("Finish! Totally " + str(len(self.attributes)) + " attributes in " + str(len(self.scenes)) + " images\n")        
            
        
    def load_images(self):
        print("Loading all image metadata ...")
        with open("./data/vgenome/scene_graphs.json") as fp:
            scenes = json.load(fp)
        self.scenes = dict()
        for scene in scenes:
            relationships = []
            for relation in scene["relationships"]:
                relationships.append(relation["relationship_id"])
            self.scenes[scene["image_id"]] = relationships
        print("Finish! Totally " + str(len(self.scenes)) + " images\n")
        
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
