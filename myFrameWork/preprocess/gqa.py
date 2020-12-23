import torch.utils.data as data
import glob
import json
import time

class GQADataset(data.Dataset):
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
  
  
  def load_scene_graph():
    with open(self.path, "r") as fp:
      scene_graph = json.load(fp)
    
  
  
  def __getitem__(self, idx):
    """
    ques_ix_iter, ques_pos_iter, ans_iter, iid = self.load_ques_ans(idx)
    frcn_feat_iter, grid_feat_iter, bbox_feat_iter = self.load_img_feats(idx, iid)

        return \
            torch.from_numpy(frcn_feat_iter),\
            torch.from_numpy(grid_feat_iter),\
            torch.from_numpy(bbox_feat_iter),\
            torch.from_numpy(ques_ix_iter),\
            torch.from_numpy(ans_iter),\
            torch.from_numpy(ques_pos_iter)
    """

  def __len__(self):
    return self.data_size

  def shuffle_list(self, list):
    random.shuffle(list)
