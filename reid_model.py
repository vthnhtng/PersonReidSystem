from torch.utils.data import DataLoader
import torchvision.transforms as T
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import glob
import re
import numpy as np
import torch
from modeling.baseline import Baseline
from sklearn.metrics.pairwise import cosine_similarity
import io



def create_transform():
    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([
        T.Resize([384, 128]),
        T.ToTensor(),
        normalize_transform
    ])

    return transform


def L2_norm(tensor):
    return torch.nn.functional.normalize(tensor, dim=1, p=2)


def euclidean_distance(tensor1, tensor2): # use for L2 norm
    diff = tensor1 - tensor2
    distance = torch.sqrt(torch.sum(diff ** 2))
    return distance

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def cosine_similarity(tensor1, tensor2):
    tensor1 = tensor1.detach().cpu().numpy()
    tensor2 = tensor2.detach().cpu().numpy()

    return cosine_similarity(tensor1, tensor2)


def are_same_person(tensor1, tensor2, threshold):
    tensor1_norm = L2_norm(tensor1)
    tensor2_norm = L2_norm(tensor2)
    dist = euclidean_distance(tensor1_norm, tensor2_norm)

    return True if dist <= threshold else False


def create_reid_model(pretrained_path, trained_path):
    reid_model = Baseline(751, 1, pretrained_path, 'bnneck', 'after', 'resnet50', 'imagenet')

    # Use io.BytesIO to pre-load the data into a buffer
    with open(trained_path, 'rb') as f:
        buffer = io.BytesIO(f.read())

    param_dict = torch.load(buffer, map_location='cuda' if torch.cuda.is_available() else 'cpu')


    for i in param_dict:
        if 'classifier' in i:
            continue
        reid_model.state_dict()[i].copy_(param_dict[i])

    if torch.cuda.device_count() > 1:
        reid_model = nn.DataParallel(model)

    return reid_model


def prepare_input_from_file(img_path, pid, camid, timestamp):
    return PersonReidModel.transform(read_image(img_path)), pid, camid, timestamp


def prepare_input_from_tensor(torch_tensor, pid, camid, timestamp):
    return torch_tensor, pid, camid, timestamp

    
class PersonReidModel():
    transform = create_transform()

    def __init__(self, pretrained_path, trained_path):
        self.model = create_reid_model(pretrained_path, trained_path)
        self.model.to('cuda')
        self.model.eval()


        # first inference took 2s to perform so put first inference to init 
        foo_tensor = torch.empty((3, 123,384))
        foo_input = [foo_tensor, None, None, None]
        self.perform_inference(foo_input)

    #perform inference
    #input là list gồm 4 phần tử lần lượt là 
    #0: torch.tensor
    #1: pid
    #2: camid
    #3: timestamp
    def perform_inference(self, input):
        image_tensor = input[0].unsqueeze(0).to('cuda')
        pid = input[1]
        camid = input[2]
        timestamp = input[3]
        feature_tensor = self.model(image_tensor)

        return feature_tensor, pid, camid, timestamp
