import reid_model
from reid_model import PersonReidModel
import time
import torch


start = time.time()

pretrained_path = "./pretrained_models/resnet50-19c8e357.pth"
trained_path = "./trained_models/market_resnet50_model_120_rank1_945.pth"

model = PersonReidModel(pretrained_path, trained_path)
end = time.time()
print("init model time:" + str(end - start) + "s")

img_path1 = './img/23.jpg'
img_path2 = './img/27.jpg'
pid = 1
camid = 1
timestamp = time.time()

start = time.time()
data1 = reid_model.prepare_input_from_file(img_path1, pid, camid, timestamp) # return tensor, pid, camid
data2 = reid_model.prepare_input_from_file(img_path2, pid, camid, timestamp)
end = time.time()
print("prepare data time:" + str(end - start) + "s")

# Perform inference
start = time.time()
output1 = model.perform_inference(data1)
output2 = model.perform_inference(data2)
end = time.time()


print("inference time:" + str(end - start) + "s")

# Calculate cosine similarity
threshold = 1.05

if reid_model.are_same_person(output1[0], output2[0], threshold):
  print("Are same person")
else:
  print("Are not same person")
