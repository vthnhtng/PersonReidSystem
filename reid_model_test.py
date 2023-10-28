from reid_model import PersonReidModel
import time
import torch


start = time.time()

pretrained_path = "./pretrained_models/resnet50-19c8e357.pth"
trained_path = "./trained_models/market_resnet50_model_120_rank1_945.pth"

reid_model = PersonReidModel(pretrained_path, trained_path)
end = time.time()
print("init model time:" + str(end - start) + "s")

img_path1 = './test1.jpg'
img_path2 = './test2.jpg'
pid = 1
camid = 1

start = time.time()
data1 = reid_model.prepare_input(img_path1, pid, camid)
data2 = reid_model.prepare_input(img_path2, pid, camid)
end = time.time()
print("prepare data time:" + str(end - start) + "s")

# Perform inference
start = time.time()
feat1 = reid_model.perform_inference(data1)
feat2 = reid_model.perform_inference(data2)

feat1 = reid_model.L2_norm(feat1[0])
feat2 = reid_model.L2_norm(feat2[0])

end = time.time()


print("inference time:" + str(end - start) + "s")

# Calculate cosine similarity

start = time.time()
distance = reid_model.euclidean_distance(feat1, feat2)
end = time.time()

print("calc euclide time:" + str(end - start) + "s")
threshold = 1.05

print(distance)
if distance >= threshold:
  print("are not same person")
elif distance < threshold:
  print("are same person")



