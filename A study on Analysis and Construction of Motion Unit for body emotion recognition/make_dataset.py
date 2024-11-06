import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F 
import glob
import os 
import sys
import re
from bvh import Bvh

file_path = "bvh/"
test_file_path = "bvh/20_YuzawaSetsuko/JP_20_anger_1_H.bvh"

def get_folder_name(file_path):
    folder_list = glob.glob(os.path.join(file_path, "*"))
    return folder_list

def get_file_name(bvh_file_path):
    file_list = glob.glob(os.path.join(bvh_file_path, "*.bvh"))
    return file_list

def get_bvh_data(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        motion_index = lines.index('MOTION\n')
        frame_time_index = motion_index + 2
        hierarcky = lines[0 : frame_time_index + 1]
        data_lines = lines[frame_time_index + 1:]
        data = np.array([list(map(float, line.split())) for line in data_lines])
    f.close()
    return hierarcky, data

def data_format(data):
    x, y, z = [], [], []
    for i in range (data.shape[0]):
        # x_rot, y_rot, z_rot = [data[i][0]], [data[i][1]], [data[i][2]]
        x_rot, y_rot, z_rot = [], [], []
        for j in range(1, int(data.shape[1] / 3), 1):
            z_rot.append(data[i][3 * j])
            y_rot.append(data[i][3 * j + 1])
            x_rot.append(data[i][3 * j + 2])
        x.append(x_rot)
        y.append(y_rot)
        z.append(z_rot)
    return [x, y, z]

def toEmo(str):
    return re.sub(r'_(.*)', "", re.sub(r'^JP_[0-9]+_', "", re.sub(r'^(.*)\\', "", str)))

def toEmotest(str):
    return re.sub(r'^(.*)\\', "", str).replace(".bvh", "")

emotion = {"anger":0, "contempt":1, "disgust":2, "fear":3, "gratitude":4, "guilt":5, "jealousy":6, "joy":7, "pride":8, "sadness":9, "shame":10, "surprise":11}

dataset = []
labelset = []

test = 0

folder_name_pre = get_folder_name(file_path)
use_file_index = [0, 2, 3, 4, 6, 7, 9, 11, 13, 14, 15, 16, 17, 18, 20, 22, 23, 24, 30, 31, 32, 33, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48]
folder_name = []

for i in range(len(use_file_index)):
    folder_name.append(folder_name_pre[use_file_index[i]])

for folder in folder_name:
    file_name = get_file_name(folder)
    print(folder)
    for file in file_name:
        print(file)
        try:
            label = emotion[toEmo(file)]
            _, data = get_bvh_data(file)
            if (999 < data.shape[0]) and (data.shape[0] < 2000):
                for i in range(data.shape[0]):
                    if data.shape[0] == 1000:
                        break
                    elif data.shape[0] == 1001:
                        data = data[1:]
                    else:
                        data = data[1:]
                        data = data[:-1]


                dataset.append(data_format(data))
                labelset.append(emotion[toEmo(file)])
                print("successful")
            else:
                print("Bad Data")
        except:
            print("No emo")


data_a = np.array(dataset)
print(data_a.shape)
np.save("train_data", data_a[0:2000])
np.save("train_label", np.array(labelset[0:2000]))
np.save("test_data", data_a[2000:])
np.save("test_label", np.array(labelset[2000:]))



        
        



# label = np.load("data/data/train_label.npy")
# print(label)