import cdflib
import numpy as np
import os

#------------------------------------------#
#            Dataset preparation           #
#------------------------------------------#

# Human3.6M Dataset
#  each pose == 96 datapoints == 32 joints x 3D coordinates (x, y, z)




def load_subject(subject_path):
    cdf = cdflib.CDF(subject_path)
    poses = cdf.varget("Pose")  # shape: (frames, 96)
    return poses

"""
remove redundant/static/useless joints
we keep 17 joints as per standard practice:
0 -> pelvis
1 -> spine
2 -> thorax/ upper spine
3 -> neck/ head base
6 -> right hip
7 -> right knee
8 -> right ankle
12 -> left hip
13 -> left knee
14 -> left ankle
15 -> left foot/toe 
17 -> left shoulder
18 -> left elbow
19 -> left wrist
25 -> right shoulder
26 -> right elbow
27 -> right wrist
"""

# number of subjects that motion was recorded from
N = 11

# standard 17 joints kept
JOINTS_17 = [
    0, 1, 2, 3,     # torso
    6, 7, 8,        # right leg
    12, 13, 14, 15, # left leg (left foot included for better foot motion)
    17, 18, 19,     # left arm
    25, 26, 27      # right arm
]

def reshape_pose(poses):
    # return: (frames, 32 joints, 3 coords)
    poses_reshaped = poses.reshape(-1, 32, 3)
    return poses_reshaped

def filter_joints(poses_reshaped):
    poses_filtered = poses_reshaped[:, JOINTS_17, :]
    return poses_filtered

# make poses root-relative
# -> each joint has an (x, y, z) coordinate in space
# -> making the coordinates of all joints root-relative means we're removing 
#    the global position of the person in space
# -> centers the position of all other joints around the pelvis 
# -> pelvis now has coordinates (0, 0, 0) and others have (x_joint _ x_pelvis, y_joint _ y_pelvis, z_joint _ z_pelvis)
def make_root_relative(poses_filtered):
    root = poses_filtered[:, 0:1, :]  # pelvis
    poses_relative = poses_filtered - root
    return poses_relative

def normalize(poses_relative):
    mean = poses_relative.mean(axis=(0,1), keepdims=True)
    std = poses_relative.std(axis=(0,1), keepdims=True) + 1e-8
    poses_normalized = (poses_relative - mean) / std
    return poses_normalized, mean, std

# create training frame sequences
# -> input_len = frames that will correspond to the past sequences
# -> pred_len = frames that will correspond to the future sequences to predict
# -> for each frame i take [i : i+input_len] as input sequence 
#    and [i+input_len : i+input_len+pred_len] as sequence to predict
# return a list containing all past sequences and a list containing all future sequences
def create_sequences(poses_normalized, input_len=20, pred_len=10):
    X, Y = [], []
    
    for i in range(len(poses_normalized) - input_len - pred_len):
        X.append(poses_normalized[i:i+input_len])
        Y.append(poses_normalized[i+input_len:i+input_len+pred_len])

    # [N, input_len, 17, 3]
    X = np.array(X)
    # [N, pred_len, 17, 3]
    Y = np.array(Y)
    
    return X, Y

def flatten_joints(X, Y, input_len = 20, pred_len = 10):
    X_flat = X.reshape(N, input_len, -1)
    Y_flat = Y.reshape(N, pred_len, -1)
    return X_flat, Y_flat




input_len = 20
pred_len = 10
local_path = "C:/Users/maril/OneDrive/Desktop/GitHub/HRI_Projects/PoseWorldModel"

# Train/Test/Val standard split
# Train: S1, S5, S6, S7
# Val: S8
# Test: S9, S11
TRAIN_SUBJECTS = ["S1", "S5", "S6", "S7"]
VAL_SUBJECTS   = ["S8"]
TEST_SUBJECTS  = ["S9", "S11"]

# Training data pre-processing
train_path = local_path + "/data/"

train_poses = []
for subj in TRAIN_SUBJECTS:
    poses_train = load_subject(train_path + subj)
    poses_reshaped_train = reshape_pose(poses_train)
    poses_filtered_train = filter_joints(poses_reshaped_train)
    poses_relative_train = make_root_relative(poses_filtered_train)
    train_poses = train_poses.append(poses_relative_train, axis = 0)

poses_normalized_train, mean_train, std_train = normalize(train_poses)
X_train, Y_train = create_sequences(poses_normalized_train, input_len, pred_len)
X_flat_train, Y_flat_train = flatten_joints(X_train, Y_train, input_len, pred_len)

np.save(local_path + "data/processed/train/X.npy", X_train)
np.save(local_path + "data/processed/train/Y.npy", Y_train)
np.save(local_path + "data/processed/train/mean.npy", mean_train)
np.save(local_path + "data/processed/train/std.npy", std_train)

# Validation data pre-processing
val_path = local_path + "/data/"

val_poses = []
for subj in VAL_SUBJECTS:
    poses_val = load_subject(val_path + subj)
    poses_reshaped_val = reshape_pose(poses_val)
    poses_filtered_val = filter_joints(poses_reshaped_val)
    poses_relative_val = make_root_relative(poses_filtered_val)
    val_poses = val_poses.append(poses_relative_val, axis = 0)

poses_normalized_val = val_poses - mean_train / std_train
X_val, Y_val = create_sequences(poses_normalized_val, input_len, pred_len)
X_flat_val, Y_flat_val = flatten_joints(X_val, Y_val, input_len, pred_len)

np.save(local_path + "data/processed/val/X.npy", X_val)
np.save(local_path + "data/processed/val/Y.npy", Y_val)

# Testing data pre-processing
test_path = local_path + "/data/"

test_poses = []
for subj in TEST_SUBJECTS:
    poses_test = load_subject(test_path + subj)
    poses_reshaped_test = reshape_pose(poses_test)
    poses_filtered_test = filter_joints(poses_reshaped_test)
    poses_relative_test = make_root_relative(poses_filtered_test)
    test_poses = test_poses.append(poses_relative_test, axis = 0)

poses_normalized_test = test_poses - mean_train / std_train
X_test, Y_test = create_sequences(poses_normalized_test, input_len, pred_len)
X_flat_test, Y_flat_test = flatten_joints(X_test, Y_test, input_len, pred_len)

np.save(local_path + "data/processed/test/X.npy", X_test)
np.save(local_path + "data/processed/test/Y.npy", Y_test)