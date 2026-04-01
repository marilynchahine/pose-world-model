#------------------------------------------#
#            Dataset preparation           #
#------------------------------------------#

# Human3.6M Dataset
#  each pose == 96 datapoints == 32 joints x 3D coordinates (x, y, z)


def reshape_pose(data):
    # return: (frames, 32 joints, 3 coords)
    return data.reshape(-1, 32, 3)

"""
remove redundant/static/useless joints
we keep 17 joints as per standard practice:

"""
