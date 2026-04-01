#-----------------------------------------------------------------#
#                Human Pose Prediction World Model                #
#-----------------------------------------------------------------#

#----------------------------Pipeline-----------------------------#
# pose sequence
#  → pose embedding
#  → temporal world model
#  → latent future states
#  → decoded future poses
#-----------------------------------------------------------------#

# Version 1
# [B, T, J*D]
# → frame encoder == small MLP over joint coordinates per frame
# → temporal model == transformer encoder over the sequence
# → decoder == MLP to reconstruct future joint coordinates
# → [B, T_future, J*D]

# Where: B = batch, T = input time steps, J = number of joints, D = 2 or 3 coordinates

