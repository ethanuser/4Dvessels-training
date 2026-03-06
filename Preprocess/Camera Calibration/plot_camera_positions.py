import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the JSON file
with open('C:\\Users\\EMC 1\\Documents\\GitHub\\3D-vessel-imaging\\Preprocess\\Camera Calibration\\calibrations\\camera_parameters.json') as f:
    data = json.load(f)

# Extract the camera positions and rotation matrices from the transform matrices
camera_positions = []
rotation_vectors = []

for camera_key in data:
    if camera_key.startswith("Camera"):
        camera_data = data[camera_key]
        transform_matrix = np.array(camera_data['transform_matrix'])
        position = transform_matrix[:3, 3]  # Extract the translation vector
        rotation_matrix = transform_matrix[:3, :3]  # Extract the rotation matrix
        camera_positions.append(position)
        rotation_vectors.append(rotation_matrix)

camera_positions = np.array(camera_positions)

# Generate colors from red to black based on the number of cameras
num_cameras = len(camera_positions)
colors = np.array([(1 - i / num_cameras, 0, 0) for i in range(num_cameras)])

# Plot the camera positions in 3D space with color gradient
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# vector_length = 0.09
vector_length = 70

for i, (pos, rot) in enumerate(zip(camera_positions, rotation_vectors)):
    # Plot camera position
    ax.scatter(pos[0], pos[1], pos[2], color=colors[i], marker='o')
    ax.text(pos[0], pos[1], pos[2], f'{i}', color='red')
    # Plot rotation vectors
    ax.quiver(pos[0], pos[1], pos[2], rot[0, 0], rot[1, 0], rot[2, 0], color='r', length=vector_length, normalize=True)
    ax.quiver(pos[0], pos[1], pos[2], rot[0, 1], rot[1, 1], rot[2, 1], color='g', length=vector_length, normalize=True)
    ax.plot([pos[0], pos[0] + rot[0, 2] * -6.4 * vector_length],
            [pos[1], pos[1] + rot[1, 2] * -6.4 * vector_length],
            [pos[2], pos[2] + rot[2, 2] * -6.4 * vector_length], color='b')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set equal aspect
max_range = np.array([camera_positions[:, 0].max() - camera_positions[:, 0].min(),
                      camera_positions[:, 1].max() - camera_positions[:, 1].min(),
                      camera_positions[:, 2].max() - camera_positions[:, 2].min()]).max()
Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (camera_positions[:, 0].max() + camera_positions[:, 0].min())
Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (camera_positions[:, 1].max() + camera_positions[:, 1].min())
Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (camera_positions[:, 2].max() + camera_positions[:, 2].min())

# Comment out the next line if you do not want to see this bounding box
for xb, yb, zb in zip(Xb, Yb, Zb):
    ax.plot([xb], [yb], [zb], 'w')

ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

plt.title(f"Cameras 1-{num_cameras} Poses")
plt.show()
