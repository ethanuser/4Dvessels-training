import json
import numpy as np
import os
import pyvista as pv
from PIL import Image

SCALE = 0.5

def load_camera_images(directory, size=(50, 50)):
    images = {}
    for filename in os.listdir(directory):
        # if filename.startswith("camera_") and filename.endswith(".png"):
        if filename.startswith("view_") and filename.endswith(".png"):
            camera_num = int(filename.split("_")[1][0]) - 1
            img_path = os.path.join(directory, filename)
            # Open and resize the image
            with Image.open(img_path) as img:
                img_resized = img.resize(size, Image.LANCZOS)
                images[camera_num] = np.array(img_resized)
    return images

# Load the JSON file
with open('C:\\Users\\elega\\Documents\\GitHub\\3D-vessel-imaging\\Preprocess\\Camera Calibration\\calibrations\\camera_parameters.json') as f:
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

# Load camera images
image_directory = 'C:\\Users\\elega\\Documents\\GitHub\\3D-vessel-imaging\\Preprocess\\Camera Calibration\\calibrations'
camera_images = load_camera_images(image_directory, size=(800, 800))

# Create a pyvista plotter object
plotter = pv.Plotter()

def plot_image(plotter, image_array, position, rotation, img_size=10 * SCALE):
    # Create a grid for the image
    x = np.linspace(-img_size/2, img_size/2, image_array.shape[1])
    y = np.linspace(-img_size/2, img_size/2, image_array.shape[0])
    x, y = np.meshgrid(x, y)
    z = np.zeros_like(x)
    
    # Flatten the grid points
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
    
    # Apply rotation and translation
    rotated_points = np.dot(points, rotation.T) + position
    
    # Reshape back to grid
    x_rot = rotated_points[:, 0].reshape(x.shape)
    y_rot = rotated_points[:, 1].reshape(y.shape)
    z_rot = rotated_points[:, 2].reshape(z.shape)
    
    # Create pyvista StructuredGrid
    grid = pv.StructuredGrid(x_rot, y_rot, z_rot)
    
    # Define the plane for texture mapping
    origin = position - rotation[:, 0] * img_size / 2 - rotation[:, 1] * img_size / 2
    point_u = origin + rotation[:, 0] * img_size
    point_v = origin + rotation[:, 1] * img_size
    
    # Apply texture coordinates
    grid.texture_map_to_plane(origin=origin, point_u=point_u, point_v=point_v, inplace=True)
    
    # Add texture
    texture = pv.numpy_to_texture(image_array)
    plotter.add_mesh(grid, texture=texture, show_edges=False)

def plot_line_segment(plotter, start_point, direction, length, color):
    end_point = start_point + direction * length
    line = np.array([start_point, end_point])
    plotter.add_lines(line, color=color, width=3)

# Plot each camera
for i, (pos, rot) in enumerate(zip(camera_positions, rotation_vectors)):
    # Plot camera position
    plotter.add_mesh(pv.Sphere(center=pos, radius=0.1 * SCALE), color='red')
    plotter.add_point_labels([pos], [f'{i}'], point_size=20)
    
    # Plot rotation vectors as line segments
    plot_line_segment(plotter, pos, rot[:, 0], 5 * SCALE, 'red')
    plot_line_segment(plotter, pos, rot[:, 1], 5 * SCALE, 'green')
    plot_line_segment(plotter, pos, rot[:, 2], -30 * SCALE, 'blue')  # Longer blue line
    
    # Display camera image if available
    if i in camera_images:
        img_array = camera_images[i]
        image_position = pos - rot[:, 2] * 15.0 * SCALE  # Place image slightly in front of the camera
        plot_image(plotter, img_array, image_position, rot)

plotter.add_axes()
plotter.show()
