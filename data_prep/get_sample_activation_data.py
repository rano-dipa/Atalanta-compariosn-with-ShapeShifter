import tarfile
import os
import numpy as np
import matplotlib.pyplot as plt
import random

# Path to the compressed CIFAR-10 dataset
extract_dir = "/content/drive/MyDrive/CSCE_614/Project/cifar-10-batches-py/"  # Directory to extract files
output_dir = "/content/drive/MyDrive/CSCE_614/Project/sample_activation_data"  # Folder to save images

# Step 1: Unpickle CIFAR-10 data
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

# Load a batch file
batch_file = os.path.join(extract_dir, "data_batch_1")  # Load the first batch
data_dict = unpickle(batch_file)
images = data_dict[b'data']  # Image data
labels = data_dict[b'labels']  # Corresponding labels

# Step 2: Reshape images to (32, 32, 3) format
images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

# Step 3: Randomly select 10 images
indices = random.sample(range(len(images)), 10)
selected_images = images[indices]
selected_labels = [labels[i] for i in indices]

# Step 4: Display the selected images
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i, ax in enumerate(axes):
    ax.imshow(selected_images[i])
    ax.set_title(f"Label: {selected_labels[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()


# Step 5: Save selected images to the output folder
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create the folder if it doesn't exist

# Save images
for i, img in enumerate(selected_images):
    # Convert the image from NumPy array to PIL Image format
    image = Image.fromarray(img)
    
    # Save the image with a filename indicating the label
    image_name = f"image_{i}_label_{selected_labels[i]}.png"
    image_path = os.path.join(output_dir, image_name)
    image.save(image_path)

    print(f"Saved image {image_name} to {output_dir}")