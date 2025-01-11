import torchvision.models.quantization as models
import torch
import numpy as np
import csv
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from collections import defaultdict
from PIL import Image
import os

sample_data_dir = "/content/drive/MyDrive/CSCE_614/Project/sample_activation_data"

resnet50_weights = models.ResNet50_QuantizedWeights.DEFAULT
mobilenet_v2_weights = models.MobileNet_V2_QuantizedWeights.DEFAULT
googlenet_weights = models.GoogLeNet_QuantizedWeights.DEFAULT

# Load pre-trained quantized models
models_dict = {
    "Resnet50": [models.resnet50(weights = resnet50_weights, quantize=True), resnet50_weights],
    "Mobilenet_v2": [models.mobilenet_v2(weights = mobilenet_v2_weights, quantize=True), mobilenet_v2_weights],
    "GoogLeNet": [models.googlenet(weights = googlenet_weights, quantize=True), googlenet_weights],
}

def write_to_csv(results, filename):
    # Filepath for the CSV
    output_file = filename

    # Write the data to the CSV
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(["Model", "Layer", "Type", "Values"])

        # Write each row
        for entry in results:
            row = [
                entry["model"],
                entry["layer_name"],
                entry["type"],
            ] + entry["values"].tolist()
            
            writer.writerow(row)

def load_images_from_directory(directory):
    image_paths = []    
    # Check if the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} not found.")
    
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Check if it's a file (not a subdirectory)
        if os.path.isfile(file_path) and filename.endswith('.png'):
            image_paths.append(file_path)
    return image_paths

def process_images(model_weights, images):
    preprocess = model_weights.transforms()

    processed_images = []
    for image in images:
        # Load the image
        image = Image.open(image)
        image = preprocess(image)
        image = image.unsqueeze(0) # Add the batch dimension (N, C, H, W)
        processed_images.append(image) 
    return processed_images

# Function to extract activations using hooks
def extract_activations(model, image_batch):
  
    activations = defaultdict(list)

    def add_hooks():
        def hook_fn(name):
            def fn(model, input, output):
                activations[name].append(output.detach().int_repr().flatten().numpy())
            return fn
        
        def get_layer_type(model, layer_name):
            layer = model
            for name in layer_name.split('.'):
                layer = getattr(layer, str(name))
            return layer

        hooks = []
        state_dict_keys = model.state_dict().keys()
        layer_names = ['.'.join(key.split('.')[:-1]) for key in state_dict_keys if '.weight' in key]

        layer_map = {}

        for layer_name in layer_names:
            layer_type = get_layer_type(model, layer_name)
            hooks.append(layer_type.register_forward_hook(hook_fn(layer_type)))
            layer_map[layer_type] = layer_name
        return hooks, layer_map

    def remove_hooks(hooks):
        for hook in hooks:
            hook.remove()

    def run_inference():
        for image in image_batch:
            with torch.no_grad():
                model(image)
        
    hooks, layer_map = add_hooks()
    run_inference()
    remove_hooks(hooks)

    # Combine activations per layer
    final_activations = {layer: np.concatenate(activation) for layer, activation in activations.items()}
    
    table = PrettyTable()
    table.field_names = ["Layer Name", "Number of Activations"]

    for layer_type, values in final_activations.items():
        table.add_row([layer_map[layer_type], len(values)])

    print(table)

    return final_activations

if __name__ == "__main__":
    # Extract and save activations
    results = []
    images = load_images_from_directory(sample_data_dir)
    for model_name in models_dict:
        print("--------------------------------------------")
        print(f"Model: {model_name}")
        print("--------------------------------------------")
        processed_images = process_images(models_dict[model_name][1], images)
        activations = extract_activations(models_dict[model_name][0], processed_images)
        for layer_name, values in activations.items():
            results.append({
                "model": model_name,
                "layer_name": layer_name,
                "type": "activations",
                "values": values
            })

    write_to_csv(results, "activations_all_layers.csv")
