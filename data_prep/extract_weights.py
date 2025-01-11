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
from prettytable import PrettyTable


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

def extract_weights(model):
    # Extract weights from the model
    weights_dict = {}
    
    table = PrettyTable()
    table.field_names = ["Layer Name", "Number of Parameters"]
    
    layer_names = model.state_dict().keys()
    for layer_name in layer_names:
        if 'weight' in layer_name:
            layer_weights = model.state_dict()[layer_name].int_repr().flatten().numpy()
            normalized_array = layer_weights + 128
            weights_dict[layer_name] = normalized_array
            table.add_row([layer_name, len(normalized_array)])

    print(table)  

    return weights_dict


if __name__ == "__main__":
    # Extract and save weights
    results = []
    for model_name in models_dict:
        print("--------------------------------------------")
        print(f"Model: {model_name}")
        print("--------------------------------------------")
        weights = extract_weights(models_dict[model_name][0])

        for layer_name, values in weights.items():
            results.append({
                "model": model_name,
                "layer_name": layer_name,
                "type": "weights",
                "values": values
            })

    write_to_csv(results, "weights_all_layers.csv")
