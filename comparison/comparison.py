import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV files
shapeshifter_file = '/content/Shape_Shifter_weights_compression_analysis.csv'
atalanta_file = '/content/atalanta_summary.csv'

# Read the data
shapeshifter_df = pd.read_csv(shapeshifter_file)
atalanta_df = pd.read_csv(atalanta_file)

# Normalize column names
shapeshifter_df.columns = shapeshifter_df.columns.str.strip().str.lower()
atalanta_df.columns = atalanta_df.columns.str.strip().str.lower()

# Models to analyze
models_to_keep = ['resnet50', 'googlenet', 'mobilenetv2']

# Filter relevant rows
shapeshifter_df = shapeshifter_df[shapeshifter_df['model'].str.lower().isin(models_to_keep)]
atalanta_df = atalanta_df[atalanta_df['model_name'].str.lower().isin(models_to_keep)]

# Calculate ShapeShifter compression ratios and memory efficiencies
shapeshifter_compression_ratios = shapeshifter_df.groupby('model')['ratio'].mean().to_dict()
shapeshifter_memory_efficiencies = (
    (shapeshifter_df['orig size (bits)'] - shapeshifter_df['comp size (bits)'])
    / shapeshifter_df['orig size (bits)']
).groupby(shapeshifter_df['model']).mean().to_dict()

# Calculate Atalanta compression ratios and memory efficiencies
atalanta_compression_ratios = atalanta_df.groupby('model_name')['compression'].mean().to_dict()
atalanta_memory_efficiencies = (
    (atalanta_df['before compression'] - atalanta_df['after compression'])
    / atalanta_df['before compression']
).groupby(atalanta_df['model_name']).mean().to_dict()

# Ensure consistency of model names
shapeshifter_compression_ratios = {k.lower(): v for k, v in shapeshifter_compression_ratios.items()}
atalanta_compression_ratios = {k.lower(): v for k, v in atalanta_compression_ratios.items()}
shapeshifter_memory_efficiencies = {k.lower(): v for k, v in shapeshifter_memory_efficiencies.items()}
atalanta_memory_efficiencies = {k.lower(): v for k, v in atalanta_memory_efficiencies.items()}

# Prepare data for plotting
compression_ratios = {
    model: [
        shapeshifter_compression_ratios.get(model, np.nan),
        atalanta_compression_ratios.get(model, np.nan),
    ]
    for model in models_to_keep
}
memory_efficiencies = {
    model: [
        shapeshifter_memory_efficiencies.get(model, np.nan),
        atalanta_memory_efficiencies.get(model, np.nan),
    ]
    for model in models_to_keep
}

# Plot settings
x = np.arange(len(models_to_keep))
width = 0.25

# Calculate reciprocal compression ratios for plotting
shapeshifter_ratios = [1 / compression_ratios[model][0] if compression_ratios[model][0] else np.nan for model in models_to_keep]
atalanta_ratios = [1 / compression_ratios[model][1] if compression_ratios[model][1] else np.nan for model in models_to_keep]

# Plot Compression Ratios
plt.figure(figsize=(12, 7))
plt.bar(x - width / 2, shapeshifter_ratios, width, label='ShapeShifter', color='coral', alpha=0.8)
plt.bar(x + width / 2, atalanta_ratios, width, label='Atalanta', color='indigo', alpha=0.8)
plt.ylabel('Compression Ratio', fontsize=12)  # Keep the same label
plt.xlabel('Models', fontsize=12)
plt.title('Compression Ratio Comparison by Model', fontsize=14, fontweight='bold')
plt.xticks(x, models_to_keep, fontsize=10)
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot Memory Efficiencies
plt.figure(figsize=(12, 7))
shapeshifter_eff = [memory_efficiencies[model][0] for model in models_to_keep]
atalanta_eff = [memory_efficiencies[model][1] for model in models_to_keep]

plt.bar(x - width / 2, shapeshifter_eff, width, label='ShapeShifter', color='coral', alpha=0.8)
plt.bar(x + width / 2, atalanta_eff, width, label='Atalanta', color='indigo', alpha=0.8)
plt.ylabel('Memory Efficiency', fontsize=12)
plt.xlabel('Models', fontsize=12)
plt.title('Memory Efficiency Comparison by Model', fontsize=14, fontweight='bold')
plt.xticks(x, models_to_keep, fontsize=10)
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Print comparison table
comparison_df = pd.DataFrame({
    'Model': models_to_keep,
    'ShapeShifter_Weight_Comp': shapeshifter_ratios,
    'Atalanta_Weight_Comp': atalanta_ratios,
    'ShapeShifter_Weight_Eff': shapeshifter_eff,
    'Atalanta_Weight_Eff': atalanta_eff,
})
print(comparison_df.to_string(index=False))
