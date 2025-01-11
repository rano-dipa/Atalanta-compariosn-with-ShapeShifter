# Atalanta
'Atalanta' vs 'ShapeShifter'- hardware compression for quantized DL models to improve off-die memory access

# Project Description

This project implements Atalanta, a novel lossless compression algorithm designed specifically for tensor data in fixed-point quantized deep neural networks. It combines arithmetic coding with hardware-efficient design, and provides a solution optimized for memory efficiency, lowe latency, and lower power consumption. The primary task for us in this project has been, to evaluate Atalanta's performance in terms of compression ratios, on 3 DNN models - ResNet50, GoogLeNet, and MovileNetV2. Also this has been compared to a baseline ShapeShifter, which is an adaptive compressive approach.
The repository consists of full implementation of:
1. Extraction of weights and activations from the three models.
2. ShapeShifter Encoder
3. Probability Table generation and Parsing
4. Atalanta Encoder & Decoder
5. Comparative Analysis and Graphs plotting


# run script to view the results
```
./run_script/run_script.ipynb
Navigate to the script and run each cell to view the results.
```

# Results

## Figure 3 Weights
![Alt text](data/final_results/Figure_3_weights.png?raw=true "Figure 3 Activations")

## Figure 3 Activations
![Alt text](data/final_results/Figure_3_activations.png?raw=true "Figure 3 Weights")

## Table 4 Compression (Atalanta over Shapeshifter)

| Network      | Dataset  | Application    | Data Type | Quantizer   | Model        | Compression %         |
|--------------|----------|----------------|-----------|-------------|--------------|-----------------------|
|||||||Atalanta over Shapeshifter|
| GoogLeNet    | ImageNet | Classification | int8      | Torchvision | activations  | 0.008685403452114188  |
| GoogLeNet    | ImageNet | Classification | int8      | Torchvision | weights      | 14.569586779026528    |
| Mobilenet_v2 | ImageNet | Classification | int8      | Torchvision | activations  | -0.5752121834045524   |
| Mobilenet_v2 | ImageNet | Classification | int8      | Torchvision | weights      | 19.522418344835778    |
| Resnet50     | ImageNet | Classification | int8      | Torchvision | activations  | 13.812717864888596    |
| Resnet50     | ImageNet | Classification | int8      | Torchvision | weights      | 16.935287329217065    |



# Project Structure
```
.
├── README.md
├── atalanta
│   ├── __init__.py
│   ├── atalanta_decode.py
│   ├── atalanta_encode.py
│   ├── codec.py
│   ├── probability_table.py
│   └── run_atalanta.py
├── comparison
│   ├── __init__.py
│   └── comparison.py
├── data
│   ├── atalanta_encoded_outputs
│   ├── cifar-10-batches-py
│   ├── comparison_reports
│   ├── extracted_weights_and_activations
│   ├── final_results
│   ├── probability_tables
│   │   ├── activations_probability_tables
│   │   └── weights_probability_tables
│   ├── sample_activations_input_data
│   ├── shapeshifter_encoded_outputs
│   └── timing_reports
├── data_prep
│   ├── __init__.py
│   ├── atalanta_numpy.py
│   ├── atalanta_search.py
│   ├── extract_activations.py
│   ├── extract_weights.py
│   ├── get_sample_activation_data.py
│   └── probability_table_gen.py
├── examples
│   ├── atalanta_encode_decode
│   │   └── encode_decode.ipynb
│   ├── compare
│   │   └── comparison.ipynb
│   ├── sample_data
│   └── shapeshifter_encode
│       └── shapeshifter_encode.ipynb
├── requirements.txt
├── run_script
│   └── run_script.ipynb
└── shapeshifter
    ├── __init__.py
    └── shapeshifter_encode.py
``` 

# Required Libraries
``` 
appnope==0.1.4
asttokens==3.0.0
attrs==24.2.0
backcall==0.2.0
beautifulsoup4==4.12.3
bleach==6.2.0
certifi==2024.8.30
charset-normalizer==3.4.0
click==8.1.7
cloudpickle==3.1.0
comm==0.2.2
contourpy==1.3.0
cycler==0.12.1
dask==2024.8.0
debugpy==1.8.9
decorator==5.1.1
defusedxml==0.7.1
docopt==0.6.2
executing==2.1.0
fastjsonschema==2.21.0
filelock==3.16.1
fonttools==4.55.1
fsspec==2024.10.0
idna==3.10
importlib_metadata==8.5.0
importlib_resources==6.4.5
ipykernel==6.29.5
ipython==8.12.3
jedi==0.19.2
Jinja2==3.1.4
jsonschema==4.23.0
jsonschema-specifications==2024.10.1
jupyter_client==8.6.3
jupyter_core==5.7.2
jupyterlab_pygments==0.3.0
kiwisolver==1.4.7
locket==1.0.0
MarkupSafe==3.0.2
matplotlib==3.9.3
matplotlib-inline==0.1.7
mistune==3.0.2
mpmath==1.3.0
nbclient==0.10.1
nbconvert==7.16.4
nbformat==5.10.4
nest-asyncio==1.6.0
networkx==3.2.1
numpy==2.0.2
packaging==24.2
pandas==2.2.3
pandocfilters==1.5.1
parso==0.8.4
partd==1.4.2
pexpect==4.9.0
pickleshare==0.7.5
pillow==11.0.0
pipreqs==0.5.0
platformdirs==4.3.6
prompt_toolkit==3.0.48
psutil==6.1.0
ptyprocess==0.7.0
pure_eval==0.2.3
Pygments==2.18.0
pyparsing==3.2.0
python-dateutil==2.9.0.post0
pytz==2024.2
PyYAML==6.0.2
pyzmq==26.2.0
referencing==0.35.1
requests==2.32.3
rpds-py==0.21.0
six==1.16.0
soupsieve==2.6
stack-data==0.6.3
sympy==1.13.1
tabulate==0.9.0
tinycss2==1.4.0
toolz==1.0.0
torch==2.5.1
torchvision==0.20.1
tornado==6.4.2
traitlets==5.14.3
typing_extensions==4.12.2
tzdata==2024.2
urllib3==2.2.3
wcwidth==0.2.13
webencodings==0.5.1
yarg==0.1.9
zipp==3.21.0
``` 

To install these dependencies, run:
pip install -r requirements.txt

# Paper Source
``` 
The Atalanta framework and its cocnepts are taken from the paper:
"Atalanta: A Bit is Worth a 'Thousand' Tensor Values
Authors: Alberto Delmas Lascorz, Mostafa Mahmoud, Ali hadi Zadeh, Milos Nikolic, Kareem Ibrahim, Christina Giannoula, Ameer Abdelhadi, Andreas Moshovos
Conference: ASPLOS '24: 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2

The ShapeShifter encoder is taken from the paper:
"ShapeShifter: Enabling Fine-Grain Data Width Adaptation in Deep Learning"
Authors: Alberto Delmas Lascorz, Sayeh Sharify, Isak Edo, Dylan Malone Stuart, Omar Mohamed Awad, Patrick Judd, Mostafa Mahmoud, Milos Nikoloc, Kevin Siu, Zissis Poulos, Andreas Moshovos
Conference: Micro '52: Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture
``` 
