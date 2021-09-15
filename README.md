# Predicting Material properties of objects and liquids inside transparent vessels, and the vessel surface material, from a single image
Given an image containing transparent containers with something inside (liquid or object), predict the Material properties of the vessel content and the vessel surface.
The properties predicted include: Color (RGB), Transparency (Transmission), Roughness, Reflectiveness( Metallic), these properties are based on the [Blender (CGI) Principle BSDF node description of materials](https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/principled.html)

Training the net was done using the TransProteus dataset [Full Dataset 1](https://e.pcloud.link/publink/show?code=kZfx55Zx1GOrl4aUwXDrifAHUPSt7QUAIfV),  [Full DataSet Link2](https://icedrive.net/1/6cZbP5dkNG), [Subset](https://zenodo.org/record/5508261#.YUGsd3tE1H4) 

See paper []() for more details


The same code with trained model included (run out of the box) could be download from: [1](https://e.pcloud.link/publink/show?code=XZDq55Z5k1jugRCGbj4OCAmdIL9M4vOo8Py),[2](https://icedrive.net/0/fbQuVti8WO).
![](/Figure1.jpg)
Figure 1. Structure of the net for predicting materials properties of the vessel content and vessel surface from an image.

  
# Requirements
## Hardware
For using the trained net for prediction, no specific hardware is needed, but the net will run much faster on Nvidia GPU.

For training the net, an Nvidia GPU is needed (the net was trained on RTX 3090)

## Setup
Create a environment with the required dependencies ([Pytorch](https://pytorch.org/), torchvision, scipy and OpenCV): *conda env create -f environment.yml*

## Software:
This network was run with Python 3.88 [Anaconda](https://www.anaconda.com/download/) with  [Pytorch 1.8](https://pytorch.org/) and OpenCV* package.
* Installing opencv for conda can usually be done using: "pip install opencv-python" or "conda install opencv"

# Inference: running the trained net on  a single image

1. Train net or download code with pre-trained net weight from [1](https://e.pcloud.link/publink/show?code=XZDq55Z5k1jugRCGbj4OCAmdIL9M4vOo8Py),[2](https://icedrive.net/0/fbQuVti8WO).
2. Open RunOnImage.py
3. Set image path to InputImage parameter (or use the pre-set example)
4. Set the path to the Vessel Mask image to InputMask parameter (or use the pre-set example)   
4. Set the path to the trained net weights  file in: Trained_model_path  (If you downloaded the code with the pre-train network from [here]() the model path is already set) 
5. Run script to get predicted material properties displayed on the screen and terminal.
Additional optional parameters: 
UseGPU: decide whether to use GPU hardware (True/False).

# For training and evaluating download TransProteus and LabPics

1. Download and extract the TransProteus dataset  [Full Dataset 1](https://e.pcloud.link/publink/show?code=kZfx55Zx1GOrl4aUwXDrifAHUPSt7QUAIfV),  [Full DataSet Link2](https://icedrive.net/1/6cZbP5dkNG), [Subset](https://zenodo.org/record/5508261#.YUGsd3tE1H4) 


## Training

1. Open Train.py
3. Set the path to TransProteus train folders in the dictionary "TransProteusFolder" in the input parameter section (the dictionary keys names don't matter). 
Note that this dictionary can get several folders, and each folder can be added more than once. If a folder appears twice, it will be used during training twice as much.
(By default, this parameter point to the example folder supplied with the code)
3. Run the script
4. The trained net weight will appear in the folder defined in the  logs_dir 


## Evaluating 

1. Train net or download code with pre-trained net weight from [1](https://e.pcloud.link/publink/show?code=XZDq55Z5k1jugRCGbj4OCAmdIL9M4vOo8Py),[2](https://icedrive.net/0/fbQuVti8WO). 
2. Open file EvaluateNet.py
3. Set a path to the trained net weights  file in: Trained_model_path  (If you downloaded the code with the pre-train network from [here]() the model path is already set)
4. Set Test data folder  path to the  TestFolder parameter (TransProteus)
5. Optional: In the InputMask parameter, set either "VesselMask" or "ContentMask" depending on whether the net is supposed to receive the region of the vessel or the region of the vessel content as an input (By default its the region of the vessel)
6. Run the script

For other parameters, see the Input parameters section.


## More Info 
See paper ()[] For more details


