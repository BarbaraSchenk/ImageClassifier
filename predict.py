# Importing requiered Packages
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np
import argparse
import json
from workspace_utils import active_session


def args_paser():
    paser = argparse.ArgumentParser(description='Parameters for Predicting')

    # Input paramters image and model that should be used for prediction
    paser.add_argument('image', type=str, default='flowers/valid/49/image_06235.jpg', help='image for prediction') # input image for making prediction
    paser.add_argument('checkpoint', type=str, default='checkpoint.pth', help='directory of checkpoint') # checkpoint that should be used as model
    
    # Additional Parameters
    paser.add_argument('--top_k', type=int, default=10, help='Return top K most likely classes')
    paser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Use a mapping of categories to real names')

    # Usage of GPU or CPU (default is GPU)
    paser.add_argument('--gpu', type=bool, default='True', help='True: GPU, False: CPU')

    args = paser.parse_args()
    return args

def set_device(user_set_gpu):
    # Sets device to GPU (if desired by user and available) or CPU
    if user_set_gpu and torch.cuda.is_available():
        # User decided to use GPU and GPU is available
        device = torch.device("cuda")
        print("GPU available - Device is set to GPU")
    else:
        device = torch.device("cpu")
        print("Device is set to CPU")
    return device

def load_checkpoint(load_dir):
    # loads saved checkpoint and creates model accordingly

    checkpoint = torch.load(load_dir)

    if checkpoint['arch'] == 'resnet50':
        model = models.resnet50(pretrained=True) # Use pre-trained resnet50 model as basis
        model.fc = checkpoint['classifier'] # use previously defined classifier architecture
        model.fc.load_state_dict(checkpoint['classifier_state_dict']) #set weigths/bias resulting from previous training
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True) # Use pre-trained resnet50 model as basis
        model.classifier = checkpoint['classifier'] # use previously defined classifier architecture
        model.classifier.load_state_dict(checkpoint['classifier_state_dict']) #set weigths/bias resulting from previous training

    model.class_to_idx = checkpoint['class_to_idx']
    for param in model.parameters():
        param.requires_grad = False

    print("Model loaded from checkpoint {}".format(load_dir))
    return model 

def process_image(image):
    '''Modifies the input image and returns a numpy array'''
    # RESIZE image with 256 pixels on shortest side
    # CROP image 224x224 pixel around center
    preprocessing = transforms.Compose([
                        transforms.Resize(256), 
                        transforms.CenterCrop(224),
                        transforms.ToTensor()])
    
    pil_img = Image.open(image)
    img_tensor = preprocessing(pil_img)
    
    # Normalize Image
    np_image = np.array(img_tensor)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return np_image

def predict(device, image_path, model, topk):
    # Predict the class (or classes) of an image using a trained deep learning model.

    print("Making prediction for image {}".format(image_path))
    # Model
    model.to(device)
    model.eval() # switch to evaluation mode --> dropout is deactivated
    
    # Image
    np_image = process_image(image_path) # numpy array image
    tensor_image =  torch.from_numpy(np_image).type(torch.FloatTensor).to(device) # converty np image to tensor
    tensor_image = tensor_image.unsqueeze(0)
    
    with torch.no_grad(): # turn off tracking of gradients
        prediction = model.forward(tensor_image)
        
        # Get top-5 probabilities and classes
        ps = torch.exp(prediction)
        top_ps, top_idx = ps.topk(topk, dim=1)
        
        # Convert tensors to lists
        top_ps = top_ps.tolist()[0]
        top_idx = top_idx.tolist()[0]        
        
        # Get classes from indices
        idx_to_class = {model.class_to_idx[k] : k for k in model.class_to_idx} # Create inverted dictionary (index --> class)
        top_classes = []
        for i in top_idx:
            top_classes.append(idx_to_class[i])

    return top_ps, top_classes

def print_probs_names(image, top_ps, top_classes, category_names):
	
	# Load mapping of category numbers to flower names 
	with open(category_names, 'r') as f:
		cat_to_name = json.load(f)

    # convert classes to flower names
	top_names = []
	for flower_class in top_classes:
	    top_names.append(cat_to_name[flower_class])

	print("// ----------------------------------------------------------------- //")
	print("{} is most likely a {} ({:.3f}%)".format(image, top_names[0].title(), top_ps[0]*100))
	for i in range(len(top_names)):
		print("   {}) {}: {:.3f}%".format(i+1, top_names[i].title(), top_ps[i]*100))
	print("// ----------------------------------------------------------------- //")

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

def main():
	args = args_paser() # get arguments (use default if user has not defined otherwise)
	device = set_device(args.gpu) # set device for computations

	model = load_checkpoint(args.checkpoint) # Load Checkpoint and create model
	top_probs, top_classes = predict(device, args.image, model, args.top_k) # get topK predictions for image

	print_probs_names(args.image, top_probs, top_classes, args.category_names)
	
if __name__ == "__main__":
	main()