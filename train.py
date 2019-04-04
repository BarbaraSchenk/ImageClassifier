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
    paser = argparse.ArgumentParser(description='Parameters for Training')

    # Directories for loading data and saving trained model
    paser.add_argument('data_dir', type=str, default='flowers', help='directory of datasets ') # directory for train, test, and validation datasets
    paser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='directory for saving trained model')

    # Architecture and HyperParameters for training
    paser.add_argument('--arch', type=str, default='densenet121', help='choose architecture (resnet50 or densenet121)')
    paser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    paser.add_argument('--hidden_units', type=int, default=512, help='hidden units in hidden layer')
    paser.add_argument('--epochs', type=int, default=5, help='number of training epochs')

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

def load_transform_data(data_dir):
    '''
    Loads data from folders and transforms them as defined
    returns dataloaders for training, validating, and testing
    '''
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224),\
                            transforms.RandomHorizontalFlip(), transforms.ToTensor(),\
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    valid_test_transforms = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224),\
                            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir+'/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir+'/valid', transform=valid_test_transforms)
    test_data = datasets.ImageFolder(data_dir+'/test', transform=valid_test_transforms)

    # Define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return train_data, trainloader, validloader, testloader

def get_pretrained_model(arch):
    # Loads pretrained model (default is resnet50)
    if arch == "resnet50":
        model = models.resnet50(pretrained=True)
        print("ResNet-50 is used as pretrained network.")
    elif arch == "densenet121":
        model = models.densenet121(pretrained=True)
        print("DenseNet-121 is used as pretrained network.")
    else:
        arch = "densenet121"
        model = models.densenet121(pretrained=True)
        print("{} is not a valid Torchvision model name. DenseNet-121 is used as pretrained network instead.".format(arch))
    return model

def create_classifier(arch, model, hidden_units):
    # creates the classifier
    # get number of input units
    if arch == "resnet50":
        input_units = 2048
    elif arch == "densenet121":
        input_units = 1024
    
    if hidden_units > input_units:
        hidden_units = 512

    # create classifier with one hidden layer and use dropout to avoid overfitting
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_units, hidden_units)),
                                     ('relu1', nn.ReLU()),
                                     ('drop1', nn.Dropout(p=0.2)),
                                     ('fc2', nn.Linear(hidden_units, 102)),
                                     ('soft2', nn.LogSoftmax(dim=1))]))
    print("Classifier with {} input units and 1 hidden layer with {} units created.".format(input_units, hidden_units))
    return classifier

def setup_neuralnetwork(arch, model, classifier, learning_rate):
    # sets up neural network for training

    if arch == "resnet50":
        model.fc = classifier # set the defined classifier to be the new classifier for the model
        criterion = nn.NLLLoss() # set error function
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate) # define optimizer
    elif arch == "densenet121":
        model.classifier = classifier # set the defined classifier to be the new classifier for the model
        criterion = nn.NLLLoss() # set error function
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate) # define optimizer

    # Freeze Parameters, so they are not manipulted durcin backward pass
    for param in model.parameters():
        param.requieres_grad=False

    print("Neural Network set up for training with NLLLoss Error Function and Adam Optimizer.")
    return criterion, optimizer

def train_neural_network(device, model, criterion, optimizer, trainloader, validloader, epochs):
    '''
    Trains the setup neural network with the training dataset
    Calculates accuracy of predictions based on validation dataset
    Returns trained neural network
    '''

    # initialize
    steps = 0 
    train_loss = 0
    print_every = 31
    accuracy = 0

    model.to(device)
    
    with active_session():
        # Train model and calculate and print accuracy every 10 steps of process
        for epoch in range(epochs):
            # TRAIN
            for images, labels in trainloader:
                steps += 1
                # Move input and label tensors to the default device (GPU, if it is available)
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad() # reset gradients to 0 (they are added otherwise)
                output = model.forward(images) # run batch of images through model and get predictions
                loss = criterion(output, labels) # calculate error by comparing predictions to labels (true values)
                loss.backward() # perform backward pass
                optimizer.step() # adjust weights

                train_loss += loss.item()
                
                # VALIDATE
                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval() # switch to evaluation mode --> dropout is deactivated
                    with torch.no_grad(): # turn off tracking of gradients
                        for images, labels in validloader:
                            images, labels = images.to(device), labels.to(device) # move images/labels of validation set to GPU
                            
                            # validate model by passing validation images through model and 
                            # comparing model prediction with true label
                            output = model.forward(images)
                            batch_loss = criterion(output, labels)
                            valid_loss += batch_loss.item()
                            
                            # Calculate accuracy for statistics
                            ps = torch.exp(output)
                            top_p, top_class = ps.topk(1, dim=1) # get prediction with highest probability for each image
                            # create equals tensor with 1 (prediction=label) and 0 (else)
                            equals = top_class == labels.view(*top_class.shape)
                            # calculate accuracy by taking mean of 1's and 0's
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item() 
                          
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {train_loss/print_every:.3f}.. "
                          f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                          f"Validation accuracy: {accuracy/len(validloader):.3f}")
                    
                    train_loss = 0
                    model.train()

    return model, accuracy

def test_neural_network(device, model, criterion, testloader):
    # Calculates accuracy of predictions based on validation dataset

    test_loss=0
    test_accuracy=0

    model.eval() # switch to evaluation mode --> dropout is deactivated

    with torch.no_grad(): # turn off tracking of gradients
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device) # move images/labels of test set to GPU
            output = model.forward(images)
            batch_loss = criterion(output, labels)
            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item() 

    print(f"Accuracy (test dataset): {test_accuracy/len(testloader):.3f}")

def save_checkpoint(arch, model, train_data, save_dir):
    # Create and save checkpoint
    model.class_to_idx = train_data.class_to_idx
    if arch == "resnet50":
        checkpoint = {'arch': arch,
                  'classifier': model.fc,
                  'classifier_state_dict': model.fc.state_dict(),
                  'class_to_idx': model.class_to_idx
                  }
    elif arch == "densenet121":
        checkpoint = {'arch': arch,
                      'classifier': model.classifier,
                      'classifier_state_dict': model.classifier.state_dict(),
                      'class_to_idx': model.class_to_idx
                      }

    torch.save(checkpoint, save_dir)
    print("Checkpoint saved as: {}".format(save_dir))

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

def main():
    args = args_paser() # get arguments (use default if user has not defined otherwise)
    arch = args.arch.lower()

    device = set_device(args.gpu) # set device for computations
    train_data, trainloader, validloader, testloader = load_transform_data(args.data_dir) # create trainloader, validloader, testloader
    pretrained_model = get_pretrained_model(arch) # get pretrained model
    classifier = create_classifier(arch, pretrained_model, args.hidden_units) # creates classifier network
    criterion, optimizer = setup_neuralnetwork(arch, pretrained_model, classifier, args.learning_rate)
    
    print("------ START TRAINING ------")
    trained_model, accuracy = train_neural_network(device, pretrained_model, criterion, optimizer, trainloader, validloader, args.epochs)
    print("------ END TRAINING ------")
    print(f"Accuracy (validation dataset): {accuracy/len(validloader):.3f}")

    test_neural_network(device, trained_model, criterion, testloader)
    save_checkpoint(arch, trained_model, train_data, args.save_dir)

if __name__ == "__main__":
	main()
