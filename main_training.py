import torch
import torch.nn as nn
import torch.optim as optim
import os
import pathlib

# Local import
from dataloader import DataLoaderSegmentation
from custom_model import initialize_model
from train import train_model

# print("PyTorch Version: ",torch.__version__)
# print("Torchvision Version: ",torchvision.__version__)

"""
    Version requirements:
        PyTorch Version:  1.4.0
        Torchvision Version:  0.5.0
"""


def main(data_dir, dest_dir, num_classes, batch_size, num_epochs, weights=None):
# def main():

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: DataLoaderSegmentation(os.path.join(data_dir, x), num_classes) for x in ['train', 
                                                                                                    'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, 
                        num_workers=1) for x in ['train', 'val']}

    print("Initializing Model...")

    # Initialize model
    if weights == None:
        model_deeplabv3 = initialize_model(num_classes, use_pretrained=True)
    else:
        model_deeplabv3 = initialize_model(num_classes, use_pretrained=True, weights_file=weights)

    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model_deeplabv3 = model_deeplabv3.to(device)
    model_deeplabv3= nn.DataParallel(model_deeplabv3)
    model_deeplabv3.to(device)
    
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_deeplabv3.parameters()
    print("Params to learn:")
    params_to_update = []
    for name, param in model_deeplabv3.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss function
    #criterion = nn.CrossEntropyLoss(weight=(torch.FloatTensor(weight).to(device) if weight else None))
    criterion = nn.CrossEntropyLoss()

    # Prepare output directory
    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)

    print("Train...")

    # Train and evaluate
    model_deeplabv3_state_dict, model_deeplabv3_trainned, history = train_model(model_deeplabv3, 
                                                                                num_classes, 
                                                                                dataloaders_dict, 
                                                                                criterion, 
                                                                                optimizer_ft, 
                                                                                device, dest_dir, 
                                                                                num_epochs=num_epochs)

    print("Save ...")
    torch.save(model_deeplabv3_state_dict, os.path.join(dest_dir, "best_DeepLabV3_floodDetection.pth"))

    return model_deeplabv3_state_dict, model_deeplabv3_trainned, history