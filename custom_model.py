import torch
import torchvision
from torchvision import models

def initialize_model(num_classes, use_pretrained=True, weights_file=None):
    """ DeepLabV3 pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    """
    if weights_file == None:
        model_deeplabv3 = models.segmentation.deeplabv3_resnet101(pretrained=use_pretrained, 
                                                                    progress=True)
    else:
        model_deeplabv3 = models.segmentation.deeplabv3_resnet101(pretrained=False, 
                                                                    progress=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(weights_file, map_location = device)
        model_deeplabv3 = model_deeplabv3.to(device)
        model_deeplabv3.load_state_dict(state_dict, strict=False)
    
    model_deeplabv3.aux_classifier = None

    for param in model_deeplabv3.parameters():
        param.requires_grad = False

    model_deeplabv3.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)
    
    return model_deeplabv3

