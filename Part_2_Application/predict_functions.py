
from train_functions import *
import torch
from torch import optim
from torchvision import datasets, transforms, models, utils
from PIL import Image
import numpy as np
import re
import json

def model_loader(file_name, gpu):
    '''This function takes in the filepath of the model's checkpoint and the device onto which the model should be loaded. 
    The function returns the loaded model with all saved attributes'''
    storage = lambda storage, loc: storage
    if gpu:
        storage = lambda storage, loc: storage.cuda()
     
    
    
    checkpoint = torch.load(file_name, map_location = storage)
    
    model = get_model(checkpoint['architecture'])
    #model = model.vgg13(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False
 

    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['mapping_class_index']
    epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_weights_bias'])
    criterion, optimizer = set_config(checkpoint['learning_rate'], model)
    optimizer.load_state_dict(checkpoint['optimizer_state'])                

    return model #check out the return
                                                 

def process_image(imagepath):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(imagepath)
    img.load()
    img_width, img_height = img.size
    if img_width > img_height:
        thumbnail_size = (img_width, 256)
    else:
        thumbnail_size = (256, img_height)
    
    resized_img = img.thumbnail(size = thumbnail_size)
    left, upper, right, lower = img_width//4-224//2, img_height//4-224//2, img_width//4+224//2, img_height//4+224//2
    cropped_img = img.crop((left, upper, right, lower))
    
    
    np_image = np.array(cropped_img)/255
                                                 
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm_np_image = (np_image-means)/std
    proccessed_image = norm_np_image.transpose(2,0,1)
                                              
               
    return proccessed_image
                                                
                                                                                                  
                                                 
def predict(image_path, model, gpu, topk):
    ''' This function takes in an image and a model and predicts the class (or classes) of the image using
    the provided deep learning model.
    '''
    device = set_device(gpu)

    #start by preprocessing the image
    input_image = process_image(image_path)
    #convert image to torch
    input_image = np.expand_dims(input_image, axis = 0)
    input_image = torch.from_numpy(input_image).type(torch.FloatTensor).to(device)

    model.eval()
    
    with torch.no_grad():
        model.to(device)

        output = model.forward(input_image)

        prediction = torch.exp(output)

        top_p, top_class = prediction.topk(topk,dim = 1)
        
        return np.array(top_p), (np.array(top_class))
                                                 
                                                 
def get_results(imagepath, cat_to_name, model, gpu, topk=5):
    '''This function takes in an image of which the category has to be predicted, a json to get the names from the classes, the model
    to make the prediction with, the device being used and the number of categories predicted. The function returns the predicted flower name and the
    probability of the prediction.'''
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    pattern = r'[\d]+'
    key = re.search(r'[\d]+', imagepath)
    

    actual_flower_name = cat_to_name[imagepath[key.span()[0]: key.span()[1]]]
    predictions, classes = predict(imagepath, model, gpu, topk)
    
    keys = {}
    for key, value in model.class_to_idx.items():
        if value in classes:
            keys[value] = key  
            
    name_dict = {}
    for item,pred in zip(classes[0],predictions[0]):
        name = cat_to_name[keys[item]]
        name_dict[name] = pred

    for name in name_dict:
        print('Predicted flower: {} | Probability of prediction: {:.2f}'.format(name, name_dict[name]*100))
    print('Actual flower label: {}'.format(actual_flower_name))
                                                 