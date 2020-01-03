import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models, utils
import torch.utils.data
from datetime import datetime


def get_model(model_name):
    '''This function takes in a model name as an argument,
    downloads it and returns the model.'''

    local = {}
    exec('model = models.'+ model_name +'(pretrained = True)', globals(), local)
    model = local['model']
    return model
    


def get_directory(path):
    '''This function takes as input the directory in which the data is stored and returns
    two directories, one for training and one for validation. WARNING: This function is used in combination
    with the load_transformed_data function and will only work if the data is organized in a training and validation 
    set named 'train' for the training set and 'valid' for the validation set'''
    train_dir = path + '/train'
    valid_dir = path + '/valid'
    return train_dir, valid_dir

def set_device(gpu):
    '''This function takes as input a boolean that signifies if there is a gpu available or not. 
    It the returns the adecuate torch.device'''
    device = torch.device('cpu')
    if gpu:
        device = torch.device('cuda')
    return device
    

def load_transformed_data(train_dir, valid_dir, batch_size = 64):
    ''' This function takes as input two directories, tranforms the data and returns
    a dataloader for the training data and a dataloader for the validation data.'''
    train_transforms = transforms.Compose([transforms.Resize(224),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomVerticalFlip(),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomRotation(30),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                           std = [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                          std = [0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True )
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size)

    return trainloader, validloader

def set_classifier(hidden_units, model):
    '''This function takes as input the number of nodes or hidden units of a layer and uses this input to create a new classifier for the model.
    It then attaches the new classifier to the model that has been provided as a second argument. The function returns the model with the new
    classifier'''
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(p = 0.3),
                               nn.Linear(hidden_units, 102),
                               nn.LogSoftmax(dim = 1))
    model.classifier = classifier
    return model
    
    

def set_config(learning_rate, model):
    '''This function is used to create an optimizer and provide a criterion for the training of the model.
    Both arguments, learning_rate and model, are used to set up the optimizer. The function returns both a criterion and 
    and optimizer'''
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    return criterion, optimizer

def validate_model(dataset, model, criterion, gpu):
    '''This function takes as inputs a validation dataset, a model and a criterion. These are used to return
    the validation loss and the validation accuracy of the model'''
    
    device = set_device(gpu)
    
    model.to(device)
        
    model.eval()
    loss = 0
    accuracy = 0

    with torch.no_grad():
        for images, labels in dataset:
            images, labels = images.to(device), labels.to(device)
            output_valid = model.forward(images)
            output_probability = torch.exp(output_valid)

            loss += criterion(output_valid, labels).item()

            top_p, top_class = output_probability.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    return loss, accuracy


def train_model(trainloader, testloader, model, criterion, optimizer, gpu, epochs, print_every= 10):
    '''This function takes as input two dataloaders, one for training and one for validation. Furthermore, it requires a model,
    a criterion, an optimizer, a boolean value for the availability of a gpu, the number of epochs and the frequency with which the results of the
    training and validation should be printed. The function then trains and validates the model and prints out the results. This function has no return
    value.'''

    device = set_device(gpu)
    print(device)
    
    model.to(device)
    
    steps = 0
    running_loss = 0
    for e in range(epochs):
        
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            steps += 1

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss, accuracy = validate_model(testloader, model, criterion, gpu)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(validation_loss/len(testloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(testloader)))

                running_loss = 0

                # Make sure dropout and grads are on for training
                model.train()
    print('Training complete')
    
def save_checkpoint(directory, trainloader, model, epochs, optimizer, args):
    '''This function takes as input the directory in which the model should be saved, a trainloader to get the class indeces, the model to be saved,
    the epochs used to train the model, the optimizer, and the argsparse arguments. The function saves all parameters to a file that is then saved in
    the provided directory with the name *yyyymmddhhmm_checkpoint.pth*, hh and mm are formatted in UTC time zone.'''
    model.class_to_idx = trainloader.dataset.class_to_idx 
    checkpoint = {'classifier': model.classifier,
                 'architecture': args.set_architecture,
                 'mapping_class_index': model.class_to_idx,
                 'epochs': epochs,
                 'learning_rate': args.learning_rate,
                 'optimizer_state': optimizer.state_dict(),
                 'state_weights_bias': model.state_dict()}
    now = datetime.now()
    now = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute)

    if directory[-1] == '/':
        save_file = directory + now + '_checkpoint.pth'
    else:
        save_file = directory + '/' + now + '_checkpoint.pth'
    torch.save(checkpoint,save_file)
    print('Checkpoint saved as {}'.format(now + '_checkpoint.pth'))
