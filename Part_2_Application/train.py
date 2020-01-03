

from train_functions import *
from helper_functions import *
import workspace_utils


import argparse

parser = argparse.ArgumentParser(description = 'Train your model', add_help = True)

parser.add_argument('data_dir', action = 'store', help = 'This is the directory where the data is stored. The data in this directory has to be organized in a training and validation set named "train" and "valid" respectively.', type = str)
parser.add_argument('--save_dir', action = 'store', dest = 'save_directory', default = '.', help = 'This is the directory where the checkpoint is saved', type = str)
parser.add_argument('--arch', action = 'store', dest = 'set_architecture', default = 'vgg16', help = 'This is the architecture used for the model. For example "vgg13"', type = str)
parser.add_argument('--learning_rate', action = 'store', dest = 'learning_rate', default = 0.001, help = 'Learning rate of the optimizer', type = float)
parser.add_argument('--hidden_units', action = 'store', dest = 'hidden_units', default = 1020, help = 'Number of hidden units of the two layer model', type = int)
parser.add_argument('--epochs', action = 'store', dest = 'epochs', default = 5, help = 'Number of epochs the model will train', type = int)
parser.add_argument('--gpu', action = 'store_true', dest = 'gpu', default = False, help = 'Device on which the model will run')


args = parser.parse_args()

#Load the data
train_dir, valid_dir = get_directory(args.data_dir)
trainloader, validloader = load_transformed_data(train_dir, valid_dir, batch_size = 64)

 
#Load the model
model = get_model(args.set_architecture) ## This function needs to be revised
#Create classifier and attach it to model
model = set_classifier(args.hidden_units, model)

#Set criterion and optimizer 
criterion, optimizer = set_config(args.learning_rate, model)

#Give the user feedback on the parameters being used for the training
user_feedback(args, model)

#Start training and print out training loss, validation loss and accuracy of the model.
with workspace_utils.active_session():
    train_model(trainloader, validloader, model, criterion, optimizer, args.gpu, args.epochs, print_every =32)
#Save checkpoint
save_checkpoint(args.save_directory, trainloader, model, args.epochs, optimizer, args)






