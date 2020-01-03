# Load the image
from predict_functions import *
from helper_functions import prediction_feedback

import argparse

parser = argparse.ArgumentParser(description = 'Predict category', add_help = True)

parser.add_argument('file_path', action = 'store', help = 'This is the directory of the image of which the category has to be predicted.', type = str)
parser.add_argument('checkpoint_name', action = 'store', help = 'This is the name of the checkpoint', type = str)
parser.add_argument('--top_k', action = 'store', dest = 'top_category', default = 5, help = 'Number of categories that will be predicted', type = int)
parser.add_argument('--category_names', action = 'store', dest = 'category_names', help = 'Name of file with category names', type = str)
parser.add_argument('--gpu', action = 'store_true', dest = 'gpu', default = False, help = 'Device on which the model will run')


args = parser.parse_args()


# Process the image
process_image(args.file_path)
#Load saved model checkpoint
model = model_loader(args.checkpoint_name, args.gpu)

#Prediction

prediction_feedback()


get_results(args.file_path, args.category_names, model, args.gpu, args.top_category)