def user_feedback(args, model):
    '''This function gives the user feedback on the parameters being used for the training of the model in the train.py file.
    It also provides information about the model starting to train'''
    device = 'cpu'
    if args.gpu:
        device = 'gpu'
        
    print('\n',
          'Using {} as model architecture\n'.format(args.set_architecture),
          'Model classifier using {} hidden units set to : {}\n'.format(args.hidden_units, model.classifier),
          'Learning rate used in optimizer set to : {}\n'.format(args.learning_rate),
          'Epochs set to {} will print every 32 optimization steps\n'.format(args.epochs),
          'Device used for computation: {}\n\n'. format(device),
          'Training starting...\n\n')
    
def prediction_feedback():
    '''This function provides the user with feedback on the steps taken in the predict.py file''' 
    print('\n',
          '\nImage processing complete\n',
          '\nModel loaded\n',
         '\nGetting results...\n')
    
