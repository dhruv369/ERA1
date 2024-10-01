'''
Assignment S6 : 
99.4% validation accuracy
Less than 20k Parameters
You can use anything from above you want. 
Less than 20 Epochs
Have used BN, Dropout,
(Optional): a Fully connected layer, have used GAP. 
'''
import sys
import os

# Add the directory containing DL_Lib to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DL_Lib')))

from DataSet import MnistDataSet
from Model import Network3
from Utils import model_summary,get_device, NNPipeLine
import torch.optim as optim
import torch.nn.functional as F

'''
This is sample file to demo how to train a neural network using some of utils files
'''

# define some of your hyper params
numEpochs = 1
batchSize = 512
kwargs = {'batch_size': batchSize, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

if __name__ == '__main__':

    # get the device info using utils
    device = get_device()
    print("Device:", device)

    # prepare your data, for example here we have used the MNIST dataset and visualize it using some utils functions
    data = MnistDataSet(batchsize=batchSize)
    train_loader = data.get_train_loader(kwargs=kwargs)
    test_loader = data.get_test_loader(kwargs=kwargs)
    #display_image_grid(train_loader[0],train_loader[1])
    #visualize_Data(train_loader=train_loader)

    # Load model and get summary
    model = Network3().to(device)
    model_summary(model, input_size=(1, 28, 28))


    # This is how we can run NN Pipeline
    nnPipeline = NNPipeLine(model,device)
    # set hyper parameters
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)
    criterion = F.cross_entropy  #nn.CrossEntropyLoss()
    # test_incorrect_predication = {'images': [], 'ground_truths': [], 'predicted_vals': []}

    for epoch in range(1, numEpochs + 1):
        print(f'Epoch {epoch}')
        nnPipeline.train(train_loader=train_loader,optimizer= optimizer, criterion=criterion)
        nnPipeline.test(test_loader=test_loader,criterion= criterion)
        scheduler.step()
    nnPipeline.print_performance()
