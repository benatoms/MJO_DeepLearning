import os
import sys
import math
import shutil
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from math import ceil

import MJO_CNN as mjonet
from MJOData import MJODataset

#Save a checkpoint for the model
def save_model(state, output_dir, rank, epoch):
    """
    Save the current state of the model so that it can be re-loaded
        if necessary

    Parameters
    ----------

    state : dictionary
    Contains current state of model
        {epoch number,
        model state dictionary
        optimizer state dictionary}

    output_dir : string
    The directory the output is stored within

    rank: int
    Rank of the MPI process

    epoch: int
    Current epoch number

    """
    filename='checkpoint_epoch' + str(epoch) + '.pth.tar'
    if rank == 0:
        torch.save(state, output_dir + '/' + filename)
        shutil.copyfile(output_dir + '/' + filename, 
            output_dir + '/' + 'model_epoch' + str(epoch) + '.pth.tar')

def init_print(rank, size, debug_print=True):
    """
    initialize printing from nodes on the NERSC CORI supercomputer

    Parameters
    ----------

    rank : int
    The rank of the node

    size : int
    Number of MPI processes

    debug_print : boolean
    Flag for debugging
        -if True, prints output from all nodes
        -if False, only prints from the master node

    """

    # in case run on lots of nodes, mute all the nodes except master 
    if not debug_print:
        if rank > 0:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
    else:
        # labelled print with info of [rank/size]
        old_out = sys.stdout
        class LabeledStdout:
            def __init__(self, rank, size):
                self._r = rank
                self._s = size
                self.flush = sys.stdout.flush

            def write(self, x):
                if x == '\n':
                    old_out.write(x)
                else:
                    old_out.write('[%d/%d] %s' % (self._r, self._s, x))

        sys.stdout = LabeledStdout(rank, size)

def train(epoch, model, training_data, optimizer, normStd, num_batches, rank, trainF):
    """
    Train the CNN

    Parameters
    ----------

    epoch : int
    The current epoch of training

    model : dict
    Object containing information about the model (weights, structure, etc.)

    training_data: PyTorch tensor
    Contains the training data for each sample, split into batches

    optimizer: dict
    Optimizer class instance that contains optimizer information (gradients, etc.)

    normStd: ndarray, shape(len(use_Chan))
    Standard deviation of zero-mean variables for normalization

    num_batches: int
    Total number of batches in the epoch

    rank: int
    The rank of the node

    trainF: file object
    File that the output for each batch will be written to for logging purposes

    """

    #Convert the model to training mode
    model.train()
    #Tracker for the epoch loss
    epoch_loss = 0.0
    #Tracker for the number of processed images
    nProcessed = 0.0
    #Define loss function
    loss_func = nn.CrossEntropyLoss()

    #Loop through each batch contained within the training_data
    for batch_idx, (data, target) in enumerate(training_data):
        #Standardize the data; assuming data is only zero-mean
        data = torch.from_numpy(data.numpy()/normStd[np.newaxis,:,np.newaxis,np.newaxis])
        #Zero the gradients for the current batch
        optimizer.zero_grad()
        #Pass the batch through the CNN
        output = model(data)
        #Calculate the loss
        loss = loss_func(output, target)
        #Collect the cumulative batch loss
        epoch_loss += loss.data[0]*len(target) #Collect the cumulative batch loss
        #Back propagate
        loss.backward()
        #Update the optimizer
        optimizer.step()
        #Iterate the number of processed batches
        nProcessed += len(data)

        ##Now calculate general statistics for logging purposes##

        #Apply a softmax to find the index of maximum probability
        probs = F.softmax(output)
        #Get the index of maximum probability
        pred = probs.data.max(1)[1]
        #Calculate the number of incorrect classifications
        incorrect = pred.ne(target.data).sum()
        incorrect = incorrect.double()
        #Calculate the percentage classification error
        err = 100.*incorrect/len(target)
        #Calculate the stage in the epoch
        partialEpoch = epoch + batch_idx / num_batches

        #Output and/or write out relevant information such as the epoch status, loss, and error for logging purposes
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, loss.data[0], int(err)))

        #Write out the following information for this batch:
            #1) partial epoch
            #2) Loss
            #3) Fractional error
            #4) Predicted class
            #5) Correct (target) class
            #6) Probabilities for each class for each sample
        trainF.write('{},{},{},{},{},{},|\n'.format(partialEpoch, loss.data[0], int(err), pred.numpy(), target.numpy(), probs.detach().numpy()))
        trainF.flush()

    #Calculate the total epoch loss
    epoch_loss /= len(training_data.dataset) 

    #Return the epoch loss for logging purposes
    return epoch_loss

def validate(epoch, model, validation_data, optimizer, normStd, num_batches, valF):
    """
    Validate the CNN

    Parameters
    ----------

    epoch : int
    The current epoch of training

    model : dict
    Object containing information about the model (weights, structure, etc.)

    validation_data: PyTorch tensor
    Contains the validation data for each sample, split into batches

    optimizer: dict
    Optimizer class instance that contains optimizer information (gradients, etc.)

    normStd: ndarray, shape(len(use_Chan))
    Standard deviation of zero-mean variables for normalization

    num_batches: int
    Total number of batches in the epoch

    valF: file object
    File that the output for each batch will be written to for logging purposes

    """


    #convert the model to evaluation mode
    model.eval()
    #Initialize the validation loss
    val_loss = 0
    #Initialize the number of incorrect labels
    incorrect_tot = 0
    #Define loss function
    loss_func = nn.CrossEntropyLoss()

    #Loop through each batch contained within the validation_data
    for batch_idx, (data, target) in enumerate(validation_data):

        #Standardize the data; data is already zero-mean, just need to divide by std. dev.
        data = torch.from_numpy(data.numpy() / normStd[np.newaxis,:,np.newaxis,np.newaxis])
        #Pass the batch through the CNN
        output = model(data)
        #Calculate the loss
        loss = loss_func(output, target) #Calculate the loss
        #Collect the cumulative batch loss
        val_loss += loss.data[0] * len(target) #Add to validation loss values


        ##Now calculate general statistics for logging purposes##

        #Apply a softmax to find the index of maximum probability
        probs = F.softmax(output)
        #Get the index of maximum probability
        pred = probs.data.max(1)[1]
        #Calculate the number of incorrect classifications
        incorrect = pred.ne(target.data).sum()
        incorrect = incorrect.double()
        #Calculate the percentage classification error
        err = 100. * incorrect / len(target)
        #Calculate the stage in the epoch
        partialEpoch = epoch + batch_idx / num_batches
        #Update the total incorrect classifications in the validation set
        incorrect_tot += incorrect


        #Write out the following information for this batch:
            #1) partial epoch
            #2) Loss
            #3) Fractional error
            #4) Predicted class
            #5) Correct (target) class
            #6) Probabilities for each class for each sample
        valF.write('{},{},{},{},{},{},|\n'.format(partialEpoch, loss, int(err), pred.numpy(), target.numpy(), probs.detach().numpy()))
        valF.flush()

    #Calculate the fraction of incorrectly predicted instances for the entire validation set
    #Calculate the set loss
    val_loss /= len(validation_data.dataset)
    #Calculate the set fractional error
    nTotal = len(validation_data.dataset)
    err = 100.*incorrect_tot/nTotal
    accuracy = 1 - err/100 #accuracy in fractional form

    #Print out set statistics for logging purposes
    print('\nValidation set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        val_loss, incorrect_tot, nTotal, err))

def load_model(output_dir, last_epoch, model, optimizer):
    """
    Load a pre-trained model

    Parameters
    ----------

    output_dir : string
    The directory the model was output into

    last_epoch : int
    The number of epoch associated with the loaded model

    model: dict
    Raw model architecture, to be updated with saved weights

    optimizer: dict
    Optimizer class instance that contains optimizer information (gradients, etc.)

    """

    #Create string for the checkpoint file location
    checkpoint_file = output_dir + '/' + 'checkpoint_epoch' + str(last_epoch) +'.pth.tar'

    #If the checkpoint file exists, load it into the current model dictionary
    if os.path.isfile(checkpoint_file):
        print("=> loading checkpoint '{}'".format(checkpoint_file))
        #Load the checkpoint
        checkpoint = torch.load(checkpoint_file)
        #Reset the starting epoch to that of the saved model
        start_epoch = checkpoint['epoch']
        #Load the state dictionary of the CNN weights
        model.load_state_dict(checkpoint['state_dict'])
        #Load the state dictionary of the CNN optimizer
        optimizer.load_state_dict(checkpoint['optimizer'])
        #Print that the model has been loaded
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_file, checkpoint['epoch']))

        return start_epoch

    #If the checkpoint file does not exist, return error
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


def main(rank, size):

    """
    Function that drives the training of the CNN

    Parameters
    ----------

    rank: int
    The rank of the node

    size: int
    The number of MPI processes requested

    """

    #Initiate the argument parser
    parser = argparse.ArgumentParser()

    #Initialize arguments that we will need for network training,
    #   but these can also be initialized in the call of my_trainer_MJO.py
    parser.add_argument('--batchSz', type=int, default=64)
    parser.add_argument('--nEpochs', type=int, default=500)
    parser.add_argument('--save_dir')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--useChan', type=int, nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('--exper_dir', type=str, help='<Required> Set flag', required=True)
    parser.add_argument('--resume_bool', type=bool, default=False)
    parser.add_argument('--resume_epoch', type=int)
    parser.add_argument('--start_epoch', default=1, type=int,
                        help='manual epoch number (useful on restarts)')
    args = parser.parse_args()


    #Create the directory that the output will be saved in...
    #If the directory already exists, this is fine, and move on...
    output_dir = '/global/cscratch1/sd/benatoms/DeepWave_output/' + args.exper_dir
    os.makedirs(output_dir, exist_ok=True)

    #Testing if we can write to files using multi-node
    trainF = open(os.path.join(output_dir, 'train' + str(rank) + '.txt'), 'w')
    valF = open(os.path.join(output_dir, 'validate' + str(rank) + '.txt'), 'w')

    #Random number generator seeded with the specified value
    torch.manual_seed(1234)

    #The channels have their means removed, but are not yet normalized by their std. dev.
    #We must therefore normalize each channel by its std. dev.
    normStd_all = np.array([11.84, #OLR
                1.95,2.88,4.96, #U850, U500, U200
                1.17,1.76,3.67, #V850, V500, V200
                0.93,0.73,0.73, #T850, T500, T200
                8.02E-4,4.80E-4,7.48E-6, #QV850, QV500, QV200
                9.13,12.5,23.5]) #H850, H500, H200

    #Now remove only the normMean and normStd values corresponding to the fields
    #   that are being used...
    normMean = normMean_all[args.useChan]
    normStd = normStd_all[args.useChan]

    #Create the loader object for the training data
    training_loader = MJODataset(events_file='/global/homes/b/benatoms/DeepWave/data/mjo_dates_phases_training_daily_1_.npy',
                                    root_dir='/global/cscratch1/sd/benatoms/npy_files/complete',
                                    channels=args.useChan)

    #Create the loader object for the validation data
    validation_loader = MJODataset(events_file='/global/homes/b/benatoms/DeepWave/data/mjo_dates_phases_validation_daily_1_.npy',
                                    root_dir='/global/cscratch1/sd/benatoms/npy_files/complete',
                                    channels=args.useChan)

    #Load the training data using the training data loader
    training_data = DataLoader(
        training_loader, batch_size=args.batchSz, shuffle=True)
    #Load the validation data using the validation data loader
    validation_data = DataLoader(
        validation_loader, batch_size=args.batchSz, shuffle=False)

    #Initialize the neural network
    model = mjonet.MJONet(num_channels=len(args.useChan), num_classes=9, dropout_rate=0.2)
    model = model.double()

    #Define the optimizer, in this case Adam
    optimizer = optim.Adam(model.parameters(), lr=1E-3)    

    #Load the checkpoint if resume_bool == True
    if args.resume_bool == True:
        args.start_epoch = load_model(output_dir, args.resume_epoch, model, optimizer)

    #Calculate the number of batches in the training dataset
    num_batches_train = ceil(len(training_data.dataset) / args.batchSz)
    #Calculate the number of batches in the validation dataset
    num_batches_val = ceil(len(validation_data.dataset) / args.batchSz)

    #Calculate the number of samples in the training dataset
    nTrain = len(training_data.dataset)

    for epoch in range(args.start_epoch, args.nEpochs):
        #Re-load the training data using the training data loader
        #This reshuffles the training data for each epoch
        training_data = DataLoader(
            training_loader, batch_size=args.batchSz, shuffle=True)

        #train the model
        epoch_loss = train(epoch, model, training_data, optimizer, normStd, num_batches_train, rank, trainF)

        #get information on the model state after the last epoch
        print('Epoch {} Loss {:.6f} Global batch size {} on {} ranks'.format(
            epoch, epoch_loss,))

        #test the accuracy of the model using the validation dataset
        validate(epoch, model, validation_data, optimizer, normStd, num_batches_val, valF)

        #save the most recent version of the model
        save_model({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),},
                    output_dir,
                    rank,
                    epoch)


    #Close the training and testing log files
    trainF.close()
    testF.close()



if __name__ == "__main__":
    #Initiate MPI process
    dist.init_process_group(backend='mpi')

    #Gather the size and rank of the MPI call
    size = dist.get_world_size()
    rank = dist.get_rank()

    #Initialize the printing for each node
    init_print(rank, size)

    #Initiate the main function
    main(rank, size)
