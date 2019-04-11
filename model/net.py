"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    
    def __init__(self, input_channels=3, output_channels=2):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size = 3)
        self.batch_norm1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 16, kernel_size = 3)
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size = 2)

        self.conv3 = nn.Conv2d(16, 32, kernel_size = 3)
        self.batch_norm3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 32, kernel_size = 3)
        self.batch_norm4 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 32, kernel_size = 3)
        self.batch_norm5 = nn.BatchNorm2d(32)
        self.pool5 = nn.MaxPool2d(kernel_size = 2)

        self.conv6 = nn.Conv2d(32, 64, kernel_size = 3)
        self.batch_norm6 = nn.BatchNorm2d(64)

        self.conv7 = nn.Conv2d(64, 64, kernel_size = 3)
        self.batch_norm7 = nn.BatchNorm2d(64)
        self.pool7 = nn.MaxPool2d(kernel_size = 2)

        self.conv8 = nn.Conv2d(64, 128, kernel_size = 3)
        self.batch_norm8 = nn.BatchNorm2d(128)

        self.conv9 = nn.Conv2d(128, 128, kernel_size = 3)
        self.batch_norm9 = nn.BatchNorm2d(128)
        self.pool9 = nn.MaxPool2d(kernel_size = 2)

        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size = 2, stride = 2)

        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size = 3, stride = 1)
        self.dbatch_norm2 = nn.BatchNorm2d(128)

        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size = 3, stride = 1)
        self.dbatch_norm3 = nn.BatchNorm2d(64)

        self.deconv4 = nn.ConvTranspose2d(64, 64, kernel_size = 2, stride = 2)

        self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size = 3, stride = 1)
        self.dbatch_norm5 = nn.BatchNorm2d(64)

        self.deconv6 = nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 1)
        self.dbatch_norm6 = nn.BatchNorm2d(32)

        self.deconv7 = nn.ConvTranspose2d(32, 32, kernel_size = 2, stride = 2)

        self.deconv8 = nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 1)
        self.dbatch_norm8 = nn.BatchNorm2d(32)

        self.deconv9 = nn.ConvTranspose2d(32, 16, kernel_size = 3, stride = 1)
        self.dbatch_norm9 = nn.BatchNorm2d(16)

        self.deconv10 = nn.ConvTranspose2d(16, 16, kernel_size = 3, stride = 1)
        self.dbatch_norm10 = nn.BatchNorm2d(16)

        self.deconv11 = nn.ConvTranspose2d(16, 16, kernel_size = 2, stride = 2)

        self.deconv12 = nn.ConvTranspose2d(32, 16, kernel_size = 3, stride = 1)
        self.dbatch_norm12 = nn.BatchNorm2d(16)

        self.deconv13 = nn.ConvTranspose2d(16, 8, kernel_size = 3, stride = 1)
        self.dbatch_norm13 = nn.BatchNorm2d(8)

        self.deconv14 = nn.ConvTranspose2d(8, output_channels, kernel_size = 3, stride = 1, padding = 1)
    
    def forward(self, x):
        # Convolution:
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x2 = F.relu(x)
        x = self.pool2(x2)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.batch_norm5(x)
        x5 = F.relu(x)
        x = self.pool5(x5)

        x = self.conv6(x)
        x = self.batch_norm6(x)
        x = F.relu(x)

        x = self.conv7(x)
        x = self.batch_norm7(x)
        x7 = F.relu(x)
        x = self.pool7(x7)

        x = self.conv8(x)
        x = self.batch_norm8(x)
        x = F.relu(x)

        x = self.conv9(x)
        x = self.batch_norm9(x)
        x9 = F.relu(x)
        x = self.pool9(x9)

        # Deconvolution:
        x = self.deconv1(x)          
        x = F.relu(x)
        merged = torch.cat((x, x9), 1)

        x = self.deconv2(merged)
        x = self.dbatch_norm2(x)
        x = F.relu(x)

        x = self.deconv3(x)          
        x = self.dbatch_norm3(x)
        x = F.relu(x)

        x = self.deconv4(x)          
        x = F.relu(x)
        merged = torch.cat((x, x7), 1)

        x = self.deconv5(merged)
        x = self.dbatch_norm5(x)
        x = F.relu(x)

        x = self.deconv6(x)          
        x = self.dbatch_norm6(x)
        x = F.relu(x)

        x = self.deconv7(x)          
        x = F.relu(x)
        merged = torch.cat((x, x5), 1)

        x = self.deconv8(merged)
        x = self.dbatch_norm8(x)
        x = F.relu(x)

        x = self.deconv9(x)          
        x = self.dbatch_norm9(x)
        x = F.relu(x)

        x = self.deconv10(x)          
        x = self.dbatch_norm10(x)
        x = F.relu(x)

        x = self.deconv11(x)          
        x = F.relu(x)
        merged = torch.cat((x, x2), 1)

        x = self.deconv12(merged)
        x = self.dbatch_norm12(x)
        x = F.relu(x)

        x = self.deconv13(x)          
        x = self.dbatch_norm13(x)
        x = F.relu(x)

        x = self.deconv14(x)          

        return x


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    num_examples = outputs.size()[0]
    return -torch.sum(outputs[range(num_examples), labels])/num_examples


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
