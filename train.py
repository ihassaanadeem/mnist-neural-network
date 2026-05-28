import numpy as np
from data import load_data
from network import NeuralNetwork

input_size = 784
hidden_size = 128
output_size = 10
batch_size = 50
epochs = 10
learning_rate = 0.1

def train_data_prep():
    train_labels, train_images, _ , _ = load_data()
    
    # data shuffling
    indices = np.random.permutation(len(train_labels))
    batches = len(train_labels) // batch_size
    images = train_images[indices]
    labels = train_labels[indices].astype(int) 
    
    return images, labels, batches

def test_data_prep():
    _, _, labels, images = load_data()
    
    images = images
    labels = labels.astype(int)
    
    return images, labels
    
def test(nn, images, labels):
    # run forward pass on all test data
    nn.forward_prop(images)
    accuracy = nn.calc_accuracy(labels)
    loss = nn.loss_func(labels)
    
    return accuracy, loss

def train():
    nn = NeuralNetwork(input_size, hidden_size, output_size, batch_size)
    
    for epoch in range(epochs):
        images, labels, batches = train_data_prep()
        loss = 0
        
        for i in range(batches):
            start = i * batch_size
            end = start + batch_size
            batch_images = images[start:end]
            batch_labels = labels[start:end]
            
            nn.forward_prop(batch_images)
            loss += nn.loss_func(batch_labels)
            nn.back_prop(batch_images, batch_labels)
            nn.update_params(learning_rate)
        
        accuracy = nn.calc_accuracy(batch_labels)
        avg_loss = loss / batches
        
        images, labels = test_data_prep()
        test_acc, test_loss = test(nn, images, labels)
        
        print('epoch number: ', epoch, '\n',
              'average loss: ', avg_loss, '\n',
              'accuracy: ', accuracy, '\n',
              'test accuracy: ', test_acc, '\n',
              'test loss: ', test_loss, '\n')

train()