import numpy as np

class NeuralNetwork():
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        
        self.a1 = None
        self.a2 = None
        
        self.grad_d2 = None
        self.grad_w2 = None
        self.grad_b2 = None
        self.grad_a1 = None
        self.grad_d1 = None
        self.grad_w1 = None
        self.grad_b1 = None
        
        # kaiming initialization compensates for 50% variance drop due to ReLU activation
        kaiming_w1 = np.sqrt(2 / self.input_size)
        self.w1 = np.random.randn(self.input_size, self.hidden_size) * kaiming_w1
        self.b1 = np.zeros((1, self.hidden_size))
        
        kaiming_w2 = np.sqrt(2 / self.hidden_size)
        self.w2 = np.random.randn(self.hidden_size, self.output_size) * kaiming_w2
        self.b2 = np.zeros((1, self.output_size))
    
    def forward_prop(self, data):
        d1 = np.dot(data, self.w1) + self.b1
        self.a1 = np.maximum(0, d1) #ReLU activation
        
        # logits
        d2 = np.dot(self.a1, self.w2) + self.b2
        
        # softmax activation function
        self.a2 = np.exp(d2)
        self.a2 = self.a2 / np.sum(self.a2, axis=1, keepdims=True)
    
    def loss_func(self, labels):
        # cross-entropy loss calc
        probs = self.a2[np.arange(len(labels)), labels]
        loss = -np.log(probs + 1e-15)
        
        # mean loss over batch
        avg_loss = np.mean(loss)
        
        return avg_loss
    
    def back_prop(self, data, labels):
        # chain rule to calculate derivatives starting from predicted vals
        self.grad_d2 = self.a2.copy()
        self.grad_d2[np.arange(len(labels)), labels] -= 1
        
        self.grad_w2 = np.dot(self.a1.T, self.grad_d2) / len(labels)
        self.grad_b2 = np.sum(self.grad_d2, axis=0, keepdims=True) / len(labels)
        
        self.grad_a1 = np.dot(self.grad_d2, self.w2.T)
        
        self.grad_d1 = self.grad_a1.copy()
        self.grad_d1[self.a1 <= 0] = 0
        
        self.grad_w1 = np.dot(data.T, self.grad_d1) / len(labels)
        self.grad_b1 = np.sum(self.grad_d1, axis=0, keepdims=True) / len(labels)
        
    def update_params(self, learning_rate):
        # update rule: 
        # x = x - learning_rate * grad_x        
        self.w1 -= learning_rate * self.grad_w1
        self.b1 -= learning_rate * self.grad_b1
        self.w2 -= learning_rate * self.grad_w2
        self.b2 -= learning_rate * self.grad_b2

    def calc_accuracy(self, labels):
        predictions = np.argmax(self.a2, axis=1)
        accuracy = np.mean(predictions == labels)
        
        return accuracy