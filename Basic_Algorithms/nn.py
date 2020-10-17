################################################################################
#
# NOTES
#
#    Accuracy display
#
#       1 round, running time: 258.4379138946533 training loss: 0.0001136459476646814 testing accuracy: 0.9611
#       2 round, running time: 248.7334930896759 training loss: 2.248543398966341e-05 testing accuracy: 0.9706
#       3 round, running time: 235.25694727897644 training loss: 2.872156741900671e-07 testing accuracy: 0.9662
#       4 round, running time: 241.85218477249146 training loss: 5.743632410466831e-07 testing accuracy: 0.9743
#       5 round, running time: 244.63632202148438 training loss: 4.291681106601366e-07 testing accuracy: 0.9687
#       6 round, running time: 243.51196694374084 training loss: 9.024964705208541e-09 testing accuracy: 0.9784
#       7 round, running time: 240.7149531841278 training loss: 7.005849236530418e-10 testing accuracy: 0.974
#       8 round, running time: 244.91519904136658 training loss: 3.1270239779820856e-09 testing accuracy: 0.9745
#       9 round, running time: 250.63504004478455 training loss: 8.692366937656583e-08 testing accuracy: 0.9742
#       10 round, running time: 229.22509789466858 training loss: 1.6728329846452382e-09 testing accuracy: 0.9804
#
#       Each round will take about 4min.
#       Once I set epoch = 50, it showed that the highest accuracy will be around 0.984 when the round reached 40.
#       After 40 rounds, the accuracy won't change and the loss will almost be 0.
#
#    Performance display
#
#       total running time: 2437.919813156128
#       Please see the attachment figure
#
################################################################################

import os.path
import urllib.request
import gzip
import time
import numpy             as np
import matplotlib.pyplot as plt

################################################################################
#
# PARAMETERS
#
################################################################################

# data
DATA_NUM_TRAIN         = 60000
DATA_NUM_TEST          = 10000
DATA_ROWS              = 28
DATA_COLS              = 28
DATA_CLASSES           = 10
DATA_URL_TRAIN_DATA    = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
DATA_URL_TRAIN_LABELS  = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
DATA_URL_TEST_DATA     = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
DATA_URL_TEST_LABELS   = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
DATA_FILE_TRAIN_DATA   = 'train_data.gz'
DATA_FILE_TRAIN_LABELS = 'train_labels.gz'
DATA_FILE_TEST_DATA    = 'test_data.gz'
DATA_FILE_TEST_LABELS  = 'test_labels.gz'

# network
input_node_num = DATA_ROWS * DATA_COLS
hidden1_node_num = 1000
hidden2_node_num = 100
output_node_num = DATA_CLASSES

################################################################################
#
# DATA
#
################################################################################

# data download
if not os.path.exists(DATA_FILE_TRAIN_DATA):
    urllib.request.urlretrieve(DATA_URL_TRAIN_DATA, DATA_FILE_TRAIN_DATA)
if not os.path.exists(DATA_FILE_TRAIN_LABELS):
    urllib.request.urlretrieve(DATA_URL_TRAIN_LABELS, DATA_FILE_TRAIN_LABELS)
if not os.path.exists(DATA_FILE_TEST_DATA):
    urllib.request.urlretrieve(DATA_URL_TEST_DATA, DATA_FILE_TEST_DATA)
if not os.path.exists(DATA_FILE_TEST_LABELS):
    urllib.request.urlretrieve(DATA_URL_TEST_LABELS, DATA_FILE_TEST_LABELS)

# training data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_train_data   = gzip.open(DATA_FILE_TRAIN_DATA, 'r')
file_train_data.read(16)
buffer_train_data = file_train_data.read(DATA_NUM_TRAIN*DATA_ROWS*DATA_COLS)
train_data        = np.frombuffer(buffer_train_data, dtype=np.uint8).astype(np.float32)
train_data        = train_data.reshape(DATA_NUM_TRAIN, 1, DATA_ROWS, DATA_COLS)

# training labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_train_labels   = gzip.open(DATA_FILE_TRAIN_LABELS, 'r')
file_train_labels.read(8)
buffer_train_labels = file_train_labels.read(DATA_NUM_TRAIN)
train_labels        = np.frombuffer(buffer_train_labels, dtype=np.uint8).astype(np.int32)

# testing data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_test_data   = gzip.open(DATA_FILE_TEST_DATA, 'r')
file_test_data.read(16)
buffer_test_data = file_test_data.read(DATA_NUM_TEST*DATA_ROWS*DATA_COLS)
test_data        = np.frombuffer(buffer_test_data, dtype=np.uint8).astype(np.float32)
test_data        = test_data.reshape(DATA_NUM_TEST, 1, DATA_ROWS, DATA_COLS)

# testing labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_test_labels   = gzip.open(DATA_FILE_TEST_LABELS, 'r')
file_test_labels.read(8)
buffer_test_labels = file_test_labels.read(DATA_NUM_TEST)
test_labels        = np.frombuffer(buffer_test_labels, dtype=np.uint8).astype(np.int32)

################################################################################
#
# MY CODE GOES HERE
#
################################################################################


# 4 layers nn
# 784 input nodes -> 1000 1st hidden layer nodes -> 100 2nd hidden layer nodes -> 10 output layer nodes
class NeuralNetwork:
    def __init__(self, inputs_num, hidden1_num, hidden2_num, output_num, lr):
        self.input_node_num = inputs_num
        self.hidden1_node_num = hidden1_num
        self.hidden2_node_num = hidden2_num
        self.output_node_num = output_num
        self.learning_rate = lr

        # randomly initialize W and b
        self.input_hidden1_weights = 0.1 * np.random.normal(0, 1, (self.hidden1_node_num, self.input_node_num))
        self.input_hidden1_biases = 0.1 * np.random.normal(0, 1, (self.hidden1_node_num, 1))
        self.hidden1_hidden2_weights = 0.1 * np.random.normal(0, 1, (self.hidden2_node_num, self.hidden1_node_num))
        self.hidden1_hidden2_biases = 0.1 * np.random.normal(0, 1, (self.hidden2_node_num, 1))
        self.hidden2_output_weights = 0.1 * np.random.normal(0, 1, (self.output_node_num, self.hidden2_node_num))
        self.hidden2_output_biases = 0.1 * np.random.normal(0, 1, (self.output_node_num, 1))

        self.cross_entropy = 0
        self.activation_function = lambda x: relu(x)

    def train(self, X, Y):
        # forward
        # z = Wx + b   a = Relu(z)
        # X -> z0 -> a0 -> z1 -> a1 -> z2 -> a2 -> ce
        z0 = self.input_hidden1_weights.dot(X) + self.input_hidden1_biases
        a0 = self.activation_function(z0)
        z1 = self.hidden1_hidden2_weights.dot(a0) + self.hidden1_hidden2_biases
        a1 = self.activation_function(z1)
        z2 = self.hidden2_output_weights.dot(a1) + self.hidden2_output_biases
        a2 = softmax(z2)

        # loss
        self.cross_entropy = cross_entropy_loss(a2, Y)

        # back propagation
        # gradient_ce_z2 means the gradient between ce to z2
        gradient_ce_z2 = a2 - Y
        gradient_z2_w2 = gradient_ce_z2.dot(a1.T)
        hidden2_output_weights_copy = self.hidden2_output_weights.copy()
        self.hidden2_output_weights -= self.learning_rate * gradient_z2_w2
        self.hidden2_output_biases -= self.learning_rate * gradient_ce_z2

        gradient_z2_a1 = hidden2_output_weights_copy.T.dot(gradient_ce_z2)
        gradient_z2_a1[a1 == 0] = 0
        gradient_z1_w1 = gradient_z2_a1.dot(a0.T)
        hidden1_hidden2_weights_copy = self.hidden1_hidden2_weights.copy()
        self.hidden1_hidden2_weights -= self.learning_rate * gradient_z1_w1
        self.hidden1_hidden2_biases -= self.learning_rate * gradient_z2_a1

        gradient_z1_a0 = hidden1_hidden2_weights_copy.T.dot(gradient_z2_a1)
        gradient_z1_a0[a0 == 0] = 0
        gradient_z0_w0 = gradient_z1_a0.dot(X.T)
        self.input_hidden1_weights -= self.learning_rate * gradient_z0_w0
        self.input_hidden1_biases -= self.learning_rate * gradient_z1_a0

    def get_softmax_probability(self, X):
        z0 = self.input_hidden1_weights.dot(X) + self.input_hidden1_biases
        a0 = self.activation_function(z0)
        z1 = self.hidden1_hidden2_weights.dot(a0) + self.hidden1_hidden2_biases
        a1 = self.activation_function(z1)
        z2 = self.hidden2_output_weights.dot(a1) + self.hidden2_output_biases
        a2 = softmax(z2)
        return a2

    def get_cross_entropy_loss(self):
        return self.cross_entropy


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z_exp = np.exp(z - np.max(z))
    return z_exp / np.sum(z_exp)


def cross_entropy_loss(a, Y):
    return np.negative(np.sum(Y * np.log(a)))


# set learning rate, epoch
lr = 0.01
epoch = 10

# initialize nn
nn = NeuralNetwork(input_node_num, hidden1_node_num, hidden2_node_num, output_node_num, lr)

# collect loss and accuracy for each epoch
loss = []
accuracy = []
all_time_start = time.time()
for e in range(epoch):
    each_epoch_time_start = time.time()
    for i in range(DATA_NUM_TRAIN):
        if (i + 1) % 1000 == 0:
            print("trained", i + 1, "data, loss", nn.cross_entropy)
        img = train_data[i].reshape(input_node_num)
        img = img / 255.0
        img = np.array(img, ndmin=2).T
        target = np.zeros(output_node_num)
        target = np.array(target, ndmin=2).T
        target[train_labels[i]] = 1
        nn.train(img, target)

    s = 0
    for i in range(DATA_NUM_TEST):
        img = test_data[i].reshape(input_node_num)
        img = img / 255.0
        img = np.array(img, ndmin=2).T
        targets = np.zeros(output_node_num)
        targets = np.array(targets, ndmin=2).T
        target[test_labels[i]] = 1
        res = nn.get_softmax_probability(img)
        # print(res.T)
        # print(i + 1, "predicted label:", res.argmax(), "actual label:", test_labels[i])
        if res.argmax() == test_labels[i]:
            s += 1
    each_epoch_time_end = time.time()
    curLoss = nn.get_cross_entropy_loss()
    curAccuracy = s / DATA_NUM_TEST
    loss.append(curLoss)
    accuracy.append(curAccuracy)
    print(e + 1, "round, running time:", each_epoch_time_end - each_epoch_time_start, "training loss:", curLoss, "testing accuracy:", curAccuracy)
print("total running time:", time.time() - all_time_start)


################################################################################
#
# DISPLAY
#
################################################################################

fig, axs = plt.subplots(1, 2)
axs[0].set_title("loss on training data")
axs[0].set_xlabel('epoch')
axs[0].set_ylabel('loss')
axs[0].plot(loss)

axs[1].set_title("accuracy on test data")
axs[1].set_xlabel('epoch')
axs[1].set_ylabel('accuracy')
axs[1].plot(accuracy)
plt.show()
