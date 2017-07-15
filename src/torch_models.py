import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pyjet.models import SLModel
from pyjet.losses import categorical_crossentropy
import pyjet.backend as J

def chess_loss(y_pred, target, size_average=True):
    """
    B x N_i
    B
    """
    assert(len(y_pred) == target.size(0))
    loss_sum = sum(categorical_crossentropy(y_pred[i].view(1, -1), target[i:i+1]) for i in range(len(y_pred)))
    return loss_sum / len(y_pred) if size_average else loss_sum

def chess_loss2(y_pred, target, size_average=True):
    """
    B x N_i
    B
    """
    assert(len(y_pred) == target.size(0))
    loss_sum = sum(categorical_crossentropy(y_pred[i][0].view(1, -1), target[i:i+1]) for i in range(len(y_pred)))
    # loss_sum += sum(categorical_crossentropy(y_pred[i][1].view(1, -1), target[i:i+1]) for i in range(len(y_pred)))
    return loss_sum / len(y_pred) if size_average else loss_sum

def accuracy(y_pred, target, k=1):
    """Computes the precision@k for the specified values of k"""
    maxk = k
    batch_size = target.size(0)
    # y_pred = [sample.data for sample in y_pred]
    # target = target.data

    _, pred = zip(*[output_vec[0].topk(maxk, 0, True, True) for output_vec in y_pred])
    pred = torch.cat(pred).view(-1, 1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:k].view(-1).float().sum(0)
    res = (correct_k.mul_(100.0 / batch_size))
    return res

def avg_prob(y_pred, target):
    return sum(y_pred[i][0][target[i].data] for i in range(len(y_pred))) / len(y_pred)

def avg_val(y_pred, target):
    return sum(y_pred[i][1][target[i].data] for i in range(len(y_pred))) / len(y_pred)

def avg_min_prob(y_pred, target):
    return sum(y_pred[i][0].min() for i in range(len(y_pred))) / len(y_pred)

def avg_min_val(y_pred, target):
    return sum(y_pred[i][1].min() for i in range(len(y_pred))) / len(y_pred)

class ChessModel(SLModel):
    def __init__(self):
        super(ChessModel, self).__init__()

    def forward(self, x_boards):
        raise NotImplementedError

    def cast_input_to_torch(self, x, volatile=False):
        return [Variable(J.Tensor(x_sample), volatile=volatile) for x_sample in x]

    def cast_output_to_numpy(self, preds):
        return [pred.data.cpu().numpy() for pred in preds]

class AlphaChess(ChessModel):
    name = "alpha_chess2"
    def __init__(self, embedding_size=128, num_filters=128):
        super(AlphaChess, self).__init__()
        # Block 0 - Square View
        self.conv1 = nn.Conv2d(17, num_filters, 1, padding=0)
        self.bn1 = nn.BatchNorm2d(num_filters)
        # Block 1 - Global (almost)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 7, padding=3)
        self.bn2 = nn.BatchNorm2d(num_filters)
        # Block 2 - Large View
        self.conv3 = nn.Conv2d(num_filters, num_filters, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(num_filters)
        self.conv4 = nn.Conv2d(num_filters, num_filters, 5, padding=2)
        self.bn4 = nn.BatchNorm2d(num_filters)
        self.conv5 = nn.Conv2d(num_filters, num_filters, 5, padding=2)
        self.bn5 = nn.BatchNorm2d(num_filters)
        # Block 3 - Local View
        self.conv6 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(num_filters)
        self.conv7 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(num_filters)
        self.conv8 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(num_filters)
        self.conv9 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(num_filters)
        self.conv10 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(num_filters)
        # Dense layer
        self.fc1 = nn.Linear(num_filters * 8 * 8, embedding_size)
        # Evaluation Layer
        self.fc2 = nn.Linear(embedding_size, 1)
        # self.logsoftmax = nn.LogSoftmax()
        # self.softmax = J.Softmax()

    def forward(self, x_boards):
        out = []
        # Do the casting to torch here
        for x in x_boards:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)))
            x = F.relu(self.bn6(self.conv6(x)))
            x = F.relu(self.bn7(self.conv7(x)))
            x = F.relu(self.bn8(self.conv8(x)))
            x = F.relu(self.bn9(self.conv9(x)))
            x = F.relu(self.bn10(self.conv10(x)))
            x = J.flatten(x)
            x = F.relu(self.fc1(x))
            x = J.zero_center(self.fc2(x)).view(1, -1) # 1 x N
            preds = J.softmax(x).view(-1), F.tanh(x).view(-1) # N
            out.append(preds)
        return out # B x N_i

class SimpleModel(ChessModel):
    name = "simple_model"
    def __init__(self, hidden_layer=2048):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(13 * 8 * 8 + 4, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer)
        self.fc3 = nn.Linear(hidden_layer, 1)

    def forward(self, x_boards):
        out = []
        for x in x_boards:
            x = J.flatten(x)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = J.zero_center(self.fc3(x)).view(1, -1) # 1 x N
            preds = J.softmax(x).view(-1), F.tanh(x).view(-1) # N
            out.append(preds)
        return out # B x N_i

class SimpleModel2(ChessModel):
    name = "simple_model2"
    def __init__(self, hidden_layers=[400, 200, 100]):
        super(SimpleModel2, self).__init__()
        self.fc1 = nn.Linear(13 * 8 * 8 + 4, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.fc4 = nn.Linear(hidden_layers[2], 1)

    def forward(self, x_boards):
        out = []
        for x in x_boards:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = J.zero_center(self.fc4(x)).view(1, -1) # 1 x N
            preds = J.softmax(x).view(-1), F.tanh(x).view(-1)  # N
            out.append(preds)
        return out # B x N_i
