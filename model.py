"""
Torch driver

This dreiver uses a deep neural network model designed for driving decisions based on the current state of the world,
It is as good as the trained models it uses.

This driver requires PyTorch to be installed, see: https://pytorch.org/get-started/locally/

Trained models are expected to be in the checkpoints directory.
"""

try:
    import torch
except ImportError:
    print("Error: torch module not found. Please install it before proceeding.")
    print("       see: https://pytorch.org/get-started/locally/")
    exit()

import torch.nn as nn
import torch.nn.functional as F


# Game classes
# ----------------------------------------------------------------------------------


class actions:
    NONE = "none"
    RIGHT = "right"
    LEFT = "left"
    PICKUP = "pickup"
    JUMP = "jump"
    BRAKE = "brake"

    ALL = (NONE, RIGHT, LEFT, PICKUP, JUMP, BRAKE)


class obstacles:
    NONE = ""
    CRACK = "crack"
    TRASH = "trash"
    PENGUIN = "penguin"
    BIKE = "bike"
    WATER = "water"
    BARRIER = "barrier"

    ALL = (NONE, CRACK, TRASH, PENGUIN, BIKE, WATER, BARRIER)


# PyTorch NN driving model
# ----------------------------------------------------------------------------------
class DriverModel(nn.Module):
    """
    A deep neural network model designed for driving decisions based on the current state of the world.

    Architecture:
    - Input layer: Size determined by
      6 (width) x 4 (height) x 7 (possible obstacles) + 6 (car current lane)= 174 neurons.
    - Hidden Layer 1: 512 neurons, followed by batch normalization and 50% dropout.
    - Hidden Layer 2: 256 neurons, followed by batch normalization and 50% dropout.
    - Hidden Layer 3: 128 neurons, followed by batch normalization and 50% dropout.
    - Hidden Layer 4: 64 neurons, followed by batch normalization and 50% dropout.
    - Hidden Layer 5: 32 neurons, followed by batch normalization and 50% dropout.
    - Output Layer: 6 neurons, representing the possible driving actions (e.g., obstacles.ALL).

    Activation Function:
    - ReLU activation function is used for all hidden layers.
    - The output layer does not have an activation function, making this model suitable for use with a softmax function externally or a criterion like CrossEntropyLoss which combines softmax and NLLLoss.

    Regularization:
    - Dropout with a rate of 50% is applied after each hidden layer to prevent overfitting.

    Note:
    - The model expects a flattened version of the 6x4x7+6 input tensor, which should be reshaped to (batch_size, 174) before being passed to the model.
    """

    def __init__(self):
        super(DriverModel, self).__init__()

        self.fc1 = nn.Linear(6 * 4 * 7 + 6, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.dropout5 = nn.Dropout(0.5)

        self.fc6 = nn.Linear(32, 6)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)

        x = F.relu(self.bn5(self.fc5(x)))
        x = self.dropout5(x)

        x = self.fc6(x)
        return x


def view_to_inputs(array, car_lane):
    """
    Convert a 2D array representation of the world into a tensor suitable for model input.

    The function maps each obstacle in the 2D array to a one-hot encoded tensor.
    The resulting tensor is then flattened and an additional dimension is added to represent the batch size.

    Args:
        array (list[list[str]]): 2D array representation of the world with obstacles as strings.
        car_lane (int): current lane of the car, can be 0..5 inclusive.

    Returns:
        torch.Tensor: A tensor of shape (1, height * width * num_obstacle_types) suitable for model input.

    Notes:
        The function uses a predefined mapping of obstacles to indices for the one-hot encoding.
    """
    height = len(array)
    width = len(array[0])
    num_obstacles = len(obstacles.ALL)

    tensor = torch.zeros((height, width, num_obstacles))

    for i in range(height):
        for j in range(width):
            obstacle = array[i][j]
            tensor[i, j, obstacles.ALL.index(obstacle)] = 1

    world_tensor = tensor.view(-1)
    car_lane_tensor = torch.zeros(width)
    car_lane_tensor[car_lane] = 1

    return torch.cat((world_tensor, car_lane_tensor))


def outputs_to_action(output):
    """
    Convert the model's output tensor into a driving action based on the current state of the world.

    The function determines the car's intended action (left, forward, right) based on the model's output.

    Args:
        output (torch.Tensor): The model's output tensor, typically representing probabilities for each position.

    Returns:
        str: The determined action for the car to take. Possible actions include those defined in the `actions` class.

    Notes:
        The function uses a predefined mapping of obstacles to actions to determine the appropriate action when moving forward.
    """
    position_index = torch.argmax(output).item()
    action = actions.ALL[position_index]

    return action
