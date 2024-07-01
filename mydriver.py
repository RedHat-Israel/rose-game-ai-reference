"""
Torch driver

This dreiver uses a deep neural network model designed for driving decisions based on the current state of the world,
It is as good as the trained models it uses.

This driver requires PyTorch to be installed, see: https://pytorch.org/get-started/locally/

Trained models are expected to be in the checkpoints directory.
"""

import os

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Try to import pytorch
try:
    import torch
except ImportError:
    print("Error: torch module not found. Please install it before proceeding.")
    print("       see: https://pytorch.org/get-started/locally/")
    exit()

from model import DriverModel, outputs_to_action, view_to_inputs  # noqa: E402

"""
Torch Car

This is a driver choosing the next action by consolting a neural network.
"""
driver_name = "Torch Car"


# Get the trained model
# ----------------------------------------------------------------------------------


# Model trained using a simulator
checkpoint = os.path.join(script_directory, "checkpoints", "driver.pth")

model = DriverModel()
model.load_state_dict(torch.load(checkpoint))
model.eval()


# Drive
# ----------------------------------------------------------------------------------


def build_lane_view(world):
    """
    Build a 6x4 2D array representation of the world based on the car's lane and x position.

    Args:
        world (World): An instance of the World class providing read-only
                       access to the current game state.

    Returns:
        list[list[str]]: 6x4 array representation of the world view from the car, where 4 is the specified height.
                          The bottom line is one line above the car's y position, and the top line is the line height lines above that.
                          The array provides a view of the world from the car's perspective, with the car's y position excluded.

    Notes:
        The function uses the car's y position to determine the vertical range of the 2D array.
    """
    height = 4
    width = 6
    car_y = world.car.y

    # Generate the 2D array from start_y up to world.car.y
    array = [
        [world.get((j, i)) for j in range(width)] for i in range(car_y - height, car_y)
    ]

    return array


def drive(world):
    """
    Determine the appropriate driving action based on the current state of the world.

    Args:
        world (World): An instance of the World class providing read-only access to the current game state.

    Returns:
        str: The determined action for the car to take. Possible actions include those defined in the `actions` class.
    """
    # Convert real world input, into a tensor
    view = build_lane_view(world)
    input_tensor = view_to_inputs(view, world.car.x).unsqueeze(0)

    # Forward propagte the inputs in the neural network model to get the outputs tensor
    output = model.forward(input_tensor)

    # Convert the output tensor into a real world response
    action = outputs_to_action(output)

    return action
