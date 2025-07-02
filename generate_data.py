"""
Data generation script

This script generates training data for the AI driver by creating world states and their corresponding actions.
The data is saved in a traditional data directory structure with separate files for inputs and labels.

The script can generate labels using either:
1. Built-in simulator (default) - uses hardcoded logic for obstacle handling
2. Driver server - queries an external driver server via HTTP requests for action labels

Usage:
    # Using built-in simulator
    python generate_data.py --num-samples 5000
    
    # Using external driver server
    python generate_data.py --num-samples 5000 --server-url http://localhost:8081
    
    # Full usage
    python generate_data.py [--num-samples NUM_SAMPLES] [--data-dir DATA_DIR] [--server-url SERVER_URL]
"""

import os
import json
import random
import argparse
import requests
import time
from model import actions, obstacles, view_to_inputs


def generate_obstacle_array(width=6, height=4):
    """
    Generates a 2D array with random obstacles.

    Parameters:
        width (int): The width of the 2D array. Default is 6.
        height (int): The height of the 2D array. Default is 4.

    Returns:
        list[list[str]]: 2D array with random obstacles.
    """
    OBSTACLES = ["", "crack", "trash", "penguin", "bike", "water", "barrier"]

    array = [["" for _ in range(width)] for _ in range(height)]

    for i in range(height):
        obstacle = random.choice(OBSTACLES)
        position = random.randint(0, width // 2 - 1)
        # lane A
        array[i][position] = obstacle
        # lane B
        array[i][width // 2 + position] = obstacle

    return array


def driver_simulator(array, car_x, width=6, height=4):
    """
    Simulates the driver's decision based on the obstacle in front of the car.

    Args:
        array (list[list[str]]): 2D array representation of the world with obstacles as strings.
        car_x (int): The car's x position.
        width (int): The width of the 2D array. Default is 6.
        height (int): The height of the 2D array. Default is 4.

    Returns:
        str: The determined action for the car to take. Possible actions include those defined in the `actions` class.
    """
    obstacle = array[height - 1][car_x]

    # Define a dictionary to map obstacles to actions
    action_map = {
        obstacles.PENGUIN: actions.PICKUP,
        obstacles.WATER: actions.BRAKE,
        obstacles.CRACK: actions.JUMP,
        obstacles.NONE: actions.NONE,
    }

    # Determine the action based on the obstacle
    action = action_map.get(obstacle)

    # If the obstacle is not in the dictionary, determine the action based on the car's x position
    if action is None:
        action = actions.RIGHT if (car_x % (width // 2)) == 0 else actions.LEFT

    return action


def query_driver_server(array, car_x, server_url, width=6, height=4, max_retries=3):
    """
    Query a driver server to get the correct action for a given world state.

    Args:
        array (list[list[str]]): 2D array representation of the world with obstacles as strings.
        car_x (int): The car's x position.
        server_url (str): URL of the driver server (e.g., "http://localhost:8081")
        width (int): The width of the 2D array. Default is 6.
        height (int): The height of the 2D array. Default is 4.
        max_retries (int): Maximum number of retry attempts if request fails.

    Returns:
        str: The action returned by the driver server.
    """
    # Create the payload
    payload = {
        "info": {"car": {"x": car_x, "y": height}},  # Car is at the bottom of the view
        "track": array,
    }

    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            response = requests.post(
                server_url, json=payload, headers=headers, timeout=5
            )
            response.raise_for_status()

            result = response.json()
            if "info" in result and "action" in result["info"]:
                return result["info"]["action"]
            else:
                print(f"Warning: Unexpected response format from server: {result}")
                return actions.NONE

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}: Error querying driver server: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
            else:
                print(
                    f"Failed to query server after {max_retries} attempts, using default action"
                )
                return actions.NONE

    return actions.NONE


def generate_training_data(num_samples, data_dir, server_url=None):
    """
    Generate training data and save it to the data directory.

    Args:
        num_samples (int): Number of training samples to generate
        data_dir (str): Directory to save the training data
        server_url (str, optional): URL of driver server to query for labels.
                                  If None, uses built-in simulator.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    inputs_data = []
    labels_data = []

    labeling_method = "driver server" if server_url else "built-in simulator"
    print(f"Generating {num_samples} training samples using {labeling_method}...")

    if server_url:
        print(f"Driver server URL: {server_url}")

    for i in range(num_samples):
        if i % 1000 == 0:
            print(f"Generated {i}/{num_samples} samples")

        # Generate random world state
        car_x = random.randint(0, 5)
        array = generate_obstacle_array()

        # Get the correct action using either server or simulator
        if server_url:
            correct_action = query_driver_server(array, car_x, server_url)
        else:
            correct_action = driver_simulator(array, car_x)

        # Create training sample
        inputs_data.append({"world_array": array, "car_x": car_x})
        labels_data.append(correct_action)

    # Save inputs and labels
    inputs_file = os.path.join(data_dir, "inputs.json")
    labels_file = os.path.join(data_dir, "labels.json")

    print(f"Saving inputs to {inputs_file}")
    with open(inputs_file, "w") as f:
        json.dump(inputs_data, f, indent=2)

    print(f"Saving labels to {labels_file}")
    with open(labels_file, "w") as f:
        json.dump(labels_data, f, indent=2)

    print(f"Successfully generated {num_samples} training samples!")
    print(f"Data saved in: {data_dir}")
    print(f"Labeling method used: {labeling_method}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for the AI driver."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of training samples to generate (default: 10000)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory to save the training data (default: data)",
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help="URL of driver server to query for labels (e.g., http://localhost:8081). If not provided, uses built-in simulator.",
    )

    args = parser.parse_args()

    generate_training_data(args.num_samples, args.data_dir, args.server_url)


if __name__ == "__main__":
    main()
