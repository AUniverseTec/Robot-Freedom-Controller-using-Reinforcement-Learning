import tensorflow as tf
import bluetooth
import serial
import time
import numpy as np
import logging
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the TensorFlow model
model = tf.keras.models.load_model('robot_control_model.h5')

# Define the Bluetooth address and port of the robot's or drone's body
bt_addr = "XX:XX:XX:XX:XX:XX"  # Replace with the robot's or drone's Bluetooth address
bt_port = 1

# Define the serial port and baudrate of the robot's or drone's body
ser_port = '/dev/ttyUSB0'
ser_baudrate = 9600

# Define the instruction for the AI to control the robot
instruction = "move_forward"  # Replace with the desired instruction (e.g. "turn_left", "move_backward", etc.)

# Define the map data
map_data = {
    "width": 100,
    "height": 100,
    "objects": [
        {"x": 20, "y": 20, "type": "wall"},
        {"x": 50, "y": 50, "type": "target"},
        {"x": 80, "y": 80, "type": "obstacle"}
    ]
}

# Define the function to connect to the robot's or drone's body using Bluetooth
def connect_to_robot():
    try:
        bt_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        bt_sock.connect((bt_addr, bt_port))
        logger.info("Bluetooth connected successfully.")
    except bluetooth.btcommon.BluetoothError as e:
        logger.error("Bluetooth connection failed: %s", e)
        return None
    return bt_sock

# Define the function to open the serial connection to the robot's or drone's body
def open_serial_connection():
    try:
        ser = serial.Serial(ser_port, ser_baudrate, timeout=1)
        logger.info("Serial connection established.")
    except serial.SerialException as e:
        logger.error("Serial connection failed: %s", e)
        return None
    return ser

# Define the function to send commands to the robot's or drone's body
def send_command(ser, command):
    try:
        ser.write(command.encode() + b'\n')
        time.sleep(0.1)
        logger.info("Command sent: %s", command)
    except serial.SerialException as e:
        logger.error("Command sending failed: %s", e)

# Define the function to get the robot's or drone's state
def get_state(ser):
    try:
        ser.write(b'get_state\n')
        time.sleep(0.1)
        state = ser.readline().decode().strip()
        logger.info("State received: %s", state)
    except serial.SerialException as e:
        logger.error("State retrieval failed: %s", e)
        return None
    return state

# Define the function to preprocess the state data
def preprocess_state(state):
    state_data = np.array([state])
    return state_data

# Define the function to use the TensorFlow model to predict the next action
def predict_action(model, state_data):
    try:
        outputs = model.predict(state_data)
        action = np.argmax(outputs)
        logger.info("Action predicted: %s", action)
    except Exception as e:
        logger.error("Action prediction failed: %s", e)
        return None
    return action

# Define the function to map the action to a command
def map_action_to_command(action):
    if action == 0:
        command = "move_forward"
    elif action == 1:
        command = "turn_left"
    elif action == 2:
        command = "turn_right"
    elif action == 3:
        command = "move_backward"
    logger.info("Command mapped: %s", command)
    return command

# Define the function to find the closest object on the map
def find_closest_object(map_data, robot_x, robot_y):
    closest_object = None
    closest_distance = float('inf')
    for obj in map_data["objects"]:
        distance = np.sqrt((obj["x"] - robot_x) ** 2 + (obj["y"] - robot_y) ** 2)
        if distance < closest_distance:
            closest_distance = distance
            closest_object = obj
    return closest_object

# Define the function to update the robot's position on the map
def update_robot_position(map_data, robot_x, robot_y, command):
    if command == " move_forward":
        robot_x += 1
    elif command == "turn_left":
        robot_y -= 1
    elif command == "turn_right":
        robot_y += 1
    elif command == "move_backward":
        robot_x -= 1
    return robot_x, robot_y

# Main loop
while True:
    # Connect to the robot's or drone's body using Bluetooth
    bt_sock = connect_to_robot()
    if bt_sock is None:
        continue

    # Open the serial connection to the robot's or drone's body
    ser = open_serial_connection()
    if ser is None:
        continue

    # Get the robot's or drone's state
    state = get_state(ser)
    if state is None:
        continue

    # Preprocess the state data
    state_data = preprocess_state(state)

    # Use the TensorFlow model to predict the next action
    action = predict_action(model, state_data)
    if action is None:
        continue

    # Map the action to a command
    command = map_action_to_command(action)

    # Find the closest object on the map
    closest_object = find_closest_object(map_data, 0, 0)

    # Update the robot's position on the map
    robot_x, robot_y = update_robot_position(map_data, 0, 0, command)

    # Send the command to the robot's or drone's body
    send_command(ser, command)

    # Close the serial connection
    ser.close()

    # Close the Bluetooth connection
    bt_sock.close()
