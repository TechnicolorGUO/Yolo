import cv2
import torch
import numpy as np
import math
import time

import serial


# Ball_Green
# Ball_Red
# Gate_Orange
# Gate_White
# Hand
# Stop_Blue
# Stop_Yellow


# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0

# UART port and configuration
uart_port = '/dev/ttyTHS0'  # Replace with the correct UART port (e.g., ttyTHS1, ttyTHS2)
baud_rate = 115200          # Baud rahhhhte (e.g., 9600, 115200)
timeout = 1                 # Timeout for reading in seconds

# Initialize the serial connection
try:
    ser = serial.Serial(
        port=uart_port,
        baudrate=baud_rate,
        timeout=timeout
    )
    print(f"Connected to {uart_port} with baud rate {baud_rate}")
except Exception as e:
    print(f"Error opening UART: {e}")
    exit()

# Function to send data over UART
def send_data(data):
    if ser.isOpen():
        ser.write(data.encode())  # Convert string to bytes and send
        print(f"UART Command : {data}")
    else:
        print("UART port is not open!")

# Function to receive data over UART
def receive_data():
    if ser.isOpen():
        data = ser.readline()  # Read a line of data (ends with \n)
        if data:
            print(f"Received: {data.decode().strip()}")  # Decode bytes to string
        else:
            print("No data received.")
    else:
        print("UART port is not open!")

send_data('B')


# Load the YOLOv5 model
# Replace 'yolov5s.pt' with your trained/custom model path if applicable
model_path = '/home/wheeltec/PycharmProjects/PythonProject/Yolo/IC_AI_Model_Export/IC_AI_model_04_EIE3360_Game_Mini/weights/best_EIE3360_Game_Mini.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Set the webcam source (0 for the default camera, or specify a video file path)
webcam_source = 0  # Change to 1, 2, etc., if multiple cameras are connected

# Open the webcam
cap = cv2.VideoCapture(webcam_source)

# Set webcam resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Check if webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting webcam... Press 'q' to quit.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Ball_Green
    # Ball_Red
    # Gate_Orange
    # Gate_White
    # Hand
    # Stop_Blue
    # Stop_Yellow
    
    m_Line_A_Color = (167, 84, 255)
    m_Line_A_XD = 400 #280
    m_Line_A_X0 = int(1280/2)
    m_Line_A_Y0 = int(720/2)
    m_Line_A_X1 = int(m_Line_A_X0-m_Line_A_XD)
    m_Line_A_X2 = int(m_Line_A_X0+m_Line_A_XD)

    start_point = (m_Line_A_X1, 0)      # Starting point (x, y)
    end_point   = (m_Line_A_X1, 200)    # Ending point (x, y)
    color = (255, 0, 0)               # Line color (BGR): Blue
    thickness = 2                     # Line thickness
    cv2.line(frame, start_point, end_point, m_Line_A_Color, thickness)

    start_point = (m_Line_A_X2, 0)      # Starting point (x, y)
    end_point   = (m_Line_A_X2, 200)    # Ending   point (x, y)
    color = (255, 0, 0)               # Line color (BGR): Blue
    thickness = 2                     # Line thickness
    cv2.line(frame, start_point, end_point, m_Line_A_Color, thickness)

    m_Line_B_Color = (5, 239, 155)
    m_Line_B_XD = 100
    m_Line_B_X0 = int(1280/2)
    m_Line_B_Y0 = int(720/2)
    m_Line_B_X1 = int(m_Line_B_X0-m_Line_B_XD)
    m_Line_B_X2 = int(m_Line_B_X0+m_Line_B_XD)

    start_point = (m_Line_B_X1, 0)      # Starting point (x, y)
    end_point   = (m_Line_B_X1, 200)    # Ending point (x, y)
    color = (255, 0, 0)               # Line color (BGR): Blue
    thickness = 2                     # Line thickness
    cv2.line(frame, start_point, end_point, m_Line_B_Color, thickness)

    start_point = (m_Line_B_X2, 0)      # Starting point (x, y)
    end_point   = (m_Line_B_X2, 200)    # Ending   point (x, y)
    color = (255, 0, 0)               # Line color (BGR): Blue
    thickness = 2                     # Line thickness
    cv2.line(frame, start_point, end_point, m_Line_B_Color, thickness)

    # Convert the frame to RGB (YOLOv5 expects RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(frame_rgb)

    # Parse detection results and plot them directly on the frame
    detections = results.pandas().xyxy[0]  # Get the detections as a Pandas DataFrame
    for _, row in detections.iterrows():
        # Extract bounding box coordinates and other info
        x1, y1, x2, y2, conf, cls, name = (
            int(row["xmin"]),
            int(row["ymin"]),
            int(row["xmax"]),
            int(row["ymax"]),
            row["confidence"],
            int(row["class"]),
            row["name"],
        )

        if(0.5<conf):

            m_Object_Color = (0, 255, 0)
            if(name=='Ball_Green'):
                m_Object_Color = (0, 128, 0)
            elif(name=='Ball_Red'):
                m_Object_Color = (0, 0, 255)
            elif(name=='Gate_Orange'):
                m_Object_Color = (0, 165, 255)
            elif(name=='Gate_White'):
                m_Object_Color = (255, 255, 255)
            elif(name=='Hand'):
                m_Object_Color = (39, 57, 88)
            elif(name=='Stop_Blue'):
                m_Object_Color = (255, 0, 0)
            elif(name=='Stop_Yellow'):
                m_Object_Color = (0, 255, 255)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), m_Object_Color, thickness=2)
            
            # Draw Center box
            m_Object_Center_X = int(((x2-x1)/2)+x1)
            m_Object_Center_Y = int(((y2-y1)/2)+y1)
            cv2.rectangle(frame, (m_Object_Center_X-5, m_Object_Center_Y-5), (m_Object_Center_X+5, m_Object_Center_Y+5), m_Object_Color, thickness=2)

            # Put label and confidence score
            label = f"{name} {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                m_Object_Color, #(0, 255, 0),
                2,
            )
            
    
    # Calculating the fps
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = "%02d" % (fps,) + ' fps'
    # putting the FPS count on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, fps, (5, 30), font, 1, (100, 255, 0), 2, cv2.LINE_AA)
    
    # Display the frame with detections
    cv2.imshow('EIE3360 - AI Object Detection - Group Number : XX', frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting webcam...")
        send_data('B')
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()