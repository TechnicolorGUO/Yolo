import cv2
import torch
import numpy as np
import math
import time

import serial
import time

m_Action_Count_01 = 0
m_Action_Step_01 = 0

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
model_path = '/home/wheeltec/PycharmProjects/PythonProject/Yolo/IC_AI_Model_Export/IC_AI_model_03_Student/weights/best.pt'
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

    m_Line_X0 = int(1280/2)
    m_Line_Y0 = int(720/2)
    m_Line_X1 = int(m_Line_X0-200)
    m_Line_Y1 = int(m_Line_Y0-200)
    m_Line_X2 = int(m_Line_X0+200)
    m_Line_Y2 = int(m_Line_Y0+200)

    start_point = (m_Line_X1, 0)      # Starting point (x, y)
    end_point   = (m_Line_X1, 720)    # Ending point (x, y)
    color = (255, 0, 0)               # Line color (BGR): Blue
    thickness = 1                     # Line thickness
    cv2.line(frame, start_point, end_point, color, thickness)

    start_point = (m_Line_X2, 0)      # Starting point (x, y)
    end_point   = (m_Line_X2, 720)    # Ending   point (x, y)
    color = (255, 0, 0)               # Line color (BGR): Blue
    thickness = 1                     # Line thickness
    cv2.line(frame, start_point, end_point, color, thickness)

    # Convert the frame to RGB (YOLOv5 expects RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(frame_rgb)

    # Parse detection results and plot them directly on the frame
    detections = results.pandas().xyxy[0]  # Get the detections as a Pandas DataFrame
    m_Number_Object_01 = len(detections)
    if(0<m_Number_Object_01):
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
            
            if(0.6<conf):
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                
                # Draw Center box
                m_Object_Center_X = int(((x2-x1)/2)+x1)
                m_Object_Center_Y = int(((y2-y1)/2)+y1)
                cv2.rectangle(frame, (m_Object_Center_X-5, m_Object_Center_Y-5), (m_Object_Center_X+5, m_Object_Center_Y+5), color=(0, 255, 0), thickness=2)

                # Object Size
                m_Object_Size_X = x2-x1
                m_Object_Size_Y = y2-y1
                m_Object_Check_X = 400

                # Put label and confidence score
                label = f"{name} {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                Temp_Message_01 = str(conf)
                if(m_Object_Size_X<m_Object_Check_X):
                    print("====================================================")
                    print("Action Count : " + str(m_Action_Count_01))
                    print("Confidence   : "+Temp_Message_01)
                    print("Object (X,Y) : " + str(m_Object_Center_X) + ", " +str(m_Object_Center_Y))
                    print("Object Size  : " + str(m_Object_Size_X))
                    print("Robot        : Forward")
                    cv2.putText(frame,"Action Count : " + str(m_Action_Count_01),(m_Line_X1+20, 20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2,)
                    cv2.putText(frame,"Confidence   : " + Temp_Message_01,(m_Line_X1+20, 40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2,)
                    cv2.putText(frame,"Object (X,Y) : " + str(m_Object_Center_X) + ", " +str(m_Object_Center_Y)  ,(m_Line_X1+20, 60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2,)
                    cv2.putText(frame,"Object Size  : " + str(m_Object_Size_X) ,(m_Line_X1+20, 80),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2,)
                    cv2.putText(frame,"Robot        : Forward"  ,(m_Line_X1+20, 100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2,)
                    send_data('8')
                else:
                    print("====================================================")
                    print("Action Count : " + str(m_Action_Count_01))
                    print("Confidence   : "+Temp_Message_01)
                    print("Object (X,Y) : " + str(m_Object_Center_X) + ", " +str(m_Object_Center_Y))
                    print("Object Size  : " + str(m_Object_Size_X))
                    print("Robot        : Stop")
                    cv2.putText(frame,"Action Count : " + str(m_Action_Count_01),(m_Line_X1+20, 20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2,)
                    cv2.putText(frame,"Confidence   : " + Temp_Message_01,(m_Line_X1+20, 40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2,)
                    cv2.putText(frame,"Object (X,Y) : " + str(m_Object_Center_X) + ", " +str(m_Object_Center_Y)  ,(m_Line_X1+20, 60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2,)
                    cv2.putText(frame,"Object Size  : " + str(m_Object_Size_X) ,(m_Line_X1+20, 80),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2,)
                    cv2.putText(frame,"Robot        : Stop"  ,(m_Line_X1+20, 100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2,)
                    send_data('B')
                
                m_Action_Count_01 = m_Action_Count_01 + 1

    # Display the frame with detections
    cv2.imshow('YOLOv5 Webcam', frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting webcam...")
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()