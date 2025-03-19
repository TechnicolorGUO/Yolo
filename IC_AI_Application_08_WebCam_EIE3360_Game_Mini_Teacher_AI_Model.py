

import cv2
import torch
import numpy as np
import math
import time

import serial

m_Action_Count_01 = 0
m_Action_Step_01 = 0
m_Action_Step_02 = 0
m_Action_Ball_Green_Step_01 = 0
m_Action_Ball_Red_Step_01 = 0
last_cmd_time = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
cmd_interval = 0.1
sleep_flag = 1


# Ball_Green
# Ball_Red
# Gate_Orange
# Gate_White
# Hand
# Stop_Blue
# Stop_Yellow

m_Hand_X_L = 0
m_Hand_X_R = 0
m_Hand_Y_T = 0
m_Hand_Y_B = 0

m_Stop_Yellow_X_L = 0
m_Stop_Yellow_X_R = 0
m_Stop_Yellow_Y_T = 0
m_Stop_Yellow_Y_B = 0
m_Stop_Yellow_Center_X = 0
m_Stop_Yellow_Center_Y = 0
#green boundary
m_Line_B_XD = 150
m_Line_A_XD =230  #280

m_Stop_Blue_X_L = 0
m_Stop_Blue_X_R = 0
m_Stop_Blue_Y_T = 0
m_Stop_Blue_Y_B = 0
m_Stop_Blue_Center_X = 0
m_Stop_Blue_Center_Y = 0

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
        #print(f"UART Command : {data}")
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
    # print("0")
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
    
    m_Line_A_X0 = int(1280/2)
    m_Line_A_Y0 = int(720/2)
    m_Line_A_X1 = int(m_Line_A_X0-m_Line_A_XD)
    m_Line_A_X2 = int(m_Line_A_X0+m_Line_A_XD)

    start_point = (m_Line_A_X1, 0)      # Starting point (x, y)
    end_point   = (m_Line_A_X1, 200)    # Ending point (x, y)
    color = (255, 0, 0)               # Line color (BGR): Blue
    thickness = 1                     # Line thickness
    #cv2.line(frame, start_point, end_point, m_Line_A_Color, thickness)

    start_point = (m_Line_A_X2, 0)      # Starting point (x, y)
    end_point   = (m_Line_A_X2, 200)    # Ending   point (x, y)
    color = (255, 0, 0)               # Line color (BGR): Blue
    thickness = 1                     # Line thickness
    #cv2.line(frame, start_point, end_point, m_Line_A_Color, thickness)

    """m_Line_C_Color = (0, 84, 255)
    m_Line_C_XD =280  #280
    m_Line_C_X0 = int(1280/2)
    m_Line_C_Y0 = int(720/2)
    m_Line_C_X1 = int(m_Line_C_X0-m_Line_C_XD)
    m_Line_C_X2 = int(m_Line_C_X0+m_Line_C_XD)

    start_point = (m_Line_C_X1, 0)      # Starting point (x, y)
    end_point   = (m_Line_C_X1, 200)    # Ending point (x, y)
    color = (255, 0, 0)               # Line color (BGR): Blue
    thickness = 1                     # Line thickness
    #cv2.line(frame, start_point, end_point, m_Line_C_Color, thickness)
    start_point = (m_Line_C_X2, 0)      # Starting point (x, y)
    end_point   = (m_Line_C_X2, 200)    # Ending   point (x, y)
    color = (255, 0, 0)               # Line color (BGR): Blue
    thickness = 1                     # Line thickness
    #cv2.line(frame, start_point, end_point, m_Line_A_Color, thickness)"""

    m_Line_B_Color = (5, 239, 155)
    
    m_Line_B_X0 = int(1280/2)
    m_Line_B_Y0 = int(720/2)
    m_Line_B_X1 = int(m_Line_B_X0-m_Line_B_XD)
    m_Line_B_X2 = int(m_Line_B_X0+m_Line_B_XD)

    start_point = (m_Line_B_X1, 0)      # Starting point (x, y)
    end_point   = (m_Line_B_X1, 200)    # Ending point (x, y)
    color = (255, 0, 0)               # Line color (BGR): Blue
    thickness = 1                     # Line thickness
    #cv2.line(frame, start_point, end_point, m_Line_B_Color, thickness)

    start_point = (m_Line_B_X2, 0)      # Starting point (x, y)
    end_point   = (m_Line_B_X2, 200)    # Ending   point (x, y)
    color = (255, 0, 0)               # Line color (BGR): Blue
    thickness = 1                     # Line thickness
    #cv2.line(frame, start_point, end_point, m_Line_B_Color, thickness)

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
            if(name=='green_ball'):
                m_Object_Color = (0, 128, 0)
            elif(name=='red_ball'):
                m_Object_Color = (0, 0, 255)

            elif(name=='orange_gate'):
                m_Object_Color = (0, 165, 255)
            elif(name=='white_gate'):
                m_Object_Color = (255, 255, 255)
                """elif(name=='Hand'):
                m_Object_Color = (39, 57, 88)
                m_Hand_X_L = x1
                m_Hand_X_R = x2
                m_Hand_Y_T = y1
                m_Hand_Y_B = y2"""
            elif(name=='stop_blue'):
                m_Object_Color = (255, 0, 0)
                m_Stop_Blue_X_L = x1
                m_Stop_Blue_X_R = x2
                m_Stop_BLue_Y_T = y1
                m_Stop_Blue_Y_B = y2
                m_Stop_Blue_Center_X = int(((x2-x1)/2)+x1)
                m_Stop_Blue_Center_Y = int(((y2-y1)/2)+y1)
            elif(name=='stop_yellow'):
                m_Object_Color = (0, 255, 255)
                m_Stop_Yellow_X_L = x1
                m_Stop_Yellow_X_R = x2
                m_Stop_Yellow_Y_T = y1
                m_Stop_Yellow_Y_B = y2
                m_Stop_Yellow_Center_X = int(((x2-x1)/2)+x1)
                m_Stop_Yellow_Center_Y = int(((y2-y1)/2)+y1)

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
            m_Stop_Yellow_Size = m_Stop_Yellow_X_R - m_Stop_Yellow_X_L
            #print(m_Stop_Yellow_Size)
            if (name == 'green_ball' or m_Action_Ball_Green_Step_01 == 1) and m_Action_Step_02!=999:
                # -------------------------
                # GREEN BALL ACTIONS
                # -------------------------
                if m_Action_Ball_Green_Step_01 == 0:
                    # Approach the green ball by checking the object's center relative
                    # to the boundaries (using the same parameters as red_ball)
                    if (m_Object_Center_X < m_Line_B_X1) and (m_Action_Ball_Green_Step_01 == 0):
                        m_Line_B_XD = 80
                        send_data('4')  # command to turn left
                        if time.time() - last_cmd_time[6] > cmd_interval:
                            send_data('B')
                            last_cmd_time[6] = time.time()
                        print("finding green, turn left")
                        
                    elif (m_Line_B_X2 < m_Object_Center_X) and (m_Action_Ball_Green_Step_01 == 0):
                        m_Line_B_XD = 80
                        send_data('6')  # command to turn right
                        if time.time() - last_cmd_time[7] > cmd_interval:
                            send_data('B')
                            last_cmd_time[7] = time.time()
                        print("finding green, turn right")
                        
                    else:
                        # Use the size (width) of the green ball to decide the motion
                        width_green_ball = x2 - x1
                        if width_green_ball < 175 and (m_Action_Ball_Green_Step_01 == 0):
                            m_Line_B_XD = 100
                            send_data('8')  # forward command
                            print(width_green_ball)
                            print("forward to green ball")
                            
                        elif 175 <= width_green_ball < 270 and (m_Action_Ball_Green_Step_01 == 0):
                            m_Line_B_XD = 60
                            send_data('B')
                            if time.time() - last_cmd_time[8] > 0.05:
                                send_data('8')
                                last_cmd_time[8] = time.time()
                                print("stop left")
                            print(width_green_ball)
                            print("approach to green ball")
                            
                        else:
                            # When the green ball is close enough
                            m_Action_Ball_Green_Step_01 = 1
                            send_data('B')
                            print("I get green ball!")
                            
                else:
                    # -------------------------
                    # YELLOW GATE (STOP) ACTIONS
                    # -------------------------
                    # After acquiring the green ball, we now look for the yellow gate.
                    print("pink left", str(m_Line_A_X1))
                    print("pink right", str(m_Line_A_X2))
                    print("yellow center", str(m_Stop_Yellow_Center_X))
                    
                    if m_Stop_Yellow_Center_X < m_Line_A_X1:
                        send_data('4')  # turn left toward the gate
                        m_Line_A_XD = 400
                        if time.time() - last_cmd_time[9] > 0.1:
                            send_data('B')
                        last_cmd_time[9] = time.time()
                        print("stop left")
                        print("finding yellow gate, turn left")
                        
                    elif m_Line_A_X2 < m_Stop_Yellow_Center_X:
                        send_data('6')  # turn right toward the gate
                        m_Line_A_XD = 400
                        if time.time() - last_cmd_time[10] > 0.1:
                            send_data('B')
                            last_cmd_time[10] = time.time()
                            print("stop right")
                        print("finding yellow gate, turn right")
                        
                    else:
                        # Calculate the size (and height) of the yellow gate area
                        m_Stop_Yellow_Size = m_Stop_Yellow_X_R - m_Stop_Yellow_X_L
                        m_Stop_Yellow_Size_Y = -m_Stop_Yellow_Y_T + m_Stop_Yellow_Y_B
                        print("The size of stop_yellow is: " + str(m_Stop_Yellow_Size))
                        print("The size of stop_yellow height is: " + str(m_Stop_Yellow_Size_Y))
                        
                        if m_Stop_Yellow_Size > 175:
                            m_Line_A_XD = 400
                        if m_Stop_Yellow_Size < 220:
                            print("move forward to gate")
                            send_data('B')
                            send_data('8')
                            if time.time() - last_cmd_time[11] > 0.1:
                                send_data('B')
                                last_cmd_time[11] = time.time()
                                print("stop forward")
                        else:
                            send_data('B')
                            time.sleep(0.1)
                            send_data('W')
                            time.sleep(0.7)
                            cv2.putText(frame, "END", (m_Line_B_X1 + 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            send_data('B')
                            print("END")
                            m_Line_B_XD = 150
                            m_Action_Step_02 = 999

            # (The rest of your code—including the red_ball branch—remains unchanged.)
            elif((name=='red_ball' or m_Action_Ball_Red_Step_01==1) and m_Action_Step_01!=999):
                if(m_Action_Ball_Red_Step_01==0):
                    if((m_Object_Center_X<m_Line_B_X1)and (m_Action_Ball_Red_Step_01==0)):
                        m_Line_B_XD = 80
                        send_data('4') #4 #a
                        # time.sleep(0.1)
                        # send_data("B")
                        if time.time() - last_cmd_time[0] > cmd_interval:
                            send_data('B')
                            last_cmd_time[0] = time.time()
                            print("stop left")
                        print("finding red,turn left")
                        
                    elif((m_Line_B_X2<m_Object_Center_X) and (m_Action_Ball_Red_Step_01==0)):
                        m_Line_B_XD = 80
                        # while (m_Object_Center_X>640):
                        send_data('6') #6 #d
                        # time.sleep(0.1)
                        # send_data("B")
                        if time.time() - last_cmd_time[1] > cmd_interval:
                            send_data('B')
                            last_cmd_time[1] = time.time()
                            print("stop left")
                        print("finding red,turn right")
                        
                        
                    else:
                        width_red_ball = x2 - x1
                        if(width_red_ball<(175) and (m_Action_Ball_Red_Step_01==0)):#250
                            m_Line_B_XD = 110
                            send_data('8') #8 #w
                            
                            print(width_red_ball)
                            print("forward to red ball")
                        elif(width_red_ball<(270) and width_red_ball>=(175) and (m_Action_Ball_Red_Step_01==0)):
                            m_Line_B_XD = 60
                            send_data('B')
                            # time.sleep(0.05)
                            # send_data('8') #8 #w
                            if time.time() - last_cmd_time[4] > 0.05:
                                send_data('8')
                                last_cmd_time[4] = time.time()
                                print("stop left")
                            #time.sleep(0.1)

                            print(width_red_ball)
                            print("approach to reb ball")
                        else:
                            m_Action_Ball_Red_Step_01 = 1
                            send_data('B')
                            print("I get red ball!")
                else:
                    print("pink left", str(m_Line_A_X1))
                    print("pink right", str(m_Line_A_X2))
                    print("blue center", str(m_Stop_Blue_Center_X))

                    if(m_Stop_Blue_Center_X<m_Line_A_X1):
                        # send_data('B')
                        send_data('4') #4 #a
                        m_Line_A_XD = 350
                        if time.time() - last_cmd_time[2] > 0.1:
                            send_data('B')
                        last_cmd_time[2] = time.time()
                        print("stop left")
                        print("finding blue gate,turn left")
                    
                    
                    elif(m_Line_A_X2<m_Stop_Blue_Center_X):
                        # send_data('B')
                        send_data('6') #6 #d
                        m_Line_A_XD = 350
                        if time.time() - last_cmd_time[3] > 0.1:
                            send_data('B')
                            last_cmd_time[3] = time.time()
                            print("stop right")
                        print("finding blue gate,turn right")
                    
                        # send_data('B')
                        # time.sleep(0.1)
                    else:
                        # m_Line_A_XD = 280
                        m_Stop_Blue_Size = m_Stop_Blue_X_R - m_Stop_Blue_X_L
                        m_Stop_Blue_Size_Y = - m_Stop_Blue_Y_T + m_Stop_Blue_Y_B
                        # print(m_Stop_Blue_Size)
                        print("The size of stop_blue is: " + str(m_Stop_Blue_Size))   
                        print("The size of stop_blue height is: " + str(m_Stop_Blue_Size_Y))
                        # send_data('B')  
                        if (m_Stop_Blue_Size>175):
                            m_Line_A_XD = 350
                        if(m_Stop_Blue_Size<220):
                        # if (m_Stop_Blue_Size_Y < 140):
                            print("move forward to gate")
                            send_data('B')
                            send_data('8') #8 #w
                            if time.time() - last_cmd_time[5] > 0.1:
                                send_data('B')
                                last_cmd_time[5] = time.time()
                                print("stop forward")
                            """time.sleep(3)
                            cv2.putText(frame,"END",(m_Line_B_X1+20, 20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2,)
                            send_data('B')
                            print("stopped")"""
                            #m_Action_Step_01 = 999
                            #time.sleep(0.1)
                            # send_data('B')
        
                        else:
                            send_data('4')
                            time.sleep(0.3)
                            send_data('W')
                            time.sleep(0.7)
                            cv2.putText(frame,"END",(m_Line_B_X1+20, 20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2,)
                            send_data('B')
                            print("END")
                            m_Line_B_XD = 150
                            m_Action_Step_01 = 999
                            
    
    # Calculating the fps
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = "%02d" % (fps,) + ' fps'

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