import cv2
import torch
import time
import serial

# ???????
m_Action_Step_01 = 0  # ??????
m_Ball_Red_X, m_Ball_Red_Y = 0, 0  # ??????
m_G3_X, m_G3_Y = 0, 0  # G3 ?????
m_Stop_Blue_Center_X, m_Stop_Blue_Center_Y = 0, 0  # ??????
m_Stop_Blue_Area = 0  # ????

# UART ???
uart_port = '/dev/ttyTHS0'
baud_rate = 115200
timeout = 1

try:
    ser = serial.Serial(port=uart_port, baudrate=baud_rate, timeout=timeout)
    print(f"Connected to {uart_port} with baud rate {baud_rate}")
except Exception as e:
    print(f"Error opening UART: {e}")
    exit()

# UART ????
def send_data(data):
    if ser.isOpen():
        ser.write(data.encode())
        print(f"UART Command: {data}")
    else:
        print("UART port is not open!")

# ?? YOLOv5 ??
model_path = '/home/wheeltec/PycharmProjects/PythonProject/Yolo/IC_AI_Model_Export/IC_AI_model_04_EIE3360_Game_Mini/weights/best_EIE3360_Game_Mini.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# ??????
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # ????? RGB ??
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLOv5 ??
    results = model(frame_rgb)

    # ??????
    detections = results.pandas().xyxy[0]
    ball_detected, g3_detected, stop_blue_detected = False, False, False

    for _, row in detections.iterrows():
        x1, y1, x2, y2, conf, cls, name = (
            int(row["xmin"]),
            int(row["ymin"]),
            int(row["xmax"]),
            int(row["ymax"]),
            row["confidence"],
            int(row["class"]),
            row["name"],
        )

        if conf > 0.65:
            # ????
            if name == "Ball_Red":
                ball_detected = True
                m_Ball_Red_X = int(((x2 - x1) / 2) + x1)
                m_Ball_Red_Y = int(((y2 - y1) / 2) + y1)

                # ???????
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Red Ball {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # ?? G3 ?
            elif name == "Gate_Orange":
                g3_detected = True
                m_G3_X = int(((x2 - x1) / 2) + x1)
                m_G3_Y = int(((y2 - y1) / 2) + y1)

                # ?? G3 ????
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(frame, f"G3 Gate {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            # ????
            elif name == "Stop_Blue":
                stop_blue_detected = True
                m_Stop_Blue_X_L, m_Stop_Blue_X_R = x1, x2
                m_Stop_Blue_Y_T, m_Stop_Blue_Y_B = y1, y2
                m_Stop_Blue_Center_X = int((x1 + x2) / 2)
                m_Stop_Blue_Center_Y = int((y1 + y2) / 2)
                m_Stop_Blue_Area = (x2 - x1) * (y2 - y1)

                # ???????
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"Blue Card {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # ????
    if m_Action_Step_01 == 0 and ball_detected and g3_detected:
        # ????? G3 ??????
        if m_Ball_Red_X < m_G3_X - 50:  # ????
            send_data("6")  # ????
        elif m_Ball_Red_X > m_G3_X + 50:  # ????
            send_data("4")  # ????
        else:  # ???? G3 ?
            send_data("8")  # ??
            if m_Ball_Red_Y > m_G3_Y - 100:  # ???? G3 ?
                send_data("8")  # ??
                print("Red ball pushed into G3 Gate!")
                m_Action_Step_01 = 1  # ????

    elif m_Action_Step_01 == 1 and stop_blue_detected:
        # ??????????????
        if m_Stop_Blue_Area > 50000:
            send_data("5")  # ??
            print("Blue card is large enough. Stopping.")
            m_Action_Step_01 = 999  # ????

    elif not ball_detected:
        # ????
        print("Searching for Red Ball...")
        send_data("1")  # ????????

    elif ball_detected and not g3_detected:
        # ?? G3 ?
        print("Searching for G3 Gate...")
        if m_Ball_Red_X < 640 - 100:  # ?????
            send_data("4")  # ??
        elif m_Ball_Red_X > 640 + 100:  # ?????
            send_data("6")  # ??
        else:
            send_data("8")  # ??

    # ?? FPS
    fps = 1 / (time.time() - prev_frame_time)
    prev_frame_time = time.time()
    cv2.putText(frame, f"{int(fps)} FPS", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

    # ?????
    cv2.imshow('EIE3360 - AI Object Detection - Group Number : XX', frame)

    # ?? 'q' ???
    if cv2.waitKey(1) & 0xFF == ord('q'):
        send_data('B')  # ????
        break

# ????
cap.release()
cv2.destroyAllWindows()