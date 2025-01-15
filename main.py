import cv2
import asyncio
import requests
from ultralytics import YOLO
from io import BytesIO
import numpy as np
from telegram import Bot
import threading
import time
# Tải mô hình YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")  # Thay thế bằng đường dẫn đúng tới model của bạn

# Thông tin Telegram
TELEGRAM_TOKEN = ''  # Thay thế bằng token bot Telegram của bạn
CHAT_ID = '6238290486'  # Thay thế bằng chat ID của bạn (có thể là nhóm hoặc cá nhân)
bot = Bot(token=TELEGRAM_TOKEN)
point = [15,16]
# Biến toàn cục để lưu trạng thái vẽ ROI
drawing = False
roi = [0, 0, 0, 0]
start_point = (0, 0)
end_point = (0, 0)
last_sent_time =0

def send_telegram_message_and_frame(message, frame):
    """Gửi tin nhắn và frame trực tiếp qua Telegram"""
    try:
        # Gửi tin nhắn
        message_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        message_payload = {"chat_id": CHAT_ID, "text": message}
        message_response = requests.post(message_url, data=message_payload, timeout=5)
        
        if message_response.status_code == 200:
            print("Đã gửi tin nhắn thành công!")
        else:
            print("Lỗi khi gửi tin nhắn:", message_response.text)

        # Chuyển frame thành byte
        _, buffer = cv2.imencode('.jpg', frame)
        photo_bytes = buffer.tobytes()

        # Gửi frame qua Telegram
        photo_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        photo_payload = {"chat_id": CHAT_ID}
        photo_files = {"photo": ("frame.jpg", photo_bytes, "image/jpeg")}
        photo_response = requests.post(photo_url, data=photo_payload, files=photo_files, timeout=5)
        
        if photo_response.status_code == 200:
            print("Đã gửi ảnh thành công!")
        else:
            print("Lỗi khi gửi ảnh:", photo_response.text)
    
    except requests.exceptions.Timeout:
        print("Lỗi: Yêu cầu bị timeout.")
    except Exception as e:
        print("Exception:", e)

def send_alert_frame_async(message, frame):
    """Gửi tin nhắn và frame trên luồng riêng"""
    st = time.time()
    thread = threading.Thread(target=send_telegram_message_and_frame, args=(message, frame))
    thread.start()
    print(time.time() - st)
def mouse_callback(event, x, y, flags, param):
    """
    Callback chuột để vẽ ROI
    """
    global drawing, start_point, end_point, roi

    if event == cv2.EVENT_LBUTTONDOWN:  # Khi nhấn chuột trái
        drawing = True
        start_point = (x, y)
        end_point = (x, y)
    
    elif event == cv2.EVENT_MOUSEMOVE:  # Khi di chuyển chuột
        if drawing:
            end_point = (x, y)
    
    elif event == cv2.EVENT_LBUTTONUP:  # Khi thả chuột trái
        drawing = False
        end_point = (x, y)
        roi = [min(start_point[0], end_point[0]), min(start_point[1], end_point[1]),
               max(start_point[0], end_point[0]), max(start_point[1], end_point[1])]


def draw_roi(frame, roi):
    """
    Vẽ vùng ROI trên khung hình
    """
    if roi[2] > roi[0] and roi[3] > roi[1]:  # Đảm bảo ROI hợp lệ
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 2)
    return frame


def check_keypoints_in_roi(keypoints, roi):
    """
    Kiểm tra nếu keypoints [15, 16] nằm trong ROI
    """
    x1, y1, x2, y2 = roi
    for i in point:
        if len(keypoints.xy[0]) > i:  # Đảm bảo keypoint tồn tại
            x, y = keypoints.xy[0][i][0], keypoints.xy[0][i][1]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True, (int(x), int(y))
    return False, None


def process_frame(frame, roi):
    """
    Xử lý khung hình để phát hiện pose keypoints và kiểm tra ROI
    """
    results = model(frame)
    warning = False
    warning_point = None

    for result in results:
        keypoints = result.keypoints.cpu().numpy()  # Lấy keypoints
        in_roi, warning_point = check_keypoints_in_roi(result.keypoints, roi)
        warning = warning or in_roi

        # Vẽ keypoints lên frame
        for i in point:
            if len(keypoints.xy[0]) > i:  # Đảm bảo keypoint tồn tại
                x, y = keypoints.xy[0][i][0], keypoints.xy[0][i][1]
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        

    # Vẽ vùng ROI
    frame = draw_roi(frame, roi)

    return frame, warning, warning_point


def main():
    global roi,last_sent_time

    # Đọc ảnh từ camera hoặc video
    cap = cv2.VideoCapture(2)  # Thay 0 bằng đường dẫn file video nếu muốn dùng video
    cv2.namedWindow("Foot Detection")
    cv2.setMouseCallback("Foot Detection", mouse_callback)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame,1)
        # Xử lý khung hình
        frame, warning, warning_point = process_frame(frame, roi)

        # Vẽ vùng ROI tạm thời khi đang kéo chuột
        if drawing:
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
        current_time = time.time()
        # Cảnh báo và gửi hình ảnh nếu có xâm nhập
        if warning:
            cv2.putText(
                frame, 
                "Warning: Point in ROI!", 
                (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
            )
            if warning_point:
                cv2.circle(frame, warning_point, 10, (0, 255, 255), -1)

                send_alert_frame_async("Co xam nhap !!!", frame )
                last_sent_time = current_time
                

        # Hiển thị ảnh
        cv2.imshow("Foot Detection", frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
