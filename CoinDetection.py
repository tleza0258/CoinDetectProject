import cv2
import numpy as np

capture = cv2.VideoCapture("Coin10.mp4")

if not capture.isOpened():
    print('ไม่สามารถโหลดวิดีโอได้')
else:
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f'ขนาดของเฟรม: {frame_width}x{frame_height}')

while True:
    ret, frame = capture.read()
    if not ret:
        print('เลิกอ่านเฟรม')
        break
    else:
        # บริเวณของเฟรมที่จะตรวจจับ กว้าง*สูง
        region = frame[:1920, :1920]

        # ปรับสีภาพเป็น grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        # ลดนอยส์
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        # แยกวัตถุออกจากพื้นหลัง
        threshold = cv2.adaptiveThreshold(blur, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11   , 3)   # 11, 3
        kernel = np.ones((3, 3), np.uint8) # 3,3
        closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations = 4)

        result_img = closed.copy()
        contours, hierachy = cv2.findContours(result_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        counter = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5000 or area > 65000:  #5000 ,55000
                continue
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(region, ellipse, (0, 0, 255), thickness = 5)
            counter += 1

        cv2.putText(region, f"Coin: {counter}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), thickness = 5 , lineType = cv2.LINE_AA)
    
        cv2.imshow('coindetection', region)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()