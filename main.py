# import thư viện NumPy, một thư viện Python được sử dụng để thao tác với các mảng đa chiều.
import numpy as np
# import thư viện pygame.mixer, một thư viện Python được sử dụng để phát âm thanh.
import pygame.mixer
# import lớp load_model từ thư viện Keras, một thư viện Python được sử dụng để xây dựng và huấn luyện
# các mô hình mạng nơ-ron (neural network).
from keras.models import load_model
# import thư viện image từ thư viện Keras, một thư viện Python được sử dụng để tải và xử lý dữ liệu hình ảnh.
from keras.preprocessing import image
# import thư viện OpenCV, một thư viện Python được sử dụng để xử lý hình ảnh và video.
import cv2
# import thư viện datetime, một thư viện Python được sử dụng để thao tác với thời gian và ngày tháng.
import datetime

# tải mô hình mạng nơ-ron đã được huấn luyện từ tệp mymodel.keras.
mymodel = load_model('mymodel.keras')


# Hàm này được sử dụng để phát âm thanh cảnh báo khi phát hiện người không đeo khẩu trang.
def play_alert_sound():
    pygame.mixer.init()
    pygame.mixer.music.load("alert.mp3")
    pygame.mixer.music.play()


# khởi tạo đối tượng VideoCapture, được sử dụng để đọc dữ liệu video từ webcam.
cap = cv2.VideoCapture(0)
# khởi tạo đối tượng CascadeClassifier, được sử dụng để phát hiện khuôn mặt trong hình ảnh.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Vòng lặp này sẽ tiếp tục chạy cho đến khi webcam bị đóng.
while cap.isOpened():
    # đọc một khung hình từ webcam và lưu trữ nó trong biến img.
    _, img = cap.read()
    # sử dụng đối tượng CascadeClassifier để phát hiện các khuôn mặt trong hình ảnh img.
    face = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
    # Vòng lặp này sẽ chạy qua tất cả các khuôn mặt được phát hiện trong hình ảnh img.
    for (x, y, w, h) in face:
        # Dòng mã này cắt khuôn mặt khỏi hình ảnh img.
        face_img = img[y:y + h, x:x + w]
        # Dòng mã này lưu trữ khuôn mặt đã được cắt vào tệp temp.jpg.
        cv2.imwrite('temp.jpg', face_img)
        # tải khuôn mặt đã được lưu vào tệp temp.jpg và resize nó thành kích thước (150, 150, 3).
        test_image = image.load_img('temp.jpg', target_size=(150, 150, 3))
        # chuyển đổi khuôn mặt từ định dạng hình ảnh sang định dạng mảng NumPy.
        test_image = image.img_to_array(test_image)
        # mở rộng kích thước của mảng NumPy thêm một chiều, để phù hợp với đầu vào của mô hình mạng nơ-ron.
        test_image = np.expand_dims(test_image, axis=0)
        # sử dụng mô hình mạng nơ-ron để dự đoán liệu người trong khuôn mặt có đeo khẩu trang hay không.
        pred = mymodel.predict(test_image)[0][0]
        '''
        Nếu mô hình mạng nơ-ron dự đoán rằng người trong khuôn mặt không đeo khẩu trang, 
        mã sẽ vẽ một khung màu đỏ xung quanh khuôn mặt và hiển thị văn bản "NO MASK". 
        Ngoài ra, mã sẽ phát âm thanh cảnh báo.
        '''
        if pred == 1:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(img, 'NO MASK', ((x + w) // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            play_alert_sound()
        # Nếu mô hình mạng nơ-ron dự đoán rằng người trong khuôn mặt đeo khẩu trang, mã sẽ vẽ một khung màu xanh lá cây
        # xung quanh khuôn mặt và hiển thị văn bản "MASK".
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, 'MASK', ((x + w) // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # hiển thị ngày và giờ hiện tại ở góc dưới bên phải của hình ảnh.
        datet = str(datetime.datetime.now())
        cv2.putText(img, datet, (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # hiển thị hình ảnh đã được xử lý trên màn hình.
    cv2.imshow('img', img)

    # kiểm tra xem người dùng đã nhấn phím q chưa. Nếu đã nhấn, mã sẽ thoát khỏi vòng lặp.
    if cv2.waitKey(1) == ord('q'):
        break

# giải phóng tài nguyên và đóng tất cả các cửa sổ.
cap.release()
cv2.destroyAllWindows()
