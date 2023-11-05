"""
Created on Sunday, November 5, 2023

@author: PhamVanPhuong x NguyenSinhTung
"""
# import thư viện NumPy, một thư viện Python được sử dụng để thao tác với các mảng đa chiều.
import numpy as np
# import thư viện Matplotlib, một thư viện Python được sử dụng để tạo các đồ thị và biểu đồ.
import matplotlib.pyplot as plt
# import lớp Conv2D từ thư viện Keras, một thư viện Python được sử dụng để xây dựng và huấn luyện
# các mô hình mạng nơ-ron (neural network). Lớp Conv2D được sử dụng để thực hiện các phép toán
# tích chập trên dữ liệu hình ảnh.
from keras.layers import Conv2D
# import lớp MaxPooling2D từ thư viện Keras. Lớp MaxPooling2D được sử dụng để giảm kích thước
# của dữ liệu hình ảnh bằng cách lấy giá trị lớn nhất trong một vùng lân cận.
from keras.layers import MaxPooling2D
# import lớp Flatten từ thư viện Keras. Lớp Flatten được sử dụng để chuyển đổi dữ liệu hình ảnh
# thành một mảng một chiều.
from keras.layers import Flatten
#  import lớp Dense từ thư viện Keras. Lớp Dense được sử dụng để tạo ra các lớp nơ-ron được kết nối đầy đủ.
from keras.layers import Dense
# import lớp Sequential từ thư viện Keras. Lớp Sequential được sử dụng để xây dựng các mô hình mạng nơ-ron theo chuỗi.
from keras.models import Sequential
# import lớp load_model từ thư viện Keras. Lớp load_model được sử dụng để tải các mô hình mạng nơ-ron đã được huấn
# luyện.
from keras.models import load_model
# import thư viện image từ thư viện Keras. Thư viện image được sử dụng để tải và xử lý dữ liệu hình ảnh.
from keras.preprocessing import image
# import lớp ImageDataGenerator từ thư viện Keras. Lớp ImageDataGenerator được sử dụng để tạo ra các bộ dữ liệu
# hình ảnh được tăng cường.
from keras.preprocessing.image import ImageDataGenerator

# bắt đầu xây dựng một mô hình mạng nơ-ron theo chuỗi.
model = Sequential()
# thêm một lớp Conv2D vào mô hình. Lớp này có 32 bộ lọc, kích thước bộ lọc là (3, 3) và sử dụng hàm kích hoạt relu.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
# một lớp MaxPooling2D vào mô hình. Lớp này sẽ giảm kích thước của dữ liệu hình ảnh bằng một nửa.
model.add(MaxPooling2D())
# thêm một lớp Conv2D khác vào mô hình. Lớp này cũng có 32 bộ lọc, kích thước bộ lọc là (3, 3) và
# sử dụng hàm kích hoạt relu.
model.add(Conv2D(32, (3, 3), activation='relu'))
# thêm một lớp MaxPooling2D khác vào mô hình. Lớp này sẽ giảm kích thước của dữ liệu hình ảnh bằng một nửa lần nữa.
model.add(MaxPooling2D())
# thêm một lớp Conv2D khác vào mô hình. Lớp này cũng có 32 bộ lọc, kích thước bộ lọc là (3, 3) và
# sử dụng hàm kích hoạt relu.
model.add(Conv2D(32, (3, 3), activation='relu'))
# thêm một lớp MaxPooling2D khác vào mô hình. Lớp này sẽ giảm kích thước của dữ liệu hình ảnh bằng một nửa lần nữa.
model.add(MaxPooling2D())
# thêm một lớp Flatten vào mô hình. Lớp này sẽ chuyển đổi dữ liệu hình ảnh thành một mảng một chiều.
model.add(Flatten())
# thêm một lớp Dense vào mô hình. Lớp này có 100 nơ-ron, sử dụng hàm kích hoạt relu.
model.add(Dense(100, activation='relu'))
# thêm một lớp Dense khác vào mô hình. Lớp này có 1 nơ-ron, sử dụng hàm kích hoạt sigmoid.
model.add(Dense(1, activation='sigmoid'))

'''
Biên dịch mô hình. Cấu hình huấn luyện bao gồm:

Optimizer: Adam là một thuật toán tối ưu hóa được sử dụng để cập nhật các trọng số của mô hình.
Loss: Binary crossentropy là một hàm mất mát được sử dụng cho các bài toán phân loại nhị phân.
Metrics: Accuracy là một chỉ số đánh giá hiệu suất của mô hình.
'''
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# tạo ra các bộ dữ liệu hình ảnh được tăng cường. Bộ dữ liệu tăng cường sẽ giúp mô hình học hỏi tốt hơn từ dữ liệu.
'''
Các tham số trong lớp ImageDataGenerator được sử dụng để tăng cường dữ liệu hình ảnh như sau:
rescale: Tham số này được sử dụng để chia tất cả các giá trị pixel của ảnh cho 255. 
Điều này sẽ giúp chuẩn hóa dữ liệu và làm cho mô hình dễ học hỏi hơn.
shear_range: Tham số này được sử dụng để thực hiện phép biến dạng shear trên dữ liệu hình ảnh. 
Phép biến dạng shear sẽ làm lệch hình ảnh theo một góc ngẫu nhiên trong khoảng từ -0,2 đến 0,2.
zoom_range: Tham số này được sử dụng để thực hiện phép biến dạng zoom trên dữ liệu hình ảnh. 
Phép biến dạng zoom sẽ phóng to hoặc thu nhỏ hình ảnh theo một tỉ lệ ngẫu nhiên trong khoảng từ 0,8 đến 1,2.
horizontal_flip: Tham số này được sử dụng để thực hiện thao tác lật ngang dữ liệu hình ảnh.
Bộ dữ liệu tăng cường được tạo ra bằng cách áp dụng các biến đổi ngẫu nhiên này cho dữ liệu hình ảnh gốc.
'''
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# tạo ra các bộ dữ liệu hình ảnh để huấn luyện và kiểm tra mô hình.
'''
Phương thức flow_from_directory() được sử dụng để tạo một trình phát dữ liệu (data generator) từ một thư mục chứa 
dữ liệu hình ảnh. Trình phát dữ liệu này sẽ trả về các ảnh và nhãn tương ứng của chúng theo từng batch.
Các tham số trong phương thức flow_from_directory() như sau:
directory: Thư mục chứa dữ liệu hình ảnh.
target_size: Kích thước của các ảnh sau khi được resize.
batch_size: Kích thước của mỗi batch.
class_mode: Chế độ phân loại. Có hai chế độ phân loại:
binary: Các ảnh được phân loại thành hai lớp.
categorical: Các ảnh được phân loại thành nhiều lớp.
'''
training_set = train_datagen.flow_from_directory(
    'train',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'test',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary')

# huấn luyện mô hình trong 10 epoch. Mỗi epoch, mô hình sẽ được huấn luyện trên tập dữ liệu
# đào tạo và sau đó được kiểm tra trên tập dữ liệu xác thực.
model_saved = model.fit(training_set, epochs=10, validation_data=test_set)

# lưu mô hình đã được huấn luyện.
model.save('mymodel.keras', model_saved)

# tải mô hình đã được huấn luyện và sử dụng nó để dự đoán nhãn của một ảnh cụ thể.
mymodel = load_model('mymodel.keras')
test_image = image.load_img(r'./test/with_mask/1-with-mask.jpg',
                            target_size=(150, 150, 3))
# test_image
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
var = mymodel.predict(test_image)[0][0]
print(var)

# Lấy lịch sử đào tạo từ mô hình
history = model.history

# Tạo danh sách độ chính xác đào tạo và xác thực
accuracy = []
val_accuracy = []

for epoch in range(len(history.history['accuracy'])):
    accuracy.append(history.history['accuracy'][epoch])
    val_accuracy.append(history.history['val_accuracy'][epoch])

# Vẽ biểu đồ
plt.plot(accuracy, label='Độ chính xác đào tạo')
plt.plot(val_accuracy, label='Độ chính xác xác thực')
plt.xlabel('Epoch')
plt.ylabel('Độ chính xác')
plt.legend()
plt.show()