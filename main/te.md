Mình đã viết một demo **Active Speaker Lock** riêng, chưa tích hợp CRG-3 hay ba encoder của bạn. Mục tiêu duy nhất là kiểm tra khả năng xác định và bám theo khuôn mặt người đang nói.

**speaker_lock_demo.zip**

HTML + Python backend + Requirements + README

Tải xuống toàn bộ source code

## **1. Các chức năng đã viết**

- Upload video từ máy tính.
- Tự động phát hiện và tracking khuôn mặt.
- Sử dụng **Light-ASD pretrained** để xác định người đang nói dựa trên cả audio và visual.
- Bounding box màu xanh bám theo người đang nói.
- Hiển thị Track ID và điểm ASD.
- Hiển thị khuôn mặt đã crop và cho phép lưu ảnh.
- Điều chỉnh ngưỡng nhận diện ngay trên giao diện.
- Trả kết quả theo từng đoạn video ngắn.

Không sử dụng phương pháp chọn miệng chuyển động nhiều nhất.

## **2. Giao diện dự kiến**

# **Active Speaker Lock**

Light-ASD + YuNet

conversation.mp4

Phân tích video

00:12 / 00:35

Active speaker crop

## **TRACK 0**

ASD Score: 0.94

Speaking detected

Minh họa giao diện, không phải kết quả inference thực tế.

## **3. Hướng dẫn chạy trên Windows**

### **Bước 1: Giải nén**

Cấu trúc thư mục:

```
speaker_lock_demo/
├── app.py
├── index.html
├── requirements.txt
└── README.md
```

### **Bước 2: Cài Git và FFmpeg (nếu chưa có)**

Mở PowerShell:

```
winget install --id Git.Git -e
winget install --id Gyan.FFmpeg -e
```

### **Bước 3: Tạo môi trường Python**

Mở terminal tại thư mục vừa giải nén. Khuyến nghị Python 3.11 hoặc 3.12.

```
py -3.11 -m venv .venv
```

Cài thư viện:

```
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### **Bước 4: Khởi động backend**

```
.\.venv\Scripts\python.exe app.py
```

Lần đầu chạy, chương trình sẽ tự tải source code Light-ASD, pretrained weights đi kèm repository và model phát hiện khuôn mặt YuNet.

Nếu PyTorch nhận được CUDA, chương trình sẽ tự động sử dụng GPU. Nếu không, nó sẽ chạy CPU.

### **Bước 5: Mở trình duyệt**

Truy cập:

[**http://127.0.0.1:8000**](http://127.0.0.1:8000?utm_source=chatgpt.com)

Sau đó chọn video và bấm **Phân tích video**.

Không mở trực tiếp `index.html` vì giao diện cần kết nối với backend Python để chạy ASD.

## **4. Pipeline bên trong**

**Upload video**

FFmpeg · 25 FPS · Audio 16 kHz

YuNet Face Detection

IoU Face Tracking

**Light-ASD**

Face sequence + Audio MFCCInference mỗi block 12 frame (480 ms)

Temporal Score Smoothing

Active Speaker Bounding Box

HTML Canvas + Face Crop

Model sử dụng từ repository chính thức [Light-ASD — CVPR 2023](https://github.com/Junhua-Liao/Light-ASD?utm_source=chatgpt.com). 

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



## **5. Cấu hình mặc định**

| Tham số             | Giá trị      |
| ------------------- | ------------ |
| Video FPS           | 25           |
| Video resolution    | 640 × 360    |
| Audio sample rate   | 16.000 Hz    |
| ASD context tối đa  | 1,48 giây    |
| ASD inference hop   | 480 ms       |
| ASD threshold       | 0.65         |
| Face crop cho ASD   | 112 × 112    |
| GPU                 | Tự động      |
| Emotion recognition | Chưa sử dụng |

## **6. Những điểm cần lưu ý khi test**

Mình đã kiểm tra cú pháp Python, JavaScript và tính toàn vẹn của file ZIP, nhưng **chưa chạy inference end-to-end trên video thực tế** do môi trường hiện tại chưa có đầy đủ dependencies và pretrained weights.

Phiên bản này chuẩn hóa video trước rồi mới phân tích tăng dần từng block. Video sẽ tự tạm dừng nếu backend chưa xử lý kịp. Đây là bản thử nghiệm near-realtime, chưa phải streaming causal hoàn toàn.

Face Tracking hiện cũng chỉ dùng IoU đơn giản, chưa có ArcFace để duy trì ID khi khuôn mặt biến mất rồi xuất hiện trở lại.

Bạn nên thử trước một video khoảng 10–30 giây có hai người đối thoại. Quan sát xem khi A đang nói thì khung xanh có bám đúng A không, và khi B bắt đầu nói thì khung có chuyển sang B không.

**Cứ kiểm tra thật tốt phần Active Speaker Detection này trước.** Nếu khóa mặt đúng người nói ổn định thì sau đó mới xây dựng module cắt utterance, lưu audio và face crops để kết nối với CRG-3.