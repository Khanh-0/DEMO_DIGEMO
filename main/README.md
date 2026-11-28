# Active Speaker Lock — demo cục bộ

Không cần CRG-3, BERT, DenseNet hay openSMILE. Đây là kiểm tra riêng khả năng khoanh mặt đang nói bằng Light-ASD (CVPR 2023) + YuNet, không dùng heuristic độ mở miệng.

## Windows (PowerShell)

Cần Python 3.11 hoặc 3.12, Git, FFmpeg có libx264/AAC trong PATH. Nếu chưa có Git/FFmpeg có thể cài bằng `winget install --id Git.Git -e` và `winget install --id Gyan.FFmpeg -e`, rồi mở terminal mới.

Mở terminal tại thư mục đã giải nén:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Nếu máy dùng Python 3.12, thay `py -3.11` bằng `py -3.12`. Nếu muốn dùng GPU NVIDIA, cài PyTorch CUDA phù hợp driver theo https://pytorch.org/get-started/locally/ trước khi cài requirements; chương trình tự chọn CUDA nếu có, nếu không dùng CPU. Kiểm tra `ffmpeg -version` và `git --version`.

Lần đầu chạy app sẽ clone Light-ASD và tải YuNet. Sau khi terminal báo server ở http://127.0.0.1:8000, mở URL đó, chọn video có audio và bấm Phân tích video. Không mở index.html trực tiếp vì nó cần API Python.

## Chọn bộ dữ liệu cảm xúc

Mặc định demo dùng IEMOCAP + CRG-3:

```powershell
$env:EMOTION_DATASET="iemocap"
```

Các đường dẫn IEMOCAP và checkpoint CRG-3 được lấy từ `emotion_runtime/config.py`.

Có thể chọn MELD để kiểm tra cấu hình/đường dẫn encoder:

```powershell
$env:EMOTION_DATASET="meld"
```

MELD dùng các artifact trong `checkpoints/meld/`, nhưng emotion runtime DiGemo-MELD cần adapter riêng; demo sẽ báo lỗi rõ thay vì dùng nhầm checkpoint CRG-3 IEMOCAP.

## Hành vi
- File video được lưu cục bộ trong `runs/` và chuyển thành MP4 640x360, 25 fps (giữ tỉ lệ, thêm viền nếu cần). Đây là bước tiền xử lý toàn bộ file, chưa phải streaming từ webcam.
- Backend chạy Light-ASD pretrained trên audio 16 kHz và face crops 112x112 theo từng block 12 frame (0,48s), tối đa 37 frame ngữ cảnh. Model gốc có GRU hai chiều nên có lookahead bên trong block, không causal tuyệt đối. Frontend nhận kết quả từng block, phát video khi dữ liệu đủ, dừng đệm nếu cần.
- Xanh: điểm ASD vượt ngưỡng; có thể có nhiều khung xanh. Không bắt buộc chọn một người nếu không ai đủ ngưỡng. Chỉnh threshold để kiểm tra. Crop hiển thị mặt có điểm cao nhất trong số những mặt vượt ngưỡng.
- Tracker IoU đơn giản, ID có thể đổi khi mất detection, khuất mặt hoặc cắt cảnh. Không có nhận dạng danh tính, diarization, voice separation, VAD riêng hoặc chống lẫn giọng ngoài khung hình. Không đảm bảo khóa chính xác trong mọi cảnh.
- Ngưỡng 0,65 dùng softmax của head, không phải score logit mặc định của demo upstream; đây là giá trị thử nghiệm, không phải xác suất đã calibration. Model/crop khác detector gốc nên kết quả có thể khác bài báo. Khuyến nghị thử clip 10–30 giây trước.
- Chưa benchmark tốc độ/độ chính xác. CPU có thể chậm đáng kể. Video gốc và bản chuẩn hóa lưu trên máy; xóa `runs/` sau khi dừng server nếu không cần. Không expose server lên Internet.

Nguồn: https://github.com/Junhua-Liao/Light-ASD và https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet . Model weights không đóng gói, được tải từ upstream lúc chạy.
