# Hướng dẫn cài đặt và chạy DiGemo MELD V2 Web Demo

Tài liệu này dành cho người nhận file `DiGemo_MELD_V2_Demo_Share.zip` để cài đặt và chạy bản demo **DiGemo MELD V2 (hai speaker A/B)** trên Windows bằng PowerShell.

## 1. Yêu cầu và lưu ý trước khi cài

- **Python 3.9** (theo lệnh tạo môi trường bên dưới).
- **PyTorch** và **torchvision** được cài riêng để chọn đúng bản **CPU** hoặc **CUDA** tùy phần cứng.
- Cần cài **FFmpeg** và bảo đảm `ffmpeg.exe` có trong biến môi trường `PATH`, vì bản MELD dùng FFmpeg để giải mã audio từ trình duyệt.
- Giải nén đầy đủ `DiGemo_MELD_V2_Demo_Share.zip` trước khi làm theo các bước.
- Máy nhận có thể cần kết nối Internet ở lần chạy đầu để tải các backbone pretrained chưa được cache.

## 2. Tạo môi trường Python

Mở **PowerShell** ngay tại thư mục demo đã giải nén, sau đó chạy:

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2.1. Cài PyTorch và torchvision

Nếu chỉ sử dụng **CPU**:

```powershell
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
```

Nếu máy có **GPU NVIDIA**, thay lệnh CPU trên bằng cách cài bản PyTorch CUDA tương thích với GPU, driver và môi trường của máy.

### 2.2. Cài các thư viện còn lại

Đảm bảo file `requirements_demo_meld.txt` nằm trong thư mục demo, rồi chạy:

```powershell
python -m pip install -r requirements_demo_meld.txt
```

### 2.3. Kiểm tra FFmpeg

```powershell
ffmpeg -version
```

Nếu PowerShell báo không nhận diện lệnh `ffmpeg`, cần cài FFmpeg và thêm thư mục chứa `ffmpeg.exe` vào `PATH` trước khi chạy demo.

## 3. Khởi động MELD V2

Từ thư mục gốc của demo, chạy:

```powershell
python .\digemo_web_v2\app.py --checkpoint ".\checkpoints\best_model_MELD_260.pth" --features-dir ".\MELD_pro\features" --speaker-a-index 0 --speaker-b-index 1 --window 10 --host 127.0.0.1 --port 8001
```

Khi backend khởi động thành công, mở trình duyệt trên **chính máy đang chạy server** và truy cập:

**http://127.0.0.1:8001**

Sử dụng giao diện để chọn Speaker A/B, cấp quyền camera và microphone, ghi âm từng lượt và xem kết quả nhận diện cảm xúc.

## 4. Kiểm tra phiên bản thư viện

Để xem các phiên bản đã được cài trong môi trường Python hiện tại:

```powershell
python -m pip show torch torchvision transformers scikit-learn numpy
```

**Lưu ý về tính tương thích:** File `requirements_demo_meld.txt` chưa khóa tuyệt đối các phiên bản. Đặc biệt, các bộ **Scaler/PCA** được lưu bằng `joblib` nên dùng cùng phiên bản `scikit-learn` với môi trường tạo các file đó sẽ an toàn hơn.

Để bản ZIP chạy giống máy gốc nhất, nên đối chiếu và sử dụng các phiên bản thư viện đã hoạt động thành công trên máy đóng gói, thay vì để pip tự chọn phiên bản mới.

## 5. Tóm tắt lệnh chạy

Sau khi đã cài đặt môi trường và thư viện, những lần sử dụng tiếp theo chỉ cần:

```powershell
.\.venv\Scripts\Activate.ps1
python .\digemo_web_v2\app.py --checkpoint ".\checkpoints\best_model_MELD_260.pth" --features-dir ".\MELD_pro\features" --speaker-a-index 0 --speaker-b-index 1 --window 10 --host 127.0.0.1 --port 8001
```

Mở **http://127.0.0.1:8001** để sử dụng demo.
