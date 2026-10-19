# DiGemo — Multimodal Emotion Recognition (MELD Demo)

Demo nhận diện cảm xúc trong hội thoại từ **3 phương thức: văn bản (Text), âm thanh (Audio) và hình ảnh (Visual)** bằng mô hình **DiGemo**, sử dụng checkpoint huấn luyện trên bộ dữ liệu **MELD**.

Dự án có hai cách chạy:

- **Web demo:** thao tác trong trình duyệt, luân phiên **Speaker A / Speaker B** (nếu sử dụng đúng bản web đã cấu hình hai speaker).
- **Terminal demo (`live_erc.py`):** thu âm trực tiếp từ microphone và hình ảnh từ webcam, sau đó dự đoán cảm xúc từng lượt nói.

> **Ngôn ngữ đầu vào:** tiếng Anh. Kết quả thuộc 7 nhãn cảm xúc: `neutral`, `surprise`, `fear`, `sadness`, `joy`, `disgust`, `anger`.
>
> **Quan trọng:** Repository **không kèm trọng số/checkpoint lớn**. Hãy tải các tệp cần thiết từ Google Drive theo mục [Tải checkpoint & pretrained assets](#3-tải-checkpoint--pretrained-assets).

## 1. Yêu cầu

- **Windows 10/11**, sử dụng **PowerShell**.
- **Python 3.9** (phù hợp với hướng dẫn triển khai demo hiện có).
- Camera và microphone hoạt động; cấp quyền sử dụng cho ứng dụng/trình duyệt.
- Kết nối Internet trong lần chạy đầu để tải backbone pretrained từ Hugging Face / torchvision nếu chưa có trong cache.
- **FFmpeg** nằm trong `PATH` nếu chạy bản **Web demo** (giải mã âm thanh nhận từ trình duyệt).
- **CPU** có thể chạy, nhưng suy luận có thể chậm; GPU NVIDIA giúp tăng tốc các encoder khi cấu hình PyTorch CUDA phù hợp.

## 2. Cấu trúc dự án

Ví dụ bố trí các tệp tại **thư mục gốc của repository**:

```text
DiGemo-Demo/
├── checkpoints/
│   └── best_model_MELD_260.pth       # TẢI TỪ GOOGLE DRIVE
├── digemo_web/                       # Giao diện web và backend
│   └── app.py                        # Kiểm tra tên/đường dẫn thực tế trong repo
├── MELD_pro/
│   └── features/
│       ├── roberta_meld_ft.pt        # TẢI TỪ GOOGLE DRIVE
│       ├── online_audio_transform.joblib    # TẢI TỪ GOOGLE DRIVE
│       └── online_visual_transform.joblib   # TẢI TỪ GOOGLE DRIVE
├── live_erc.py                       # Demo qua terminal (nếu sử dụng)
├── predict.py
├── model.py
├── module.py
├── utils.py
├── recover_feature_transforms.py    # Tiện ích khôi phục transforms (không cần chạy thường xuyên)
└── README.md
```

**Lưu ý khi đưa file lên GitHub:**

- Nếu file bạn tải lên đang mang tên `model(1).py`, hãy đổi thành **`model.py`** vì `predict.py` dùng `from model import DiGemo`.
- Ảnh cấu trúc thư mục hiện có là `digemo_web/`, còn tài liệu hướng dẫn cũ ghi `digemo_web_v2/app.py`. **Dùng tên thư mục và đường dẫn `app.py` đúng với phiên bản thực tế của bạn**; tệp `app.py` chưa được cung cấp để đối chiếu.
- Các file Python dùng chung (`predict.py`, `model.py`, `module.py`, `utils.py`) phải nằm ở vị trí mà Python import được, ví dụ cùng cấp với `live_erc.py`.

## 3. Tải checkpoint & pretrained assets

Vì dung lượng lớn hoặc cần giữ đúng bộ tham số đã huấn luyện, **checkpoint và các file feature transform không được đóng gói trực tiếp trong repository**.

[Google Drive:](https://drive.google.com/file/d/1KiGVb6jZPwFJaysvie8EHoAzn0hM15YJ/view?usp=drive_link)

Sau khi tải, giải nén hoặc chép từng tệp theo bảng bên dưới:

| Tên tệp | Nơi đặt trong repo | Công dụng |
| --- | --- | --- |
| `best_model_MELD_260.pth` | `checkpoints/best_model_MELD_260.pth` | Checkpoint chính của DiGemo trên MELD. |
| `roberta_meld_ft.pt` | `MELD_pro/features/roberta_meld_ft.pt` | Trọng số RoBERTa-large đã fine-tune để trích xuất đặc trưng văn bản. |
| `online_audio_transform.joblib` | `MELD_pro/features/online_audio_transform.joblib` | Biến đổi đặc trưng audio về 300 chiều, khớp với checkpoint. |
| `online_visual_transform.joblib` | `MELD_pro/features/online_visual_transform.joblib` | Biến đổi đặc trưng visual về 342 chiều, khớp với checkpoint. |

Hai tệp `.joblib` **phải thuộc cùng pipeline đặc trưng với checkpoint**, không thay thế tùy ý bằng PCA/Scaler mới. Các file dữ liệu thô `.npy`, video hoặc toàn bộ MELD không bắt buộc chỉ để chạy inference nếu đã có đúng bốn tệp bên trên.

Các backbone thường được tải tự động khi chạy `live_erc.py`, nếu chưa cache:

- Text: `roberta-large` (tokenizer/config; trọng số encoder fine-tune lấy từ `roberta_meld_ft.pt`).
- Audio: `facebook/wav2vec2-base-960h`.
- Visual: `DenseNet121` với weights `IMAGENET1K_V1`.
- Speech-to-text: `faster-whisper` với model `base.en` (có thể dùng `tiny.en` để nhẹ hơn).

### Kiểm tra đã đặt đủ file chưa

Mở PowerShell tại thư mục gốc và chạy:

```powershell
@(
  ".\checkpoints\best_model_MELD_260.pth",
  ".\MELD_pro\features\roberta_meld_ft.pt",
  ".\MELD_pro\features\online_audio_transform.joblib",
  ".\MELD_pro\features\online_visual_transform.joblib"
) | ForEach-Object { "$(Test-Path $_)  $_" }
```

Nếu dòng nào hiện `False`, hãy kiểm tra lại đường dẫn và tên file sau khi tải từ Drive.

## 4. Cài đặt môi trường

Chạy lệnh tại **thư mục gốc** của dự án.

### 4.1. Tạo môi trường ảo

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Nếu PowerShell chặn lệnh `Activate.ps1`, có thể chạy riêng trong phiên PowerShell hiện tại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 4.2. Cài PyTorch

**Chạy trên CPU:**

```powershell
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
```

**Chạy trên GPU NVIDIA:** cài cặp `torch` / `torchvision` tương thích với driver và CUDA trên máy, thay cho lệnh CPU phía trên. Tham khảo [PyTorch — Previous Versions](https://pytorch.org/get-started/previous-versions/) để chọn đúng lệnh cài.

### 4.3. Cài thư viện khác

Các lệnh sau là **bộ phiên bản tham khảo nhất quán cho demo**, không phải bản `pip freeze` được xác nhận từ máy train checkpoint:

```powershell
python -m pip install numpy==1.26.4 scipy==1.11.4 pandas==1.5.3 scikit-learn==1.6.1 joblib
python -m pip install transformers==4.41.2 Pillow opencv-python==4.10.0.84 faster-whisper sounddevice
python -m pip install fastapi uvicorn python-multipart
```

**Không cài nguyên trạng file `requirements(1).txt` đã gửi kèm:** tệp đó vừa chứa `torch==2.0.0` vừa hướng dẫn dùng PyTorch `2.1.2`, và lặp lại một số khai báo phụ thuộc. Khi chuẩn bị một file `requirements_demo_meld.txt` chính thức, hãy bỏ dòng `torch==2.0.0`, giữ cặp `torch`/`torchvision` nhất quán, và ưu tiên dùng phiên bản `scikit-learn` đã tạo các file `.joblib`.

**Kiểm tra FFmpeg (bắt buộc với bản web theo hướng dẫn gốc):**

```powershell
ffmpeg -version
```

Nếu chưa nhận diện được `ffmpeg`, hãy cài FFmpeg rồi thêm thư mục chứa `ffmpeg.exe` vào biến môi trường `PATH`.

## 5. Chạy Web demo

> Lệnh mẫu bên dưới dựa trên tài liệu triển khai **MELD V2 web** và cấu trúc thư mục trong ảnh (`digemo_web/`). **Chưa thể xác nhận chính xác entrypoint hoặc toàn bộ tham số** do chưa có mã nguồn `digemo_web/app.py` trong tệp đính kèm.

Nếu trong repo có file `digemo_web/app.py`, thử chạy:

```powershell
python .\digemo_web\app.py --checkpoint ".\checkpoints\best_model_MELD_260.pth" --features-dir ".\MELD_pro\features" --speaker-a-index 0 --speaker-b-index 1 --window 10 --host 127.0.0.1 --port 8001
```

Nếu đường dẫn khác, hãy sửa `digemo_web\app.py` thành vị trí thật của `app.py`; có thể tìm bằng:

```powershell
Get-ChildItem .\digemo_web -Filter app.py -Recurse
```

Khi backend khởi động thành công, mở trình duyệt trên **cùng máy**:

**http://127.0.0.1:8001**

Cấp quyền microphone/camera, chọn Speaker A/B, ghi từng lượt nói bằng **tiếng Anh** và xem cảm xúc dự đoán.

## 6. Chạy trực tiếp bằng terminal (`live_erc.py`)

Đây là cách chạy được đối chiếu từ mã nguồn `live_erc.py` đã cung cấp. Đảm bảo các file Python như sơ đồ ở mục 2 và checkpoint ở mục 3 đã sẵn sàng.

```powershell
python .\live_erc.py --checkpoint ".\checkpoints\best_model_MELD_260.pth" --features-dir ".\MELD_pro\features" --torch-device auto --window 10
```

**Cách dùng:**

- Nhấn **Enter** để bắt đầu ghi một lượt nói; nhấn **Space** trong cửa sổ webcam để kết thúc sớm (mặc định tối đa 7 giây).
- Sau khi ghi, chương trình chạy ASR, trích xuất text/audio/visual và hiển thị cảm xúc dự đoán cùng các xác suất nổi bật.
- Nhập `r` rồi Enter để xóa lịch sử hội thoại; nhập `q` rồi Enter để thoát.
- Bản terminal mặc định gán người nói vào `speaker-index 8` (nhóm `Other` của MELD), **khác với giao diện A/B của bản web**.

Nếu muốn nhập bản ghi lời nói thủ công (không chạy Whisper):

```powershell
python .\live_erc.py --checkpoint ".\checkpoints\best_model_MELD_260.pth" --features-dir ".\MELD_pro\features" --manual-transcript
```

Nếu microphone/camera không đúng thiết bị:

```powershell
python .\live_erc.py --list-mics
```

Sau đó thêm `--mic-index <SỐ>` và/hoặc `--camera-index <SỐ>` vào lệnh chạy. Có thể thêm `--torch-device cpu` khi máy không dùng được CUDA.

## 7. Lỗi thường gặp

| Hiện tượng | Cách kiểm tra |
| --- | --- |
| `FileNotFoundError: Missing ...` | Kiểm tra 4 tệp cần tải từ Drive ở mục 3 và đường dẫn `--features-dir`. |
| `No module named 'model'` | Đổi `model(1).py` thành `model.py` và đặt đúng vị trí Python có thể import. |
| `ModuleNotFoundError` | Đảm bảo đã kích hoạt `.venv` và cài đầy đủ thư viện. |
| `CUDA not available` | Chạy bằng `--torch-device cpu` (terminal) hoặc cài lại PyTorch CUDA phù hợp. |
| Sai số chiều đặc trưng hoặc PCA/Scaler | Kiểm tra checkpoint, `.joblib` và các encoder có thuộc cùng pipeline train không. |
| Không mở được webcam/microphone | Kiểm tra quyền thiết bị của Windows, đóng ứng dụng đang dùng camera, chọn đúng index. |
| Không nhận lệnh `ffmpeg` | Cài FFmpeg và cập nhật `PATH` trước khi chạy web. |
| `app.py` không tồn tại hoặc báo `unrecognized arguments` | Kiểm tra đúng backend web và tham số của phiên bản thực tế; mã nguồn web chưa được đối chiếu. |


