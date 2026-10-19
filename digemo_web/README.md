# DiGemo Emotion Studio — Web UI chạy trên Windows

Ứng dụng web thật, sử dụng chính `live_erc.py`, `predict.py`, `recover_feature_transforms.py` và checkpoint **DiGemo MELD** bạn đã train. Không sử dụng dữ liệu `.pkl` để dự đoán. Không có mock prediction.

## 1. Cấu trúc dự kiến

```text
E:\Do_an\DiGemo\
├── live_erc.py
├── predict.py
├── model.py
├── module.py
├── recover_feature_transforms.py
├── checkpoints\best_model_MELD_260.pth
├── MELD_pro\features\
│   ├── roberta_large_meld_ft.pt (hoặc roberta_meld_ft.pt)
│   ├── online_audio_transform.joblib
│   └── online_visual_transform.joblib
└── digemo_web\
    ├── app.py
    ├── static\index.html
    ├── static\styles.css
    ├── static\app.js
    ├── requirements-web.txt
    └── start_web.ps1
```

Giải nén ZIP sao cho thư mục `digemo_web` nằm **trực tiếp dưới** `E:\Do_an\DiGemo`.

## 2. Cài dependencies (PowerShell, .venv đã kích hoạt)

```powershell
cd E:\Do_an\DiGemo
python -m pip install -r .\digemo_web\requirements-web.txt
ffmpeg -version
```

Nếu lệnh `ffmpeg -version` không nhận diện: cài FFmpeg CLI, thêm thư mục chứa `ffmpeg.exe` vào `PATH`, rồi mở PowerShell mới. FFmpeg được dùng để chuyển audio `.webm` từ Firefox/Chrome thành mono 16 kHz; không lưu file ghi âm xuống ổ cứng.

Các thư viện RoBERTa, Wav2Vec2, DenseNet, PyTorch, OpenCV, scikit-learn và joblib đã có sẵn nếu `live_erc.py` trước đó chạy được.

## 3. Start web server

Chạy **một dòng** ở thư mục DiGemo:

```powershell
python .\digemo_web\app.py --checkpoint checkpoints/best_model_MELD_260.pth --features-dir MELD_pro/features --host 127.0.0.1 --port 8000
```

Khi PowerShell in `[web] Ready!`, mở:

**http://127.0.0.1:8000**

Hoặc chạy `powershell -ExecutionPolicy Bypass -File .\digemo_web\start_web.ps1`.

Lần đầu Whisper có thể cần tải `base.en`; các encoder load **một lần** khi khởi động server, không load lại ở mỗi lượt ghi.

## 4. Hướng dẫn sử dụng

1. Bấm **Bật thiết bị** → chọn **Allow** cho Camera & Microphone trong trình duyệt.
2. Chọn camera/mic từ 2 dropdown khi cần.
3. Bấm **Bắt đầu nói**, nói tiếng Anh khoảng 1–6 giây rồi bấm **Kết thúc lượt**; hoặc tối đa 12 giây tự dừng.
4. Đợi Whisper transcribe, encoder T/A/V trích xuất, DiGemo dự đoán. Trang hiển thị 7 xác suất, emotion, transcript, thời gian từng bước và lịch sử lượt.
5. Bấm **Reset dialogue** để làm mới bộ nhớ ngữ cảnh DiGemo (10 utterance gần nhất).
6. Có thể bật **Nhập transcript thủ công** khi Whisper nghe sai; yêu cầu điền transcript trước khi ghi.

Mẹo: với **lượt nói khác cảm xúc nhưng cùng một transcript**, reset mỗi lần để tránh history làm nhiễu bài kiểm tra. Dùng nút Space khi focus không ở textbox/button để start/stop.

## 5. Pipeline thật

- `navigator.mediaDevices.getUserMedia`: xin quyền webcam và microphone trong trình duyệt.
- `MediaRecorder`: lưu tạm audio ở RAM của tab; frontend lấy 8 JPEG frames phân bố trên đoạn quay. Không gửi video liên tục.
- `POST /api/predict`: gửi audio `.webm`/`.ogg` và 8 JPEG frames qua `multipart/form-data` đến FastAPI chạy trên **cùng PC**.
- `ffmpeg`: decode audio thành mono 16,000 samples/s float32; OpenCV decode JPEG thành BGR frames.
- `faster-whisper base.en`: ASR tiếng Anh (mặc định CPU int8), hoặc dùng transcript bạn nhập.
- RoBERTa-large fine-tuned MELD → `(1024,)` text.
- Wav2Vec2-base → `(768,)` pooled, áp dụng `online_audio_transform.joblib` → `(300,)` audio.
- DenseNet121 ImageNet → `(1024,)` averaged frames, áp dụng `online_visual_transform.joblib` → `(342,)` visual.
- Đưa prefix/cửa sổ 10 utterance thành `(L,1,1024)`, `(L,1,342)`, `(L,1,300)`; `qmask=(L,1,9)` (speaker Other = 8), `umask=(L,1)` vào checkpoint MELD.
- Softmax nhãn cuối (7 cảm xúc). Chỉ giữ feature đã trích xuất (không lưu raw audio/video) trong session RAM trên server; reset hoặc restart xóa history.

## 6. Giới hạn và bảo mật

- Bản này là **online theo từng lượt nói**, chưa phải frame-by-frame streaming / VAD tự động.
- Mô hình dùng MELD (tiếng Anh), chưa kiểm chứng đầy đủ với dữ liệu đời thực/tiếng Việt.
- Speaker slot `Other=8` là giải pháp thực nghiệm, không phải speaker cá nhân được training riêng.
- Chỉ dùng cho chạy local qua `127.0.0.1` theo mặc định. Nếu mở `--host 0.0.0.0`, trang qua IP LAN HTTP thường **không được cấp quyền microphone/webcam** (trừ localhost), cần HTTPS đáng tin cậy; đồng thời hạn chế truy cập qua mạng không tin cậy vì demo chưa có user authentication.
- Nhãn và confidence là đầu ra mô hình, không phải chẩn đoán cảm xúc/tâm lý.
- Audio/video **không ghi xuống đĩa**. Transcript và các vector embedding lưu trong RAM theo phiên; xóa khi reset history/hết hạn phiên (~3 giờ)/restart server.
- `ffmpeg` CLI và `base.en` phải sẵn có. Bản IEMOCAP cần audio extractor khác (Wav2Vec2-large và 8192 → 1582) nên chưa được hỗ trợ trong UI MELD này.

## 7. Khắc phục lỗi

- `ModuleNotFoundError: fastapi`: `python -m pip install -r .\digemo_web\requirements-web.txt`.
- `Missing roberta_meld_ft.pt`: chương trình tự tìm `roberta_meld_ft.pt` hoặc `roberta_large_meld_ft.pt` trong `MELD_pro\features`; nếu khác, chỉ định `--roberta-checkpoint`.
- `Cannot find ffmpeg`: chạy `ffmpeg -version`, kiểm tra PATH.
- Camera/microphone không cho phép: truy cập chính xác `http://127.0.0.1:8000`, cấp quyền trình duyệt, đóng app khác đang sử dụng webcam.
- Nếu API báo `Microphone gần như không có tiếng`: tăng gain hoặc nói gần mic.
- Nếu máy chậm: thử `--whisper-model tiny.en` (ASR có thể kém chính xác hơn) hoặc `--torch-device cuda` khi CUDA hoạt động.
- Cổng 8000 bị dùng: `--port 8001` và mở `http://127.0.0.1:8001`.
