# DiGemo — Nhận diện cảm xúc hội thoại đa phương thức (MELD Demo)

Demo nhận diện cảm xúc từng câu thoại trong video hội thoại bằng mô hình **DiGemo**, kết hợp **3 phương thức: văn bản (Text), âm thanh (Audio) và hình ảnh (Visual)**. Checkpoint được huấn luyện trên bộ dữ liệu **MELD**.

- **Ngôn ngữ đầu vào:** tiếng Anh.
- **7 nhãn cảm xúc:** `neutral`, `surprise`, `fear`, `sadness`, `joy`, `disgust`, `anger`.

## 1. Tính năng

Giao diện web **DiGemo Emotion Studio** (`/video`) xử lý cả đoạn hội thoại trong một lần:

1. Tải video lên (mp4, avi, mov, mkv, webm, m4v).
2. Hệ thống tách âm thanh (FFmpeg), nhận dạng lời nói (Whisper), phân biệt người nói (pyannote) và gộp thành các câu thoại.
3. Bạn kiểm tra và chỉnh sửa bảng câu thoại (transcript, người nói, thời gian).
4. Với mỗi câu, hệ thống trích đặc trưng Text (RoBERTa), Audio (Wav2Vec2) và Visual (DenseNet FER+ trên khuôn mặt người đang nói, xác định bằng chuyển động môi qua MediaPipe).
5. DiGemo dự đoán cảm xúc cho toàn bộ hội thoại, hiển thị xác suất từng nhãn và cho tải kết quả CSV.

Ngoài chế độ tự động, còn hai chế độ nhập bảng câu thoại:

- **Chế độ MELD:** chọn `Dialogue_ID` trong `test_sent_emo.csv` để lấy transcript, người nói và nhãn gốc (có tính độ chính xác).
- **Tải CSV riêng:** CSV có cột `StartTime`, `EndTime` và tùy chọn `Speaker`, `Utterance`, `Dialogue_ID`, `Emotion`.

**Giới hạn mặc định:** video tối đa 800 MB và 10 phút, tối đa 300 câu thoại mỗi lần.

## 2. Cấu trúc dự án

```text
demo_meld/
├── README.md
├── requirements.txt
├── start_web.ps1                  # chạy nhanh web
├── checkpoints/                   # trọng số (không đưa lên git)
│   ├── digemo_meld.pth            # DiGemo (MELD)
│   ├── text_ft.pt                 # RoBERTa-base đã fine-tune
│   └── face_fer.pt                # DenseNet FER+
├── data/
│   ├── utterance_timestamps.csv   # mốc thời gian video MELD đã gộp
│   ├── csv/                       # train/dev/test_sent_emo.csv
│   └── feats/                     # đặc trưng + chuẩn hóa (*.pkl, *.npz, *.json)
├── digemo/
│   ├── model/                     # model.py, module.py, utils.py
│   ├── encoders/                  # encoders.py, encoders_online.py, common.py, densenet_fer.py
│   ├── pipeline/                  # video_pipeline.py, diarizer.py, overlap_*.py,
│   │                              # predict.py, infer.py, live_erc.py
│   └── web/
│       ├── app.py                 # entrypoint FastAPI
│       └── static/                # video.html, video.js, css/
├── scripts/                       # chạy một lần để chuẩn bị dữ liệu/huấn luyện
│   ├── build_features.py
│   ├── extract_face_embed.py
│   ├── text_finetune.py
│   └── merge_meld_video.py
└── docs/
    └── digemo_flow.md             # mô tả luồng xử lý
```

`app.py` tự thêm `digemo/model`, `digemo/encoders`, `digemo/pipeline` vào `sys.path`, nên các module import với nhau theo tên phẳng (`from predict import ...`). Không cần `pip install -e .`.

## 3. Yêu cầu

- **Windows 10/11**, PowerShell.
- **Python 3.10** (môi trường đã chạy thử; 3.9 cũng có thể dùng nếu cài đủ thư viện).
- **FFmpeg** nằm trong `PATH` (kiểm tra bằng `ffmpeg -version`).
- GPU NVIDIA giúp tăng tốc rõ rệt (Whisper, Wav2Vec2, pyannote); CPU vẫn chạy nhưng chậm.
- Internet ở lần chạy đầu để tải model từ Hugging Face (Whisper `large-v3`, `facebook/wav2vec2-base`, `roberta-base`, pyannote, speechbrain) nếu chưa có trong cache.

## 4. Tải checkpoint và dữ liệu

Repository không kèm trọng số lớn. Tải từ Google Drive: [liên kết tải](https://drive.google.com/file/d/1KiGVb6jZPwFJaysvie8EHoAzn0hM15YJ/view?usp=drive_link) và đặt đúng chỗ:

| Tệp | Vị trí | Công dụng |
| --- | --- | --- |
| `digemo_meld.pth` | `checkpoints/` | Checkpoint chính DiGemo (MELD, đặc trưng 1024/342/300, 9 speaker slots). |
| `text_ft.pt` | `checkpoints/` | RoBERTa-base đã fine-tune cho đặc trưng văn bản. |
| `face_fer.pt` | `checkpoints/` | DenseNet FER+ cho đặc trưng khuôn mặt. |
| `audio_head.npz`, `norm.npz`, `speaker_map.json` | `data/feats/` | Đầu chiếu audio 768→300, chuẩn hóa visual và ánh xạ người nói. **Bắt buộc**, phải cùng pipeline với checkpoint. |
| `test_sent_emo.csv` | `data/csv/` | Dùng cho chế độ MELD. |
| `utterance_timestamps.csv` | `data/` | Mốc thời gian thật của video MELD đã gộp (tạo bởi `scripts/merge_meld_video.py`). |

Các tệp `*.pkl` trong `data/feats/` chỉ cần khi huấn luyện lại hoặc dùng `scripts/`; chạy web chỉ cần các tệp bắt buộc ở trên.

Kiểm tra đã đủ tệp chưa (tại thư mục gốc):

```powershell
@(
  ".\checkpoints\digemo_meld.pth",
  ".\checkpoints\text_ft.pt",
  ".\checkpoints\face_fer.pt",
  ".\data\feats\audio_head.npz",
  ".\data\feats\norm.npz",
  ".\data\feats\speaker_map.json"
) | ForEach-Object { "$(Test-Path $_)  $_" }
```

## 5. Cài đặt môi trường

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Nếu PowerShell chặn `Activate.ps1`:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Cài PyTorch (chọn một):

```powershell
# CPU
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# GPU: chọn lệnh phù hợp CUDA tại https://pytorch.org/get-started/locally/
```

Cài các thư viện còn lại:

```powershell
python -m pip install -r requirements.txt
```

Các thư viện chính mà web sử dụng: `fastapi`, `uvicorn`, `python-multipart`, `transformers`, `faster-whisper`, `pyannote.audio`, `speechbrain`, `mediapipe`, `opencv-python`, `scikit-learn`, `numpy`, `Pillow`.

> Phiên bản cụ thể nên lấy từ `pip freeze` của môi trường đã chạy được, để cập nhật vào `requirements.txt`.

## 6. Chạy web

Tại thư mục gốc, sau khi kích hoạt `.venv`:

```powershell
python digemo\web\app.py --host 127.0.0.1 --port 8000
```

Hoặc dùng `.\start_web.ps1`. Chờ thông báo `[web] Ready!`, rồi mở **http://127.0.0.1:8000** (tự chuyển tới `/video`).

Tất cả tham số đều có giá trị mặc định theo cấu trúc trên:

| Tham số | Mặc định |
| --- | --- |
| `--checkpoint` | `checkpoints/digemo_meld.pth` |
| `--encoder-dir` | `digemo/encoders` |
| `--feat-dir` | `data/feats` |
| `--text-ckpt` | `checkpoints/text_ft.pt` |
| `--face-ckpt` | `checkpoints/face_fer.pt` |
| `--meld-csv` | `data/csv/test_sent_emo.csv` |
| `--meld-timestamps` | `data/utterance_timestamps.csv` |
| `--torch-device` | `auto` (`cuda` nếu có, không thì `cpu`) |
| `--whisper-model` | `large-v3` (đổi sang `medium`/`small` nếu thiếu VRAM) |
| `--host`, `--port` | `127.0.0.1`, `8000` |

### Biến môi trường tùy chọn


| Biến | Tác dụng |
| --- | --- |
| `DIGEMO_MELD_CSV`, `DIGEMO_MELD_TS` | Ghi đè đường dẫn CSV MELD và file mốc thời gian (thường do `--meld-csv`, `--meld-timestamps` tự đặt). |

### Truy cập từ máy khác trong mạng

```powershell
python digemo\web\app.py --host 0.0.0.0
```

Server **không có đăng nhập**, chỉ nên dùng trong mạng tin cậy.

## 7. Sử dụng

1. Mở `/video`, chọn video (hoặc chọn một `Dialogue_ID` MELD cùng video tương ứng).
2. Chọn số người nói nếu để hệ thống tự phân biệt.
3. Bấm phân tích, chờ bảng câu thoại xuất hiện. Kiểm tra bằng nút phát, sửa transcript, người nói hoặc thời gian nếu cần. Các dòng có dấu ⚠ là vùng nói chồng, độ tin cậy thấp hơn.
4. Gán slot người nói (Joey, Ross,... theo `speaker_map.json`; người lạ vào nhóm `Other`).
5. Bấm dự đoán để xem cảm xúc, xác suất từng nhãn, thống kê và tải CSV kết quả.

## 8. Chuẩn bị dữ liệu / huấn luyện lại (tùy chọn)

Các script trong `scripts/` chỉ dùng khi tạo lại đặc trưng hoặc fine-tune encoder. Xem tham số bằng `--help`, ví dụ:

```powershell
python scripts\build_features.py --help
```

| Script | Việc làm |
| --- | --- |
| `extract_face_embed.py` | Trích embedding khuôn mặt (342 chiều). |
| `build_features.py` | Dựng bộ đặc trưng text/audio/visual vào `data/feats/`. |
| `text_finetune.py` | Fine-tune RoBERTa cho đặc trưng văn bản. |
| `merge_meld_video.py` | Gộp video MELD theo dialogue và tạo `utterance_timestamps.csv`. |

`digemo/pipeline/live_erc.py` là bản demo thu âm/webcam trực tiếp từ giai đoạn trước; nó chưa được cập nhật và chưa kiểm chứng với cấu trúc mới.

## 9. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
| --- | --- |
| `Thiếu file bắt buộc: ...` khi khởi động | Kiểm tra các tệp ở mục 4 và đường dẫn tham số. |
| `ModuleNotFoundError` | Kích hoạt `.venv` và cài lại `requirements.txt`; chạy từ thư mục gốc bằng `python digemo\web\app.py`. |
| `Không tìm thấy test_sent_emo.csv` / thiếu mốc thời gian | Đặt file vào `data/csv/` và `data/`, hoặc dùng `--meld-csv`, `--meld-timestamps`. |
| `Không tìm thấy ffmpeg` | Cài FFmpeg và thêm vào `PATH`. |
| `CUDA not available` | Chạy với `--torch-device cpu` hoặc cài lại PyTorch bản CUDA. |
| Hết VRAM khi load Whisper | Dùng `--whisper-model medium` hoặc `small`. App tự lùi về CPU nếu GPU không dùng được. |
| Cảnh báo `torchcodec ... WinError 127` | Vô hại, pipeline truyền audio trong bộ nhớ. Có thể bỏ qua hoặc `pip uninstall torchcodec`. |
| Cảnh báo pyannote/độ chính xác người nói thấp | pyannote không chạy được nên đã lùi về gom cụm Wav2Vec2; kiểm tra cài đặt/quyền tải model pyannote và cột Speaker. |
| Video "không có luồng hình/âm thanh" | Video cần có cả hình và tiếng, dài tối đa 10 phút. |
| Sai số chiều đặc trưng | Checkpoint, `norm.npz`, `audio_head.npz` và các encoder phải cùng pipeline huấn luyện (1024 / 342 / 300). |