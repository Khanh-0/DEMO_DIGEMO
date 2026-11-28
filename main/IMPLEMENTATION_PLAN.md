# Realtime IEMOCAP Emotion Demo — Implementation Plan

## Mục tiêu

```text
Upload video
→ FFmpeg chuẩn hóa
→ YuNet face detection
→ IoU tracking / Track ID
→ Light-ASD active speaker detection
→ buffer audio + face crop theo Track ID
→ faster-whisper transcript
→ IEMOCAP encoders: Text 1024 / Audio 1582 / Visual 342
→ CRG-3 IEMOCAP
→ emotion theo từng Track ID
→ hiển thị trên giao diện
```

`main/app.py` vẫn là backend chính. Không thay thế `digemo/web/app.py` và không sửa pipeline MELD.

## Giai đoạn 0 — Kiểm tra môi trường và model

```text
Light-ASD:
  main/Light-ASD/weight/pretrain_AVA_CVPR.model
YuNet:
  main/face_detection_yunet_2023mar.onnx
Whisper:
  faster-whisper trong môi trường hiện tại
CRG-3 IEMOCAP:
  E:/Do_an/DEMO/demo_meld/CRG3/outputs/iemocap/best_test.pt
IEMOCAP Text:
  E:/Do_an/DEMO/demo_meld/checkpoints/iemocap/models/text_ft.pt
IEMOCAP norm:
  E:/Do_an/DEMO/demo_meld/checkpoints/iemocap/stats/norm.npz
DenseNet FER+:
  E:/Do_an/DEMO/demo_meld/checkpoints/shared/face_fer.pt
Runtime encoders:
  E:/Do_an/DEMO/demo_meld/digemo/encoders/
```

Tiêu chí:

- Không tải lại Light-ASD nếu checkpoint local đã có.
- Không tải lại Whisper nếu model local/cache đã có.
- Không copy checkpoint lớn sang folder `main`.
- Không ảnh hưởng MELD artifacts.

## Giai đoạn 1 — Cấu hình dùng chung

Tạo:

```text
emotion_runtime/config.py
```

Nội dung:

- Root project `demo_meld`.
- Đường dẫn CRG-3 checkpoint.
- Đường dẫn IEMOCAP checkpoints.
- Device `cuda/cpu`.
- Sample rate `16000`.
- FPS chuẩn hóa `25`.
- Block size `12 frames`.
- Emotion labels:
  ```python
  ["hap", "sad", "neu", "ang", "exc", "fru"]
  ```

Không hard-code đường dẫn model trong các module xử lý.

## Giai đoạn 2 — IEMOCAP Encoder Runtime

Tạo:

```text
emotion_runtime/iemocap_encoders.py
```

Tái sử dụng class trong:

```text
demo_meld/digemo/encoders/encoders.py
```

API:

```python
class IEMOCAPEncoders:
    def text(self, transcript) -> np.ndarray:   # (1024,)
        ...
    def audio(self, wav) -> np.ndarray:         # (1582,)
        ...
    def visual(self, face_crops) -> np.ndarray: # (342,)
        ...
```

Các encoder được load một lần khi app khởi động, không load lại theo block.

## Giai đoạn 3 — Buffer theo Track ID

Tạo:

```text
emotion_runtime/track_buffer.py
```

Mỗi track lưu:

```python
{
    "track_id": int,
    "audio_samples": ..., 
    "face_crops": [...],
    "start_frame": int,
    "last_frame": int,
    "text_history": [...],
    "emotion": None,
    "confidence": 0.0,
    "speaking": False,
}
```

Chức năng:

- Append audio/frame/crop cho đúng Track ID.
- Giới hạn buffer để tránh tốn RAM.
- Đóng utterance khi speaker đổi hoặc có pause.
- Giữ emotion cuối cùng của từng Track ID.

## Giai đoạn 4 — Whisper Runtime

Tạo:

```text
emotion_runtime/whisper_runtime.py
```

Dùng `faster-whisper` có sẵn. Có hai chế độ:

```text
partial: cập nhật nhanh cho giao diện
final: chạy khi utterance kết thúc, dùng cho CRG-3
```

API:

```python
transcribe(wav) -> str
```

Whisper không được block vòng tracking quá lâu; xử lý trong worker hoặc sau khi đóng block.

## Giai đoạn 5 — CRG-3 Runtime

Tạo:

```text
emotion_runtime/crg3_runtime.py
```

Nhiệm vụ:

1. Load `CRG3/outputs/iemocap/best_test.pt`.
2. Khôi phục `model_config`.
3. Khởi tạo CRG-3 IEMOCAP.
4. Nhận feature Text/Audio/Visual.
5. Duy trì lịch sử utterance causal.
6. Tạo batch đúng format CRG-3.
7. Trả xác suất 6 lớp.

Output:

```python
{
    "emotion_id": int,
    "emotion": str,
    "confidence": float,
    "probabilities": list[float],
}
```

Lưu ý: CRG-3 cần history theo thứ tự thời gian; không truyền ba vector rời mà bỏ qua graph/history.

## Giai đoạn 6 — Emotion Pipeline

Tạo:

```text
emotion_runtime/emotion_pipeline.py
```

Luồng:

```text
ASD score theo Track ID
→ active speaker
→ append buffer Track ID
→ đủ audio/context
→ Whisper transcript
→ Text/Audio/Visual encoder
→ CRG-3 prediction
→ cập nhật state Track ID
```

API:

```python
class EmotionPipeline:
    def update(self, active_tracks, audio_chunk, frame_index):
        return list_of_track_results
```

## Giai đoạn 7 — Mở rộng `main/app.py`

Giữ nguyên upload, FFmpeg, YuNet, IoU tracking, Light-ASD, playback và API job/status. Bổ sung:

- Khởi tạo `EmotionPipeline` một lần.
- Sau mỗi `flush()` của ASD, gọi pipeline cảm xúc.
- Thêm vào frame result:

```json
{
  "track_id": 0,
  "bbox": [120, 40, 280, 260],
  "asd_score": 0.91,
  "speaking": true,
  "text": "I do not understand",
  "emotion": "fru",
  "confidence": 0.86
}
```

Không phá API cũ; chỉ mở rộng response.

## Giai đoạn 8 — Giao diện

Sửa `index.html` để hiển thị:

```text
Track ID
Speaking / Listening
ASD score
Transcript
Emotion
Confidence
```

Màu đề xuất:

```text
Speaking: xanh lá
Neutral: xám
Happy/Excited: vàng
Sad: xanh dương
Angry/Frustrated: đỏ
```

## Giai đoạn 9 — Kiểm thử

1. Import toàn bộ module và compile không lỗi.
2. Kiểm tra encoder:
   ```text
   Text   → (1024,)
   Audio  → (1582,)
   Visual → (342,)
   ```
3. Upload video 10–30 giây; kiểm tra Track ID và ASD score.
4. Có audio thì Whisper trả transcript; không có audio thì pipeline không crash.
5. CRG-3 load thành công, prediction thuộc 6 lớp, history tăng theo thời gian.
6. Test end-to-end:
   ```text
   Upload → tracking → ASD → Whisper → encoders → CRG-3 → UI
   ```

## Giai đoạn 10 — Tối ưu hiệu năng

- Load model một lần khi start app.
- Dùng GPU cho Light-ASD, DenseNet và CRG-3 nếu có.
- Batch face crops trước khi chạy DenseNet.
- Không chạy emotion inference ở mọi frame.
- Chạy inference mỗi `0.5–1.5 giây`.
- Giới hạn buffer theo Track ID.
- Không chạy nhiều job nặng đồng thời nếu thiếu VRAM.
- Whisper `small/base` cho realtime thử nghiệm; `large-v3` khi cần chất lượng cao.

## Kết quả mong muốn

```text
a/
├── app.py
├── index.html
├── face_detection_yunet_2023mar.onnx
├── Light-ASD/
└── emotion_runtime/
    ├── __init__.py
    ├── config.py
    ├── track_buffer.py
    ├── whisper_runtime.py
    ├── iemocap_encoders.py
    ├── crg3_runtime.py
    └── emotion_pipeline.py
```

Demo cuối:

```text
Video upload
→ bounding box theo Track ID
→ active speaker bằng Light-ASD
→ transcript bằng Whisper
→ feature IEMOCAP 1024/1582/342
→ cảm xúc CRG-3 theo từng Track ID
```
