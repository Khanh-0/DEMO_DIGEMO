# Local encoder bundle

This folder contains the encoder implementation and its runtime assets used by the live upload pipeline.

- `code/`: copied encoder modules from the original `digemo/encoders/` package.
- `checkpoints/iemocap/`: IEMOCAP text checkpoint and normalization statistics.
- `checkpoints/meld/`: MELD text/audio encoder assets and normalization statistics.
- `checkpoints/shared/face_fer.pt`: shared visual FaceEncoder checkpoint.

The live runtime resolves all paths from `a/emotion_runtime/config.py`; it does not depend on `digemo/encoders` for loading.
The emotion-model checkpoints (CRG-3, DiGemo, DMCER) remain separate from this encoder bundle.
