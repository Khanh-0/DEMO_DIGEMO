# MELD test: DiGEmo vs DM-CER (2610 câu chung)

| Model | ACC | W-F1 | Macro-F1 | neutral F1 | surprise F1 | fear F1 | sadness F1 | joy F1 | disgust F1 | anger F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| DiGEmo | 66.09 | 64.20 | 44.20 | 79.72 | 55.11 | 6.35 | 39.19 | 61.99 | 16.47 | 50.56 |
| DM-CER | 62.38 | 62.03 | 44.78 | 76.20 | 57.52 | 12.96 | 35.73 | 58.75 | 23.85 | 48.47 |

Cả hai đúng 1429 · chỉ DiGEmo đúng 296 · chỉ DM-CER đúng 199 · cả hai sai 686 · trùng dự đoán 73.7% · McNemar p = 0.0000

Phủ: pkl 2610 câu, DiGEmo 2610, DM-CER 2610, chung 2610. Nhãn thật khác file DM-CER: 0 câu.
