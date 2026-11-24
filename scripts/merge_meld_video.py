import sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / 'digemo' / d) for d in ('encoders', 'model', 'pipeline')]
import re
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict

# ================= CONFIG =================

INPUT_DIR = Path(
    r"E:\Do_an\Dataprocessing\MELD.Raw\test\output_repeated_splits_test"
)

OUTPUT_DIR = INPUT_DIR / "merged_videos"
MASTER_NAME = "utterance_timestamps.csv"   # 1 file chung cho mọi dialogue
MASTER_HEADER = ["dialogue_id", "utterance_id", "original_video", "start", "end", "duration"]

# ==========================================


def run_ffmpeg(args):
    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr[-3000:])


def get_duration(path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path)
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return float(result.stdout.strip())


def get_dialogues():
    pattern = re.compile(
        r"dia(\d+)_utt(\d+)\.mp4$",
        re.IGNORECASE
    )

    dialogues = defaultdict(list)

    for video in INPUT_DIR.glob("*.mp4"):
        match = pattern.match(video.name)

        if match:
            dia_id = int(match.group(1))
            utt_id = int(match.group(2))
            dialogues[dia_id].append((utt_id, video))

    for dia_id in dialogues:
        dialogues[dia_id].sort(key=lambda x: x[0])

    return dialogues


def select_dialogues(dialogues):
    print("\n========== DIALOGUES AVAILABLE ==========")

    ids = sorted(dialogues.keys())

    print(f"Tổng số dialogue: {len(ids)}")
    print("Các dialogue:", ", ".join(map(str, ids)))

    print("\nNhập lựa chọn:")
    print("  21       -> Ghép dialogue 21")
    print("  21,22,25 -> Ghép nhiều dialogue")
    print("  all      -> Ghép toàn bộ")

    choice = input("\nChọn dialogue: ").strip().lower()

    if choice == "all":
        return ids

    try:
        selected = sorted(set(
            int(x.strip()) for x in choice.split(",")
        ))
    except ValueError:
        raise ValueError("Lựa chọn không hợp lệ.")

    missing = [x for x in selected if x not in dialogues]

    if missing:
        raise ValueError(f"Không tìm thấy dialogue: {missing}")

    return selected


def read_master():
    """Đọc file mốc thời gian chung -> {(dialogue_id, utterance_id): [các cột]}."""
    path = OUTPUT_DIR / MASTER_NAME
    data = {}
    if path.is_file():
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    key = (int(r["dialogue_id"]), int(r["utterance_id"]))
                except (KeyError, ValueError):
                    continue
                data[key] = [key[0], key[1], r.get("original_video", ""),
                             float(r["start"]), float(r["end"]), float(r["duration"])]
    return data


def write_master(data):
    """Ghi file chung (ghi file tạm rồi thay thế để không hỏng file khi đang ghi dở)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / MASTER_NAME
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(MASTER_HEADER)
        for key in sorted(data):
            writer.writerow(data[key])
    try:
        tmp.replace(path)
    except PermissionError:
        raise RuntimeError(f"Không ghi được {path}. Hãy đóng file này nếu đang mở bằng Excel rồi chạy lại.")


def update_master(dia_id, rows):
    """Thay toàn bộ dòng của dialogue này trong file chung bằng rows (giữ nguyên dialogue khác)."""
    data = read_master()
    for key in [k for k in data if k[0] == dia_id]:
        del data[key]
    for r in rows:
        data[(int(r[0]), int(r[1]))] = list(r)
    write_master(data)
    return len(data)


def import_legacy():
    """Gom các file diaX_timestamps.csv cũ (bản gộp trước đây) vào file chung, nếu dialogue đó chưa có."""
    data = read_master()
    have = {k[0] for k in data}
    added = []
    for p in sorted(OUTPUT_DIR.glob("dia*_timestamps.csv")):
        m = re.match(r"dia(\d+)_timestamps\.csv$", p.name, re.IGNORECASE)
        if not m or int(m.group(1)) in have:
            continue
        with open(p, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    data[(int(r["dialogue_id"]), int(r["utterance_id"]))] = [
                        int(r["dialogue_id"]), int(r["utterance_id"]), r.get("original_video", ""),
                        float(r["start"]), float(r["end"]), float(r["duration"])]
                except (KeyError, ValueError):
                    continue
        added.append(int(m.group(1)))
    if added:
        write_master(data)
        print(f"[OK] Đã gom mốc thời gian cũ của dialogue {added} vào {MASTER_NAME}")


def merge_dialogue(dia_id, utterances):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_video = OUTPUT_DIR / f"dia{dia_id}.mp4"

    print(f"\n========== MERGING DIA{dia_id} ==========")

    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)

        normalized = []
        current_time = 0.0
        timestamp_rows = []

        for index, (utt_id, video) in enumerate(utterances):
            temp_video = temp / f"utt_{index:04d}.mp4"

            print(f"[{index + 1}/{len(utterances)}] {video.name}")

            # Chuẩn hóa từng video để tránh lỗi khác codec,
            # độ phân giải hoặc FPS khi nối.
            run_ffmpeg([
                "ffmpeg", "-y", "-v", "error",
                "-i", str(video),
                "-map", "0:v:0",
                "-map", "0:a:0?",
                "-vf",
                "scale=1280:720:force_original_aspect_ratio=decrease,"
                "pad=1280:720:(ow-iw)/2:(oh-ih)/2,"
                "setsar=1,fps=24,format=yuv420p",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "20",
                "-c:a", "aac",
                "-ac", "2",
                "-ar", "44100",
                "-b:a", "192k",
                "-movflags", "+faststart",
                str(temp_video)
            ])

            duration = get_duration(temp_video)

            start = current_time
            end = start + duration
            current_time = end

            normalized.append(temp_video)

            timestamp_rows.append([
                dia_id,
                utt_id,
                video.name,
                round(start, 4),
                round(end, 4),
                round(duration, 4)
            ])

        # Tạo danh sách video để concat
        concat_file = temp / "concat.txt"

        with open(concat_file, "w", encoding="utf-8") as f:
            for path in normalized:
                safe_path = path.resolve().as_posix()
                f.write(f"file '{safe_path}'\n")

        # Ghép thành video dialogue
        run_ffmpeg([
            "ffmpeg", "-y", "-v", "error",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            # Mã hóa lại 1 lần thay vì "-c copy": mốc thời gian liền mạch, trình duyệt phát được cả hình lẫn tiếng.
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.0",
            "-r", "24", "-g", "24", "-fps_mode", "cfr",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
            "-af", "aresample=async=1:first_pts=0",
            "-movflags", "+faststart",
            str(output_video)
        ])

    # Ghi mốc thời gian vào file chung (chỉ thay dòng của dialogue này)
    total = update_master(dia_id, timestamp_rows)

    print(f"\n[OK] Video: {output_video}")
    print(f"[OK] Timestamp: {OUTPUT_DIR / MASTER_NAME} ({total} dòng, mọi dialogue)")
    print(f"[OK] Duration: {current_time:.2f}s")


def main():
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Không tìm thấy ffmpeg trong PATH.")

    if not shutil.which("ffprobe"):
        raise RuntimeError("Không tìm thấy ffprobe trong PATH.")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(INPUT_DIR)

    import_legacy()
    dialogues = get_dialogues()

    if not dialogues:
        print("Không tìm thấy file diaX_uttY.mp4")
        return

    selected = select_dialogues(dialogues)

    for dia_id in selected:
        merge_dialogue(dia_id, dialogues[dia_id])

    print("\n========== COMPLETED ==========")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()