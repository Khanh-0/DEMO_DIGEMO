"""Tab "So sánh model": DiGEmo vs DM-CER trên tập test MELD / IEMOCAP.

Logic nằm ở scripts/compare_models.py (dùng chung với dòng lệnh); file này chỉ thêm route:
  GET  /compare                         trang so sánh
  GET  /api/compare/config              đường dẫn đang dùng + file nào đang thiếu
  POST /api/compare/run                 {"dataset": "meld"|"iemocap", "run_dmcer": bool}
  GET  /api/compare/result/{dataset}    kết quả lần chạy trước (compare.json)
  GET  /api/compare/file/{dataset}/{n}  tải csv/png/md đã xuất
Không cần DiGemoEngine (không dùng Whisper/encoder), nên chạy được cả khi app mở bằng --compare-only.
"""
from __future__ import annotations

import io
import json
import logging
import sys
import threading
from contextlib import redirect_stdout
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))

import compare_models as cm  # noqa: E402

LOG = logging.getLogger('digemo.compare')
FILES = {'per_utterance.csv', 'disagreements.csv', 'metrics.json', 'summary.md', 'cm_compare.png',
         'f1_per_class.png', 'compare.json'}


class RunRequest(BaseModel):
    dataset: str
    run_dmcer: bool = False


def register_compare_routes(app, config_path=None):
    lock = threading.Lock()
    state = {'running': None}

    def cfg():
        return cm.load_config(config_path)

    def check_ds(ds):
        ds = ds.lower()
        if ds not in cm.DATASETS:
            raise HTTPException(400, f'dataset phải là {", ".join(cm.DATASETS)}')
        return ds

    @app.get('/compare')
    def compare_page():
        return FileResponse(str(HERE / 'static' / 'compare.html'))

    @app.get('/api/compare/config')
    def compare_config():
        c = cfg()
        d = cm.describe_config(c)
        d['config_file'] = str(Path(config_path) if config_path else cm.DEFAULT_CONFIG)
        d['config_exists'] = Path(d['config_file']).is_file()
        d['running'] = state['running']
        d['labels'] = cm.DISPLAY
        d['has_result'] = {ds: (cm.in_root(c['out_dir']) / ds / 'compare.json').is_file() for ds in cm.DATASETS}
        return d

    @app.post('/api/compare/run')
    def compare_run(req: RunRequest):          # def (không async): FastAPI chạy trong threadpool
        ds = check_ds(req.dataset)
        if not lock.acquire(blocking=False):
            raise HTTPException(409, f'Đang chạy so sánh {state["running"]}, đợi xong rồi bấm lại.')
        state['running'] = ds
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                res = cm.run_compare(ds, cfg(), run_dmcer=req.run_dmcer, log=print)
            res['log'] = buf.getvalue()[-6000:]
            return res
        except SystemExit as e:                # lỗi cấu hình/dữ liệu có thông báo rõ ràng
            raise HTTPException(400, f'{e}\n\n{buf.getvalue()[-3000:]}'.strip())
        except Exception as e:
            LOG.exception('compare %s failed', ds)
            raise HTTPException(500, f'{type(e).__name__}: {e}\n\n{buf.getvalue()[-3000:]}'.strip())
        finally:
            state['running'] = None
            lock.release()

    @app.get('/api/compare/result/{dataset}')
    def compare_result(dataset: str):
        ds = check_ds(dataset)
        p = cm.in_root(cfg()['out_dir']) / ds / 'compare.json'
        if not p.is_file():
            raise HTTPException(404, f'Chưa có kết quả {ds.upper()}: bấm "Chạy so sánh".')
        return json.loads(p.read_text(encoding='utf-8'))

    @app.get('/api/compare/file/{dataset}/{name}')
    def compare_file(dataset: str, name: str):
        ds = check_ds(dataset)
        if name not in FILES:
            raise HTTPException(404, 'Không có file này')
        p = cm.in_root(cfg()['out_dir']) / ds / name
        if not p.is_file():
            raise HTTPException(404, f'Chưa có {name}')
        return FileResponse(str(p), filename=f'{ds}_{name}')
