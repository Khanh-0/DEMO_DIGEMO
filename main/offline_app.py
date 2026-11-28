from flask import Flask, request, jsonify
import threading, uuid
from pathlib import Path
from runtime import config, ModelRegistry
from adapters import get_adapter
from predictors import get_predictor

ROOT = Path(__file__).resolve().parent
config.validate()
app = Flask(__name__)
jobs, lock = {}, threading.Lock()

def update_job(job_id, **kw):
    with lock:
        if job_id in jobs: jobs[job_id].update(kw)

def process_job(job_id):
    try:
        job = jobs[job_id]
        ds, model = job['dataset'], job['model']
        adapter = get_adapter(ds)
        dialogues = adapter.load_dialogues(job.get('dialogue_ids'))
        update_job(job_id, status='predicting', total_dialogues=len(dialogues))
        predictor = get_predictor(model, ds)
        results = []
        for i, dia in enumerate(dialogues):
            results.extend(predictor.predict(dia.utterances))
            update_job(job_id, processed_dialogues=i+1)
        update_job(job_id, status='done', results=results)
    except Exception as e:
        update_job(job_id, status='error', error=str(e))

@app.get('/')
def home():
    return (ROOT / 'index.html').read_text(encoding='utf-8')

@app.post('/api/predict')
def predict():
    data = request.get_json() or {}
    ds, model = data.get('dataset','').upper(), data.get('model','')
    if not ModelRegistry.is_valid(ds, model):
        return jsonify(error='Invalid dataset/model'), 400
    jid = uuid.uuid4().hex
    jobs[jid] = {'dataset': ds, 'model': model, 'dialogue_ids': data.get('dialogue_ids'), 'status': 'queued', 'processed_dialogues':0, 'total_dialogues':0, 'results':[], 'error':None}
    threading.Thread(target=process_job, args=(jid,), daemon=True).start()
    return jsonify(id=jid)

@app.get('/api/status/<jid>')
def status(jid):
    job = jobs.get(jid)
    if not job: return jsonify(error='Not found'), 404
    return jsonify(status=job['status'], processed_dialogues=job.get('processed_dialogues',0), total_dialogues=job.get('total_dialogues',0), results=job.get('results',[]) if job['status']=='done' else [], error=job.get('error'))

if __name__ == '__main__':
    print('Demo: http://127.0.0.1:5000')
    app.run(host='127.0.0.1', port=5000, threaded=True)
