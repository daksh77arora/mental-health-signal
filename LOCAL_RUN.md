# Run locally

Use the Python environment that has the dependencies installed.

```powershell
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 2200 --reload
```

Open http://127.0.0.1:2200/ in a browser. The API documentation is at http://127.0.0.1:2200/docs and the health check is at http://127.0.0.1:2200/health.

The bundled model was serialized with scikit-learn 1.9.0. Keep that version aligned with the runtime when deploying.
