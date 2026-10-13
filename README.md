# Biome-Locked Forest Cover Classification


LIVE LINK :https://forest-cover-app.onrender.com


Full-stack companion to "Biome-Locked Learning and the Cost of Multimodal Reasoning".
Six classifiers on the UCI Forest Cover Type data, with measured reasoning cost, ablations,
cross-validation, significance tests and a biome-locked (leave-one-wilderness-area-out) evaluation.

```
forest-cover-app/
├── README.md
├── REVIEWER_NOTES.md      # reviewer-style critique, paper fixes, slide outline
├── backend/
│   ├── requirements.txt
│   ├── train.py           # experiments -> models/*.joblib + models/metrics.json
│   └── app.py             # FastAPI: /api/metrics, /api/predict, /api/predict_all
└── frontend/index.html    # presentation dashboard served by the backend
```

## Run
```bash
cd backend
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train.py --quick          # development run
uvicorn app:app --reload         # http://localhost:8000
```
For paper-quality numbers run `python train.py` (full 581,012 rows). Add `--skip-gb` if
Gradient Boosting takes too long. Full-data Random Forest / Extra Trees files can reach hundreds of MB.
