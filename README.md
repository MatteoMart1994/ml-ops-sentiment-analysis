# Sentiment Inference Service

**Repository:** https://github.com/MatteoMart1994/ml-ops-sentiment-analysis

Servizio di inferenza per classificazione del sentiment (positivo / neutro / negativo) basato su [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), esposto via **FastAPI**.

## Quick start (locale)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000
```

```bash
curl -s localhost:8000/health | jq
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"text":"I love this!"}' | jq
```

## Test e qualità

```bash
flake8 app tests && black --check app tests
pytest -q
```

Il modello è caricato **una sola volta** all'avvio (`lifespan` in `app/main.py`), non a ogni richiesta.

## Dati di prova

`data/sample_sentiment.csv` contiene 102 tweet (34 per classe) estratti dallo split di test di [TweetEval — sentiment](https://huggingface.co/datasets/cardiffnlp/tweet_eval), il benchmark pubblico di riferimento per il modello. I testi provengono da Twitter e possono contenere linguaggio esplicito; le etichette sono quelle originali del dataset. Il test `test_model_on_tweeteval_sample` verifica che l'accuracy sul campione non scenda sotto 0.5, valore ampiamente inferiore a quello pubblicato per il modello: lo scopo è intercettare un modello errato o una pipeline di preprocessing rotta, non misurare le prestazioni.

## Docker

L'immagine usa `requirements-docker.txt` e installa **PyTorch CPU** dall'index ufficiale (inferenza RoBERTa su CPU, layer più leggeri nel Dockerfile).

```bash
docker build -t sentiment-inference:local .
docker run --rm -p 8000:8000 sentiment-inference:local
```

## Monitoraggio (Prometheus + Grafana)

Segnali monitorati:

| Segnale | Metrica | Uso |
|--------|---------|-----|
| Volume | `http_requests_total{handler="/predict"}` | carico, capacity |
| Latenza | `http_request_duration_seconds` | SLA, saturazione CPU |
| Errori | status 5xx su `/predict` | regressioni deploy |
| Distribuzione classi | `sentiment_predictions_total{label}` | **concept drift** |
| Lunghezza testi | `sentiment_input_text_length_chars` | **data drift** (dominio / lingua) |

Prometheus raccoglie le metriche; gli alert sono definiti in **Grafana Alerting** e provisionati da `monitoring/grafana/provisioning/alerting/sentiment_alerts.yml` (cartella *Sentiment* in *Alerting → Alert rules*):

- **HighInferenceLatency**: p95 di `/predict` > **2 s** per 5 minuti. L'inferenza su CPU resta normalmente sotto il secondo; un p95 stabilmente oltre 2 s indica saturazione o input anomali.
- **HighErrorRate**: quota di risposte 5xx > **5%** per 5 minuti. Soglia oltre la quale l'impatto sugli utenti non è più trascurabile; tipicamente un deploy difettoso o un problema di memoria.
- **NegativeClassDominance**: quota di predizioni `negative` > **60%** per 30 minuti. Segnala uno sbilanciamento persistente del traffico o un degrado del modello (vedi sezione sul drift); la finestra di 30 minuti evita allarmi su picchi brevi.

Nella demo non è configurato un canale di notifica (email, Slack, PagerDuty): gli alert sono visibili nella UI di Grafana.

```bash
docker compose up --build
# API :8000, Prometheus :9090, Grafana :3000 (admin/admin)
# Airflow UI :8088 (admin/admin), Postgres (solo rete interna)
```

Le credenziali `admin/admin` e la Fernet key di Airflow sono valori di default per l'esecuzione locale. Possono essere sovrascritti creando un file `.env` a partire da `.env.example`; in un ambiente condiviso vanno gestiti tramite un secret manager e non devono comparire nel repository.

## CI/CD

GitHub Actions (`.github/workflows/ci-cd.yml`):

1. **quality** — flake8 + black + pytest (con cache Hugging Face).
2. **build-and-publish** — build immagine e push su `ghcr.io` (dipende da `quality`).
3. **cd-deploy** — riporta l'immagine pubblicata (tag con lo SHA del commit) come punto di partenza per il rollout su staging; le credenziali del target di deploy sono gestite esclusivamente tramite repository secrets (es. `HUGGINGFACE_TOKEN`).

### Deploy opzionale su Hugging Face Spaces

1. Creare uno Space **Docker** collegato al repo.
2. Impostare `secrets.HUGGINGFACE_TOKEN` in GitHub.
3. Aggiungere uno step che esegue `huggingface-cli upload` o triggera il webhook del Space con l'immagine `ghcr.io/...`.

## Airflow (nello stesso Docker Compose)

Nel `docker-compose.yml`: **Postgres**, Airflow (init, webserver, scheduler). Servizi:

- **postgres** — metadata DB
- **airflow-init** — `db migrate` + utente admin (one-shot)
- **airflow-webserver** — UI su `http://localhost:8088`
- **airflow-scheduler** — esecuzione DAG

DAG `sentiment_retrain_cycle` in `infrastructure/airflow/dags/sentiment_retrain_dag.py`: raccolta → valutazione → branch su F1 → (re)train → deploy, con **XCom** e artifact in `MLOPS_ARTIFACT_DIR` (default compose: `/tmp/airflow_sentiment`).

Dopo `docker compose up`, apri Airflow, **unpause** il DAG e lancia un run manuale o attendi lo schedule `@weekly`.

## Struttura

```
app/                         # FastAPI + modello
tests/                       # pytest API + modello
data/                        # CSV di esempio con etichette
monitoring/                  # Config Prometheus, dashboard e alert Grafana
infrastructure/airflow/dags/ # DAG MLOps (montati nel container)
```

## Data drift e concept drift

Il modello `cardiffnlp/twitter-roberta-base-sentiment-latest` è stato addestrato su post Twitter in lingua inglese. In esercizio il servizio riceve i testi inviati dai sistemi client, la cui natura può evolvere nel tempo. Le prestazioni possono quindi degradare in assenza di qualsiasi difetto applicativo. Il monitoraggio distingue due fenomeni.

### Data drift

Per data drift si intende una variazione della distribuzione degli input rispetto a quella osservata in fase di addestramento o di messa in esercizio, a parità di definizione del task. Il modello continua a rispondere, ma opera su testi che non rappresentano più il dominio su cui è stato validato.

Segnali osservati dal servizio:

- **Lunghezza dei testi** (`sentiment_input_text_length_chars`): uno spostamento persistente della media o della distribuzione indica un cambiamento della sorgente (ad esempio, da messaggi brevi a testi strutturati). Per input superiori a 512 token interviene inoltre il troncamento del tokenizer.
- **Tasso di richieste rifiutate (HTTP 422)**: un aumento di input vuoti o eccedenti il limite di 2000 caratteri suggerisce nuove integrazioni o un utilizzo non previsto del servizio.

In un contesto produttivo il monitoraggio andrebbe esteso con l'analisi periodica di un campione di testi (lingua, vocabolario, distanza tra embedding rispetto a una baseline).

Poiché le etichette reali non sono disponibili per le richieste online, questi indicatori consentono di individuare un potenziale drift, che va confermato tramite valutazione su un campione etichettato.

### Concept drift

Per concept drift si intende una variazione della relazione tra testo ed etichetta corretta: input simili a quelli storici richiederebbero oggi una classificazione diversa, oppure il modello non riflette più il criterio di valutazione adottato dal business. Un fenomeno affine, spesso osservato insieme, è la variazione della frequenza relativa delle classi nel traffico reale (*prior shift*).

Segnali considerati:

- **Distribuzione delle classi predette** (`sentiment_predictions_total{label}`): uno scostamento stabile rispetto alla baseline storica è un indicatore da approfondire. L'alert `NegativeClassDominance` (quota di predizioni negative superiore al 60% per 30 minuti) ha questa funzione: segnala una condizione anomala senza attribuirne la causa, che può essere un reale cambiamento del traffico oppure un degrado del modello.
- **Metriche offline**: il task `evaluate_model` del DAG Airflow confronta la macro-F1 su un set etichettato con la soglia di 0.75 e, se inferiore, avvia il ramo di retraining. Nella presente implementazione il valore è dimostrativo; in esercizio verrebbe calcolato su etichette prodotte da revisione umana.

Come estensione, l'esposizione della confidenza delle predizioni come metrica consentirebbe di rilevare un aumento delle classificazioni incerte, segnale utile sia per il data drift sia per il concept drift.

### Interpretazione dei segnali

| Evidenza | Fenomeno più probabile | Azione |
|----------|------------------------|--------|
| Variazione di lunghezza, formato o lingua dei testi | Data drift | Analisi di un campione di input, valutazione su dati etichettati, eventuale validazione a monte |
| Variazione delle classi predette con input stabili, calo della F1 offline | Concept drift | Rivalutazione del modello, aggiornamento della baseline o retraining |

I due fenomeni possono manifestarsi contemporaneamente; in entrambi i casi la decisione finale si basa sulla valutazione offline, non sulle sole metriche di servizio.
