# Docker RAG Multi-Agent Chatbot

Мультиагентный RAG-чатбот по документации Docker / контейнеров.

Стек:
- **LangGraph** — оркестрация агентов (moderation → rag agent → tools → writer)
- **Qdrant** — векторная БД (cosine)
- **FastAPI** — 3 эндпоинта
- **Pydantic Settings** + `.env` — смена LLM и embedding-модели без изменения кода
- **uv** — управление зависимостями и виртуальным окружением
- Локальная LLM через **Ollama** (рекомендуется `llama3.1:8b`) или любой OpenAI-compatible API

## Быстрый старт (локально с uv)

### 1. Установка uv (если ещё нет)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Подготовка проекта

```bash
cd docker-rag-bot
cp .env.example .env
# отредактируйте .env при необходимости
```

Создать venv и установить зависимости:

```bash
uv sync
```

Активировать окружение (опционально):

```bash
source .venv/bin/activate
```

Убедитесь, что Ollama запущена и модель скачана:

```bash
ollama pull llama3.1:8b
```

### 3. Запуск Qdrant (через Docker)

```bash
docker compose up qdrant -d
```

### 4. Ingestion (загрузка документов)

```bash
uv run python -c "
from app.services.ingestion import ingest_directory
print(ingest_directory(clear_existing=True))
"
```

Или через API после запуска сервиса (см. ниже).

### 5. Запуск FastAPI-сервиса

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Сервис: `http://localhost:8000`

### 6. Запросы

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Как создать Docker volume?", "thread_id": "demo-1"}'
```

Streaming:

```bash
curl -N -X POST "http://localhost:8000/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is docker compose?"}'
```

Ingestion через API:

```bash
curl -X POST "http://localhost:8000/ingest" -F "clear_existing=true"
```

### 7. Оценка retrieval

```bash
uv run python -m app.evaluation.retrieval_eval
```

---


## Развёртывание через Docker Compose (основной способ)

Приложение — FastAPI-сервис, готовый к деплою одной командой:

```bash
cp .env.example .env          # при необходимости отредактируйте
docker compose up --build -d
```

Что поднимается:
- **qdrant** — векторная БД (порты 6333/6334)
- **app** — FastAPI на порту **8000** (сборка через Dockerfile + uv)

Проверка:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

Загрузка документов в индекс:

```bash
curl -X POST "http://localhost:8000/ingest" -F "clear_existing=true"
```

Оценка retrieval:

```bash
docker compose exec app python -m app.evaluation.retrieval_eval
```

Логи:

```bash
docker compose logs -f app
```

Остановка:

```bash
docker compose down
```


---

## Архитектура агентов

```
User query
    │
    ▼
┌─────────────┐
│ moderation  │  ← structured output (relevant / refuse)
└──────┬──────┘
       │ relevant?
       ├─ no  → refuse → END
       │
       ▼
┌─────────────┐
│  rag_agent  │  ← tool-calling LLM, решает когда вызывать vector_search
└──────┬──────┘
       │ tool_calls?
       ├─ yes → ToolNode (vector_search) → writer
       └─ no  → writer
                 │
                 ▼
              final answer + documents
```

- Все промпты лежат в `app/prompts/*.txt`
- Состояние графа — `TypedDict` (`app/graph/state.py`)
- История диалога сохраняется через LangGraph `MemorySaver` (по `thread_id`)

## Смена моделей без изменения кода

В `.env`:

```env
LLM_PROVIDER=ollama          # или openai
OLLAMA_MODEL=llama3.1:8b
# или
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=...

EMBEDDING_PROVIDER=huggingface
HF_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
# или
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Перезапустите сервис — код менять не нужно.

## Дополнительная загрузка документов

```bash
uv run python scripts/download_docs.py --source github \
  --repo docker/docs --path content/manuals/engine --branch main
```

После скачивания снова вызовите `/ingest` или `ingest_directory()`.

## Структура проекта

```
app/
  config.py          # pydantic-settings
  main.py            # FastAPI
  graph/
    state.py         # TypedDict
    tools.py         # vector_search Tool
    nodes.py         # moderation / rag / writer
    graph.py         # LangGraph
  services/
    llm.py
    embeddings.py
    qdrant.py
    ingestion.py
  prompts/           # все промпты в .txt
  evaluation/        # precision@k / recall@k
data/documents/      # исходные документы
scripts/download_docs.py
pyproject.toml       # зависимости (uv)
docker-compose.yml
Dockerfile
```