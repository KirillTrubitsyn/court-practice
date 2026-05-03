# court-practice MCP

Production-ready MCP-сервер на Python (FastMCP, Streamable HTTP) для гибридного семантического поиска по корпусу определений Верховного Суда РФ (СКГД и СКЭС, 2018–2026, ~5346 кейсов).

Слои поиска:
- **Лексический** — BM25 поверх лемматизированных pymorphy3 токенов.
- **Семантический** — Voyage AI (`voyage-3-large`, 1024D) с cosine similarity на mmap-эмбеддингах.
- **Гибрид** — Reciprocal Rank Fusion (RRF) с настраиваемыми весами.

Кэш эмбеддингов запросов — Redis (TTL 30 дней, float16 → 2КБ/запрос).

---

## 1. Локальный запуск

### Требования

- Python 3.11+
- Redis (через `docker-compose` или нативно)
- Voyage AI API key

### Через docker-compose (быстрее всего)

```bash
cp .env.example .env
# Заполни VOYAGE_API_KEY и MCP_SECRET_KEY (генерация ниже)
python -c "import secrets; print(secrets.token_urlsafe(48))"

# Стартуем Redis + сервер
docker compose up --build -d

# Положи court_practice_db.json в ./data, затем:
docker compose exec server python -m scripts.build_index \
    --source /data/court_practice_db.json \
    --out-dir /data \
    --embeddings

# Проверка
curl http://localhost:8000/health
python -m scripts.healthcheck http://localhost:8000 "$MCP_SECRET_KEY"
```

### Без Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Запусти Redis отдельно (brew services start redis / docker run redis)
cp .env.example .env  # заполни VOYAGE_API_KEY и MCP_SECRET_KEY

# Скачать словари pymorphy3 — устанавливаются автоматически из pymorphy3-dicts-ru.

python -m scripts.build_index --source ./data/court_practice_db.json --out-dir ./data --embeddings

uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload
```

### Тесты и линт

```bash
pytest -q
ruff check app tests scripts
ruff format --check app tests scripts
mypy app
```

---

## 2. Деплой на Railway

### 2.1 Создать проект

```bash
railway login
railway init                       # выбираем "Empty project"
railway link                       # привязываем локальную директорию
```

В дашборде Railway:
- New → Service → Deploy from GitHub (или `railway up` из CLI).
- New → Database → Redis. Railway проставит `REDIS_URL` в env автоматически (Variables → Reference → Redis.REDIS_URL).
- В сервисе: Settings → Volumes → New Volume → mount path `/data`, размер 1 GB.

### 2.2 Выставить env-переменные

В Variables сервиса:

| Ключ | Значение |
|---|---|
| `VOYAGE_API_KEY` | ключ Voyage AI |
| `MCP_SECRET_KEY` | `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `REDIS_URL` | `${{Redis.REDIS_URL}}` (reference) |
| `DATA_DIR` | `/data` |
| `LOG_LEVEL` | `INFO` |
| `MCP_AUTH_PASSWORD` | Опционально. Включает OAuth для claude.ai web. Все, кому раздашь пароль, смогут подключиться. Если оставить пустым — работает только static Bearer для Desktop/Code. |
| `PUBLIC_BASE_URL` | Опционально. Если Railway почему-то не возвращает корректный Host — пропиши `https://<service>.up.railway.app` явно. |

`PORT` Railway проставляет сам — не трогай.

### 2.3 Индекс и эмбеддинги

Готовые `data/index.pkl.gz` и `data/embeddings.npy` уже в репозитории. При первом старте контейнера сервер автоматически копирует их в Volume `/data` (logger: `seed_copied`). Дополнительных действий **не требуется**.

Если хочешь пересобрать индекс из свежего корпуса:

```bash
# Положи court_practice_db.json в /data (railway shell, scp, либо pull с S3/GitHub Release):
curl -L https://your-bucket/court_practice_db.json -o /data/court_practice_db.json

python -m scripts.build_index \
    --source /data/court_practice_db.json \
    --out-dir /data \
    --embeddings \
    --force   # перезаписать существующие в /data
```

Время: ~30 сек на BM25 + ~5–10 минут на Voyage эмбеддинги для 5346 кейсов (~10М токенов, ~$0.2 при `voyage-3-large`).

### 2.4 Проверка

```bash
curl https://your-service.up.railway.app/health
# {"status":"ok","version":"0.1.0"}

python -m scripts.healthcheck https://your-service.up.railway.app "$MCP_SECRET_KEY"
# POST /mcp initialize -> 200
```

---

## 3. Подключение к Claude

URL MCP-сервера: `https://your-service.up.railway.app/mcp`

Сервер поддерживает **два механизма авторизации**:

| Клиент | Механизм | Что нужно |
|---|---|---|
| **claude.ai web** | OAuth 2.1 + DCR + PKCE | `MCP_AUTH_PASSWORD` в Variables на Railway |
| **Claude Desktop** / **Claude Code** | static Bearer | `MCP_SECRET_KEY` в `Authorization` header |

### 3.1 claude.ai (web) — multi-user через OAuth

**Один раз на Railway**: добавь в Variables `MCP_AUTH_PASSWORD = <общий пароль>`. Этот пароль играет роль "Wi-Fi-пароля для команды": все коллеги, которым ты его сообщишь, смогут подключиться. Если кто-то ушёл — поменяй пароль и сделай redeploy, существующие сессии инвалидируются по истечении access_token (24h по умолчанию).

**Каждому коллеге** (включая себя):

1. claude.ai → Settings → Connectors → **Add custom connector**.
2. Name: `court-practice`. URL: `https://your-service.up.railway.app/mcp`. Save.
3. В списке коннекторов нажать **Connect** на новом коннекторе.
4. Откроется страница авторизации твоего MCP-сервера → ввести `MCP_AUTH_PASSWORD` → **Войти**.
5. Claude автоматически выпустит refresh_token, инструменты появятся в списке.

OAuth flow:
- claude.ai дёргает `/.well-known/oauth-protected-resource` → `/.well-known/oauth-authorization-server`.
- Регистрируется как public client через `POST /register` (DCR, RFC 7591).
- Открывает `/authorize` с PKCE S256.
- Меняет authorization code на JWT access_token через `/token`.
- Все запросы к `/mcp` идут с `Authorization: Bearer <JWT>`.

### 3.2 Claude Desktop — static Bearer

Правка `claude_desktop_config.json` (`~/Library/Application Support/Claude/claude_desktop_config.json` на macOS):

```json
{
  "mcpServers": {
    "court-practice": {
      "type": "http",
      "url": "https://your-service.up.railway.app/mcp",
      "headers": {
        "Authorization": "Bearer ВАШ_MCP_SECRET_KEY"
      }
    }
  }
}
```

Перезапусти Claude Desktop. Иконка инструментов должна показать 5 tools.

### 3.3 Claude Code — static Bearer

```bash
claude mcp add --transport http court-practice \
    https://your-service.up.railway.app/mcp \
    --header "Authorization: Bearer ВАШ_MCP_SECRET_KEY"
```

Проверка: `claude mcp list` → должен быть court-practice со статусом ✓.

### 3.4 Ручной тест соединения

```bash
curl -X POST https://your-service.up.railway.app/mcp \
  -H "Authorization: Bearer $MCP_SECRET_KEY" \
  -H "Accept: application/json, text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0",
    "id":1,
    "method":"initialize",
    "params":{
      "protocolVersion":"2024-11-05",
      "capabilities":{},
      "clientInfo":{"name":"curl","version":"1"}
    }
  }'
```

Ожидаешь `200 OK` с JSON или SSE-потоком (зависит от версии MCP-клиента).

---

## 4. Tools API

| Tool | Назначение |
|---|---|
| `search_practice(query, mode, court, tag, article, year_from, year_to, limit)` | Гибридный поиск, возвращает компактные хиты (id, title, court, date, score, snippet, tags). |
| `get_case_details(case_id)` | Полная карточка: фабула, позиции нижестоящих и ВС, нормы, теги. |
| `find_similar(case_id, limit)` | Семантически близкие определения по cosine similarity. |
| `list_tags(min_count)` | Теги с частотами (топ → низ). |
| `stats()` | Метаданные базы: размер, разбивка по коллегиям и годам, дата индексации. |

Принцип «list + detail»: search возвращает только сниппеты, тяжёлые тексты — отдельно через `get_case_details`.

---

## 5. Архитектура

```
app/
├── server.py              FastMCP инстанс, lifespan, ASGI обвязка
├── config.py              Pydantic Settings
├── auth.py                BearerAuthMiddleware
├── logging_setup.py       Structured JSON-логи
├── search/
│   ├── lemmatizer.py      pymorphy3 + lru_cache на 200к лемм
│   ├── bm25.py            BM25Okapi обёртка
│   ├── semantic.py        Voyage клиент + SemanticIndex (matmul cosine)
│   ├── fusion.py          Reciprocal Rank Fusion
│   └── engine.py          SearchEngine — связывает все слои
├── storage/
│   ├── index_loader.py    pickle.gz / np.save с mmap
│   └── redis_cache.py     Async кэш float16-эмбеддингов
└── tools/                 Регистрация MCP-tools на инстансе

scripts/
├── build_index.py         Индексация корпуса + опциональные эмбеддинги
├── build_embeddings.py    Достройка эмбеддингов поверх готового индекса
└── healthcheck.py         Probe /health и /mcp initialize
```

### Ключевые решения

- **Lifespan-state**. Тяжёлые объекты (BM25, mmap-эмбеддинги, Voyage клиент, Redis) создаются ОДИН раз при старте и доступны в каждом tool через `ctx.request_context.lifespan_context`.
- **mmap для эмбеддингов**. `np.load(..., mmap_mode='r')` — матрица 5346×1024 float32 это ~22МБ, но при росте корпуса не упрёмся в RAM.
- **Graceful degradation**. Если Redis недоступен — запросы идут напрямую в Voyage. Если эмбеддинги не построены — `mode=hybrid` молча даунгрейдится в `lexical`.
- **Кэш float16**. На cosine similarity потеря точности ниже 1e-3, ранкинг неотличим.
- **Bearer middleware** через `secrets.compare_digest` — без timing-атак.

---

## 6. Тюнинг весов RRF

Параметры env-переменными (без передеплоя кода):

```
RRF_K=60          # сглаживание; меньше → агрессивнее доминирует топ-1
BM25_WEIGHT=1.0
SEMANTIC_WEIGHT=1.0
```

Если ловишь много синтаксических совпадений без смыслового — увеличь `SEMANTIC_WEIGHT`. Если наоборот, семантика «уносит» от точных формулировок — увеличь `BM25_WEIGHT`.

---

## 7. TODO для тебя

После генерации кода нужно сделать руками:

1. **Сгенерировать `MCP_SECRET_KEY`**:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(48))"
   ```
   Сохрани в Railway Variables и в локальном `.env`.

2. **Подложить корпус**. Загрузить `court_practice_db.json` в `/data` Volume на Railway (через `railway shell` + `curl` либо `railway volume push` если есть). Проверить, что схема ключей JSON совпадает с `_normalize()` в `scripts/build_index.py` — при необходимости поправить маппинг.

3. **Запустить индексацию**:
   ```bash
   railway run python -m scripts.build_index \
       --source /data/court_practice_db.json \
       --out-dir /data \
       --embeddings
   ```

4. **Проверить healthcheck**:
   ```bash
   curl https://your-service.up.railway.app/health
   python -m scripts.healthcheck https://your-service.up.railway.app "$MCP_SECRET_KEY"
   ```

5. **Подключить connector** в claude.ai web / Desktop / Code (см. раздел 3).

6. **Положить эталонный JSON в репозиторий** как пример (или его mini-семпл из 10 кейсов в `data/sample.json`) — поможет тестам и будущей переиндексации.
