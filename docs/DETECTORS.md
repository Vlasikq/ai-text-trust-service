# Детекторы

Сервис поддерживает три режима детекции. Переключение делается через `DETECTOR_TYPE` без пересборки образа:

| Режим | Латентность | Точность на невидимых генераторах | Когда выбирать |
|---|---|---|---|
| `tfidf` | ~50 мс | стабильно средняя | по умолчанию; когда SLA важнее редких сложных случаев |
| `transformer` | ~500 мс на GPU, 2–3 с на CPU | лучшая на знакомых генераторах | когда нужно «качество любой ценой» |
| `cascade` | ~50 мс по быстрому пути, 5–15 с по медленному | лучшее среднее | когда есть GPU или запас CPU, и редкая «серая зона» не мешает UX |

Все три детектора реализуют общий контракт `BaseDetector` в [src/app/detectors/base.py](src/app/detectors/base.py):

```python
class BaseDetector(ABC):
    @abstractmethod
    def load(self) -> None: ...
    @abstractmethod
    def predict(self, text: str) -> DetectionResult: ...
    @abstractmethod
    def is_ready(self) -> bool: ...
    def get_info(self) -> dict: ...  # для /ready и логов


@dataclass
class DetectionResult:
    prob_ai: float            # вероятность класса "AI"
    method: str               # "tfidf" | "transformer" | "cascade"
    inference_ms: float       # время инференса (без препроцессинга и калибровки)
    metadata: dict            # дополнительно: cascade_path, device, warnings и т.п.
```

За счёт общего контракта новые детекторы добавляются без правок в `routers/` и `services/`: `services/detection.py` принимает любой `BaseDetector` одинаково.

## TF-IDF и LogReg

[src/app/detectors/tfidf.py](src/app/detectors/tfidf.py) — самая лёгкая и быстрая модель. Архитектура:

```
text → TfidfVectorizer(char, ngram=2-5)  ─┐
                                          ├→ hstack → LogisticRegression → P(AI)
text → TfidfVectorizer(word, ngram=1-2)  ─┘
```

Артефактов три, все в формате `joblib`:

- `tfidf_char_tm.joblib` (~8 МБ) — символьный векторизатор, обучен на topic-matched сплитах;
- `tfidf_word_tm.joblib` (~2 МБ) — словный векторизатор;
- `model_C_logreg.joblib` (<1 МБ) — LogReg, обученный поверх объединённой матрицы `hstack`.

Когда в `artifacts/service/` нужных файлов нет, подгружаются запасные `artifacts/tfidf_char.joblib`, `artifacts/tfidf_word.joblib`, `artifacts/model_A_logreg.joblib`. Это артефакты с не-topic-matched сплитов, и они нужны, чтобы сервис стартовал на чистом репозитории без вызова `scripts/export_service_artifacts.py`.

Почему именно такие фичи. Символьные n-граммы длины 2–5 ловят пунктуационные привычки, типографские артефакты (длинные тире, специфические кавычки) и частоту аффиксов. Словные n-граммы длины 1–2 покрывают лексические маркеры и устойчивые биграммы. Совмещение через `hstack` даёт разреженную матрицу примерно на 100 000 признаков, и `LogReg` обучается на 50 000 текстов за считаные минуты.

По латентности — около 50 мс на текст в 4000 символов (CPU, один поток). Доминирует трансформация векторизаторов, сама функция `predict` отрабатывает меньше чем за миллисекунду.

У модели есть три заметных плюса:

- Признаки не привязаны к конкретному домену (политика, новости, эссе и т. д.), и модель не разваливается на новых темах.
- Артефакт компактный, образ Docker с моделью укладывается примерно в 330 МБ.
- Работает на CPU без затрат на GPU и стоит в проде дёшево.

И три ограничения:

- Хуже работает на коротких текстах (меньше 500 символов): разреженная матрица почти пустая.
- Не улавливает семантические признаки — связность аргументации, логические переходы.
- Чувствительна к пре- и пост-редактуре AI-текста человеком.
- Обобщается на невидимые генераторы (например, на yandexgpt в holdout-сете) лишь при корректно собранных сплитах.

## Трансформер ruRoBERTa-large

[src/app/detectors/transformer.py](src/app/detectors/transformer.py) — это дообученная под нашу бинарную задачу AI/human модель `ai-forever/ruRoBERTa-large` (355 миллионов параметров). Артефакт около 1.4 ГБ — `pytorch_model.bin`, токенизатор и конфиг.

```
text → AutoTokenizer (truncation max_length=512)
     → AutoModelForSequenceClassification
     → softmax(logits)[1] = P(AI)
```

В образ Docker модель не зашита, иначе он распух бы на 1.4 ГБ. В YC она подтягивается отдельным контейнером `artifact-sync` из Object Storage в общий volume до старта API и воркера (см. [scripts/deploy/docker-compose.prod.yml:65](scripts/deploy/docker-compose.prod.yml#L65)).

Дообучение собрано в ноутбуке `notebooks/05_transformer_kaggle.ipynb`, тренировка занимает 3 эпохи на T4 в Kaggle. Воспроизводимый скрипт — `scripts/train_transformer.py`.

По латентности:

- GPU (T4 или RTX 4090) — 300–500 мс на текст;
- CPU с 16 ядрами — 2–3 секунды;
- standard-виртуалка YC s2.medium без GPU — 5–15 секунд.

Сильные стороны модели — лучшая точность на in-domain-распределении (когда генератор присутствовал в обучении) и способность ловить семантические признаки: логические нестыковки и шаблонность аргументации. Слабых сторон тоже хватает: на невидимых генераторах ruRoBERTa проседает сильнее, чем TF-IDF (см. `notebooks/06_robustness.ipynb`), артефакт в 1.4 ГБ замедляет холодный старт и требует прогрева, а без GPU модель плохо ложится в real-time-сценарии.

`DETECTOR_TYPE=transformer` оправдан только для batch-режима с GPU. Для синхронного HTTP — нет.

## Каскад

[src/app/detectors/cascade.py](src/app/detectors/cascade.py) — это гибрид. По быстрому пути работает TF-IDF, а трансформер запускается только в «серой зоне» неопределённости.

```python
def predict(self, text: str) -> DetectionResult:
    fast_result = self._fast.predict(text)                      # TF-IDF, ~50 мс
    if fast_result.prob_ai <= cascade_lo or fast_result.prob_ai >= cascade_hi:
        fast_result.metadata["cascade_path"] = "fast"
        return fast_result                                       # уверенный ответ
    if self._slow is not None and self._slow.is_ready():
        slow_result = self._slow.predict(text)                   # серая зона: трансформер
        slow_result.metadata["cascade_path"] = "slow"
        return slow_result
    # Запасной путь: трансформер недоступен — отдаём TF-IDF с предупреждением
    fast_result.metadata["cascade_path"] = "fallback"
    fast_result.metadata["warning"] = "TRANSFORMER_UNAVAILABLE"
    return fast_result
```

Пороги по умолчанию: `cascade_lo=0.30`, `cascade_hi=0.70`. Под конкретные артефакты их можно подобрать командой `make tune-cascade`, которая запишет рекомендуемые значения в `artifacts/cascade_threshold_sweep.json`.

Распределение путей на holdout-сете (см. `notebooks/06_robustness.ipynb`) такое: примерно 92% запросов резолвится TF-IDF за порогами `0.30` и `0.70`, оставшиеся 8% попадают в серую зону и уходят на трансформер. Средняя латентность остаётся близка к TF-IDF (около 100 мс на текст), а точность — к трансформеру.

Если трансформер недоступен, каскад работает как обычный TF-IDF и добавляет в ответ `WarningCode.transformer_unavailable`. Это явный сигнал клиенту: вердикт получен по быстрому пути, доверие частичное.

`DETECTOR_TYPE=cascade` имеет смысл выбирать, когда выполнены три условия: есть CPU или GPU под медленный путь, в проде допустимы 5–15 секунд латентности на 8% запросов, и для серой зоны есть async-режим через `/api/v1/jobs`.

У синхронного `POST /api/v1/analyze` стоит таймаут `CASCADE_TIMEOUT_MS` (по умолчанию 3000 мс). Если каскад в серой зоне не успевает уложиться, клиенту возвращается 200 со `Status.ERROR` и `WarningCode.inference_timeout`, и тогда запрос нужно повторить через async-job.

## Калибровка по Платту

[src/app/calibration/platt.py](src/app/calibration/platt.py) исправляет ситуацию, когда сырая вероятность из `detector.predict` плохо отражает реальную точность. Например, модель сообщает уверенность 0.95, а правильно угадывает только 70% таких случаев. Разрыв измеряется метрикой ECE (Expected Calibration Error).

Логика следующая:

1. На валидационной выборке считается ECE для каждого метода: `tfidf`, `transformer`, `cascade`.
2. Если ECE меньше 0.05, калибратор не нужен и работает как passthrough.
3. Иначе обучается `LogReg` поверх сырых вероятностей и сохраняется в `artifacts/calibration/{method}_calibrator.joblib`.
4. На инференсе сырая `prob_ai` проходит через калибратор.

Калибратор не критичен для жизненного цикла. Если файлов нет, `app.state.calibrator = None`, и сервис работает на сырых вероятностях, оставляя предупреждение в логе. В текущих артефактах папка `artifacts/calibration/` пустая, поэтому предупреждение `calibration_unavailable` в ответ не попадает: код есть, но никто не загружен.

## Уровни риска и вердикт

После калибровки получаем `confidence`. Если калибратора нет, в это поле идёт сырая `prob_ai`. На основе `confidence` определяются три уровня риска и бинарный вердикт:

| confidence | risk_level | warning |
|---|---|---|
| `< 0.30` | LOW | — |
| `0.30 ≤ x < 0.70` | MEDIUM | `LOW_CONFIDENCE` |
| `≥ 0.70` | HIGH | — |

Пороги задаются в `Settings`:

- `risk_thresh_low = 0.30`;
- `risk_thresh_high = 0.70`;
- `verdict_threshold = 0.50` — отдельный порог для бинарного вердикта: `ai`, если `confidence ≥ 0.50`, иначе `human`.

Здесь есть тонкость: уровень риска и вердикт разнесены. Текст с `confidence = 0.55` получит `verdict = ai`, `risk_level = MEDIUM` и предупреждение `LOW_CONFIDENCE`. UI должен трактовать это как «модель склоняется к AI, но уверенности мало», а не как «AI-текст».

## Подбор порогов каскада

Скрипт `scripts/tune_cascade_thresholds.py` (запуск через `make tune-cascade`) делает следующее:

1. Загружает текущие артефакты сервиса.
2. Прогоняет их на стратифицированной подвыборке holdout-сета размером до 800 строк. На большем объёме трансформер на CPU считал бы около часа.
3. Перебирает сетку порогов `(lo, hi)` с шагом 0.05.
4. Считает метрики: F1 по AI-классу и долю запросов, ушедших на медленный путь.
5. Возвращает Парето-фронт по оси «качество vs латентность».

Результат записывается в `artifacts/cascade_threshold_sweep.json`, и рекомендованные значения подставляются в `cascade_lo` и `cascade_hi`. Дефолты `0.30` и `0.70` — это разумный компромисс на случай, когда калибровки нет.

## Посегментная оценка через `/api/v1/analyze/segments`

В [src/app/services/segment_scoring.py](src/app/services/segment_scoring.py) текст режется на скользящие окна по `N` предложений (по умолчанию `window=3`, `step=2`). Каждое окно оценивается как самостоятельный «текст», и endpoint возвращает вероятности по сегментам плюс сводку.

Сценарий применения такой: пользователь подозревает, что AI написал только часть текста — например, помог с введением, а остальное автор писал сам. Endpoint отдаёт массив сегментов с `prob_ai`, и фронт подсвечивает рискованные куски прямо в тексте. Реализация на фронте — `apps/web/src/components/SegmentedText.tsx`.

Сегментный путь вызывает тот же `detector.predict`, поэтому работает с любым `DETECTOR_TYPE`. У каскада в этом режиме «серая зона» встречается редко: на коротких сегментах TF-IDF почти всегда даёт категоричные вероятности.

## Дисклеймер

Сервис возвращает вероятностную оценку, а не доказательство авторства. Что он не делает:

- не проверяет факты;
- не оценивает качество или содержание текста;
- ограниченно работает на коротких фрагментах меньше `min_chars=300`;
- проседает на пост-отредактированных AI-текстах;
- проседает на моделях, которых не было в обучающей выборке.

Текст дисклеймера попадает в каждый `AnalyzeResponse.disclaimer` и в описание API в Swagger. Переопределяется переменной окружения `DISCLAIMER_TEXT` — это даёт возможность подставить разные формулировки в разные каналы (PWA, Telegram-бот, API).

Решения, у которых есть правовые или дисциплинарные последствия, должен принимать человек с учётом дополнительного контекста.

## Как добавить новый детектор

1. Создайте `src/app/detectors/your_detector.py` с реализацией контракта:

   ```python
   class YourDetector(BaseDetector):
       def load(self) -> None: ...
       def predict(self, text: str) -> DetectionResult: ...
       def is_ready(self) -> bool: ...
   ```

2. Добавьте новый вариант в `enum DetectorType` ([src/app/config.py:19](src/app/config.py#L19)).
3. Дополните `_build_detector` в [src/app/main.py:37](src/app/main.py#L37) и копию в [src/app/worker.py:54](src/app/worker.py#L54).
4. Положите артефакты в `artifacts/service/<your_detector>/` и подгружайте их в `load()`.
5. Тесты: добавьте фикстуру в `tests/conftest.py` и используйте тот же шаблон, что в `tests/test_detectors.py`.

Если детектору нужна калибровка, `scripts/calibration_eval.py` посчитает ECE на валидационной выборке и подскажет, стоит ли поднимать калибратор.
