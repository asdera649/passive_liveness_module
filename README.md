# Модуль пассивной проверки живости (Passive Liveness Detection)

## API

### `POST /api/liveness/check/`

Принимает изображение (multipart или base64), возвращает результат проверки живости.

**Вариант 1 - multipart (curl):**
```bash
curl -X POST http://localhost:8000/api/liveness/check/ \
     -F "image=@/path/to/face.jpg"
```

**Вариант 2 - JSON base64 (JavaScript):**
```js
const response = await fetch('/api/liveness/check/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ image_base64: base64String }),
});
const result = await response.json();
```

**Успешный ответ (200):**
```json
{
  "success": true,
  "is_real": true,
  "score": 0.8741,
  "label": 1,
  "label_name": "real",
  "raw_scores": [0.0521, 0.8741, 0.0738],
  "face_bbox": [142, 68, 310, 387],
  "error": null
}
```

**Ответ при отсутствии лица (422):**
```json
{
  "success": false,
  "is_real": false,
  "score": 0.0,
  "label": -1,
  "label_name": "unknown",
  "raw_scores": [],
  "face_bbox": null,
  "error": "No face detected in image."
}
```

---

### `GET /api/liveness/health/`

Проверка состояния сервиса.

```bash
curl http://localhost:8000/api/liveness/health/
```

```json
{
  "status": "ok",
  "device": "cpu",
  "models_loaded": 2,
  "models": [
    { "filename": "2.7_80x80_MiniFASNetV2.pth", "path": "..." },
    { "filename": "4_0_0_80x80_MiniFASNetV1SE.pth", "path": "..." }
  ],
  "real_threshold": 0.5
}
```

---

## Интеграция

```python
from liveness.service import LivenessService

def authenticate_user(request):
    face_image = request.FILES.get('face_image')

    service = LivenessService.get_instance()
    result = service.check(face_image)

    if not result.is_real:
        reason = result.error or f"Liveness failed (score={result.score:.2f})"
        # переход к активной проверке
        return {'step': 'active_liveness', 'reason': reason}

    # живость подтверждена
    return {'step': 'face_recognition', 'face_bbox': result.face_bbox}
```



## Настройки (`settings.py`)

| Ключ | По умолчанию | Описание                                     |
|---|---|----------------------------------------------|
| `MODEL_DIR` | `resources/anti_spoof_models` | Папка с `.pth` файлами                       |
| `DETECTOR_CAFFEMODEL` | `resources/detection_model/...caffemodel` | Веса детектора                               |
| `DETECTOR_PROTOTXT` | `resources/detection_model/deploy.prototxt` | Архитектура детектора                        |
| `DEVICE_ID` | `0` | GPU id; CPU если CUDA недоступна             |
| `REAL_SCORE_THRESHOLD` | `0.5` | Минимальный порог для признания лица живым   |
| `MAX_IMAGE_MB` | `10` | Максимальный размер загружаемого изображения |
