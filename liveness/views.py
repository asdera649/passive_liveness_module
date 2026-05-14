# liveness/views.py
"""
API views for passive liveness detection.

Endpoints
─────────
POST /api/liveness/check/
    Accepts multipart/form-data with field `image` OR
    application/json with field `image_base64`.
    Returns a JSON liveness result.

GET  /api/liveness/health/
    Service health-check — returns loaded model names and device info.
"""

import base64
import logging

import numpy as np
from rest_framework import status
from rest_framework.parsers import MultiPartParser, JSONParser
from rest_framework.response import Response
from rest_framework.views import APIView

from liveness.service import LivenessService

logger = logging.getLogger('liveness')


class LivenessCheckView(APIView):
    """
    POST /api/liveness/check/

    Multipart upload:
        Content-Type: multipart/form-data
        Body field: image  (file)

    JSON base64 upload:
        Content-Type: application/json
        Body: { "image_base64": "<base64 encoded image>" }

    Response 200:
    {
        "success": true,
        "is_real": true,
        "score": 0.87,          // probability of being a real face
        "label": 1,
        "label_name": "real",
        "raw_scores": [0.05, 0.87, 0.08],
        "face_bbox": [x, y, w, h],
        "error": null
    }
    """

    parser_classes = [MultiPartParser, JSONParser]

    def post(self, request):
        # ── 1. Resolve image source ──────────────────────────────────────────
        image_input = None

        # Multipart file upload
        if 'image' in request.FILES:
            image_input = request.FILES['image']

        # JSON base64
        elif 'image_base64' in request.data:
            raw_b64 = request.data['image_base64']
            # Strip optional data-uri prefix: "data:image/jpeg;base64,..."
            if ',' in raw_b64:
                raw_b64 = raw_b64.split(',', 1)[1]
            try:
                image_input = base64.b64decode(raw_b64)
            except Exception:
                return Response(
                    {'success': False, 'error': 'Invalid base64 data.'},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        if image_input is None:
            return Response(
                {
                    'success': False,
                    'error': (
                        "No image provided. "
                        "Send a multipart file as 'image' or "
                        "a base64 string as 'image_base64'."
                    ),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # ── 2. Run liveness check ────────────────────────────────────────────
        try:
            service = LivenessService.get_instance()
            result = service.check(image_input)
        except Exception as exc:
            logger.exception("Liveness check failed")
            return Response(
                {'success': False, 'error': f'Inference error: {exc}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # ── 3. Build response ────────────────────────────────────────────────
        response_data = {'success': True, **result.to_dict()}

        # If there was a soft error (no face, decode issue) return 422
        if result.error:
            return Response(response_data, status=status.HTTP_422_UNPROCESSABLE_ENTITY)

        return Response(response_data, status=status.HTTP_200_OK)


class LivenessHealthView(APIView):
    """
    GET /api/liveness/health/

    Returns the current status of the liveness service, including
    which models are loaded and on which device.
    """

    def get(self, request):
        try:
            service = LivenessService.get_instance()
            models_info = [
                {
                    'filename': __import__('os').path.basename(path),
                    'path': path,
                }
                for path in service.loaded_models
            ]
            return Response(
                {
                    'status': 'ok',
                    'device': str(service.device),
                    'models_loaded': len(models_info),
                    'models': models_info,
                    'real_threshold': service.real_threshold,
                }
            )
        except Exception as exc:
            logger.exception("Health check failed")
            return Response(
                {'status': 'error', 'detail': str(exc)},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
