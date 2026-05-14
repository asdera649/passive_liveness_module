"""
Django settings for liveness detection project.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get('SECRET_KEY', 'django-insecure-change-me-in-production')

DEBUG = os.environ.get('DEBUG', 'True') == 'True'

ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost 127.0.0.1').split()

INSTALLED_APPS = [
    'django.contrib.contenttypes',
    'django.contrib.auth',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'liveness.apps.LivenessConfig',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.MultiPartParser',
        'rest_framework.parsers.JSONParser',
    ],
    'EXCEPTION_HANDLER': 'liveness.exceptions.custom_exception_handler',
}

CORS_ALLOW_ALL_ORIGINS = DEBUG
CORS_ALLOWED_ORIGINS = os.environ.get('CORS_ALLOWED_ORIGINS', '').split() or []

LIVENESS_CONFIG = {
    'MODEL_DIR': os.environ.get(
        'ANTI_SPOOF_MODEL_DIR',
        str(BASE_DIR / 'resources' / 'anti_spoof_models')
    ),
    'DETECTOR_CAFFEMODEL': os.environ.get(
        'DETECTOR_CAFFEMODEL',
        str(BASE_DIR / 'resources' / 'detection_model' / 'Widerface-RetinaFace.caffemodel')
    ),
    'DETECTOR_PROTOTXT': os.environ.get(
        'DETECTOR_PROTOTXT',
        str(BASE_DIR / 'resources' / 'detection_model' / 'deploy.prototxt')
    ),
    # GPU device id; -1 → CPU
    'DEVICE_ID': int(os.environ.get('LIVENESS_DEVICE_ID', '0')),
    'REAL_SCORE_THRESHOLD': float(os.environ.get('LIVENESS_THRESHOLD', '0.5')),
    'MAX_IMAGE_MB': int(os.environ.get('LIVENESS_MAX_IMAGE_MB', '10')),
}

DATA_UPLOAD_MAX_MEMORY_SIZE = LIVENESS_CONFIG['MAX_IMAGE_MB'] * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = DATA_UPLOAD_MAX_MEMORY_SIZE

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'liveness': {
            'handlers': ['console'],
            'level': 'DEBUG' if DEBUG else 'INFO',
            'propagate': False,
        },
    },
}
