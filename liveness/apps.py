from django.apps import AppConfig
import logging

logger = logging.getLogger('liveness')


class LivenessConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'liveness'
    verbose_name = 'Liveness Detection'

    def ready(self):
        """
        Called once when Django starts.
        Pre-loads all anti-spoof models into memory so the first
        request doesn't pay the cold-start penalty.
        """
        # Skip during management commands that don't need the model
        # (e.g. migrate, collectstatic)
        import sys
        if any(cmd in sys.argv for cmd in ('migrate', 'collectstatic', 'makemigrations')):
            return

        try:
            from liveness.service import LivenessService
            service = LivenessService.get_instance()
            logger.info(
                "Liveness service ready — %d model(s) loaded on %s",
                len(service.loaded_models),
                service.device,
            )
        except Exception as exc:
            # Log but don't crash Django startup; the view will surface
            # a proper 503 if the service isn't available.
            logger.error("Failed to initialize LivenessService: %s", exc)
