from django.urls import path, include

urlpatterns = [
    path('api/liveness/', include('liveness.urls')),
]
