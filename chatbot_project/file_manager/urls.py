from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_ppt, name='upload_ppt'),
    path('ask/', views.ask_question, name='ask_question'),
]
