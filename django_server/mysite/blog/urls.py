from django.urls import path
from . import views

urlpatterns = [
    path('', views.post_list),
    path('test/', views.data_return)
]
