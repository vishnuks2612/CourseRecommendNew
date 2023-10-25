from django.urls import path
from . import views

urlpatterns = [
    # Other URL patterns
    path('', views.course_recommendation, name='course_recommendation'),
]
