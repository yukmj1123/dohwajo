"""siteprac URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
import firstpage.views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', firstpage.views.video_first, name='video_first'),
    path('video_first/', firstpage.views.video_first, name = 'video_first'),
    path('video_second/', firstpage.views.video_second, name = 'video_second'),
    path('print_result/', firstpage.views.print_result, name = 'print_result'),
    path('voice_record/', firstpage.views.voice_record, name = 'voice_record'),
    path('recandprint/', firstpage.views.recandprint, name = 'recandprint'),
    path('opvideo/', firstpage.views.opvideo, name = 'opvideo'),
    path('op2video/', firstpage.views.op2video, name = 'op2video'),
]
