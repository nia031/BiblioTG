from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import auth
from biblioteca.models import TitleAuthor, Transacciones
import pandas as pd


def login(request):
    if request.method == 'POST':
        user = auth.authenticate(username=request.POST['username'],password=request.POST['password'])
        if user is not None:
            auth.login(request, user)
            return redirect('accounts:profile')
        else:
            return render(request, 'accounts/login.html',{'error':'username or password is incorrect.'})
    else:
        return render(request, 'accounts/login.html')

def logout(request):
    if request.method == 'POST':
        auth.logout(request)
        return redirect('accounts:login')
        
def profile(request):
    trans_user=Transacciones.objects.filter(usuario=request.user.get_username())

    if request.user.is_authenticated:
        return render(request,'accounts/profile.html',{'transacciones':trans_user})
    else:
        return render(request, 'accounts/login.html')
