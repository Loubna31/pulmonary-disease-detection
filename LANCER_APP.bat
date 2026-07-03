@echo off
title PneumoIA - Serveur Flask
cd /d C:\Users\hp\Desktop\newprjt
python run_flask.py > flask_log.txt 2>&1
echo Erreur detectee - voir flask_log.txt
pause
