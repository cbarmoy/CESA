@echo off
REM ========================================
REM   CESA v3.0 - Script d'installation
REM ========================================
REM
REM Script d'installation des dépendances pour CESA (Complex EEG Studio Analysis) v3.0
REM Installe automatiquement toutes les bibliothèques Python requises.
REM
REM Auteur: Côme Barmoy (IRBA)
REM Version: 3.0.0
REM Date: 2025-10-26

echo ========================================
echo   CESA v3.0 - Installation
echo ========================================
echo.

REM Vérification de Python
echo [1/3] Verification de Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Python n'est pas installe ou pas dans le PATH
    echo.
    echo [SOLUTION]:
    echo    1. Installez Python 3.8+ depuis https://www.python.org/downloads/
    echo    2. Cochez "Add Python to PATH" pendant l'installation
    echo.
    pause
    exit /b 1
)
python --version
echo.

REM Mise à jour de pip
echo [2/3] Mise a jour de pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo [ATTENTION] Echec de la mise a jour de pip
    echo Continuation avec la version actuelle...
)
echo.

REM Installation des dépendances
echo [3/3] Installation des dependances Python...
cd /d %~dp0\..\..
python -m pip install -r documentation\requirements.txt
if %errorlevel% neq 0 (
    echo [ERREUR] Echec de l'installation des dependances
    echo.
    echo [SOLUTION]:
    echo    - Verifiez votre connexion Internet
    echo    - Verifiez que Python 3.8+ est installe
    echo    - Essayez: pip install --user -r documentation\requirements.txt
    echo.
    pause
    exit /b 1
)
python -m pip install neurokit2
if %errorlevel% neq 0 (
    echo [ATTENTION] Echec installation neurokit2
    echo           Les methodes RR 'neurokit2' et 'kubios' ne seront pas disponibles.
)

echo.
echo ========================================
echo   Installation terminee avec succes!
echo ========================================
echo.
echo Pour lancer CESA:
echo   RUN.bat
echo.
echo Ou directement avec Python:
echo   python run.py
echo.
echo Pour plus d'informations, consultez README.md
echo.
pause
