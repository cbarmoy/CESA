@echo off
REM Forcer l'encodage UTF-8 pour afficher correctement les caracteres Unicode
chcp 65001 >nul
REM Script de lancement CESA avec detection automatique de Python
REM ==============================================================

echo CESA - Comprehensive EEG Studio for Analysis
echo =============================================
echo.

REM Essayer d'abord avec python dans le PATH
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Python trouve dans le PATH
    python run.py
    goto :end
)

echo [INFO] Python non trouve dans le PATH, recherche automatique...

REM Rechercher Python dans les emplacements courants
set PYTHON_PATHS=
set PYTHON_PATHS=%PYTHON_PATHS% "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python313\python.exe"
set PYTHON_PATHS=%PYTHON_PATHS% "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe"
set PYTHON_PATHS=%PYTHON_PATHS% "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe"
set PYTHON_PATHS=%PYTHON_PATHS% "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe"
set PYTHON_PATHS=%PYTHON_PATHS% "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python39\python.exe"
set PYTHON_PATHS=%PYTHON_PATHS% "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python38\python.exe"
set PYTHON_PATHS=%PYTHON_PATHS% "C:\Program Files\Python313\python.exe"
set PYTHON_PATHS=%PYTHON_PATHS% "C:\Program Files\Python312\python.exe"
set PYTHON_PATHS=%PYTHON_PATHS% "C:\Program Files\Python311\python.exe"
set PYTHON_PATHS=%PYTHON_PATHS% "C:\Program Files\Python310\python.exe"
set PYTHON_PATHS=%PYTHON_PATHS% "C:\Program Files\Python39\python.exe"
set PYTHON_PATHS=%PYTHON_PATHS% "C:\Program Files\Python38\python.exe"

for %%p in (%PYTHON_PATHS%) do (
    if exist %%p (
        echo [OK] Python trouve: %%p
        %%p --version
        echo [LANCEMENT] Lancement de CESA...
        %%p run.py
        goto :end
    )
)

echo [ERREUR] Python non trouve!
echo.
echo [SOLUTION]:
echo    1. Installez Python depuis https://www.python.org/downloads/
echo    2. Cochez "Add Python to PATH" pendant l'installation
echo    3. Ou ajoutez Python manuellement au PATH systeme
echo.
pause

:end
