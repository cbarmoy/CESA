@echo off
REM Toujours travailler dans le dossier du script (double-clic = sinon run.py introuvable, fenetre qui se ferme)
cd /d "%~dp0"
REM Forcer l'encodage UTF-8 pour afficher correctement les caracteres Unicode
chcp 65001 >nul
REM CESA 0.0beta1.1 - Installation Complete
REM ========================================
REM Installation automatique complete de CESA 0.0beta1.1
REM Developpe pour l'Unite Neuropsychologie du Stress (IRBA)

echo.
echo ========================================
echo    CESA 0.0beta1.1 - Installation Complete
echo ========================================
echo.
echo Developpe pour l'Unite Neuropsychologie du Stress (IRBA)
echo Auteur: Come Barmoy
echo Version: 0.0beta1.1
echo.
echo ========================================
echo.

REM Activer les extensions de commandes pour les variables
setlocal enabledelayedexpansion

echo [1/6] Recherche de Python...
echo.

set FOUND_PYTHON=

REM D'abord Python.org (chemins entre guillemets : "C:\Program Files\..." sinon FOR casse sur les espaces)

for %%P in (
  "%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python39\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python38\python.exe"
  "C:\Python314\python.exe"
  "C:\Python313\python.exe"
  "C:\Python312\python.exe"
  "C:\Python311\python.exe"
  "C:\Python310\python.exe"
  "C:\Python39\python.exe"
  "C:\Python38\python.exe"
  "C:\Program Files\Python314\python.exe"
  "C:\Program Files\Python313\python.exe"
  "C:\Program Files\Python312\python.exe"
  "C:\Program Files\Python311\python.exe"
  "C:\Program Files\Python310\python.exe"
  "C:\Program Files\Python39\python.exe"
  "C:\Program Files\Python38\python.exe"
  "C:\Program Files (x86)\Python314\python.exe"
  "C:\Program Files (x86)\Python313\python.exe"
  "C:\Program Files (x86)\Python312\python.exe"
  "C:\Program Files (x86)\Python311\python.exe"
  "C:\Program Files (x86)\Python310\python.exe"
  "C:\Program Files (x86)\Python39\python.exe"
  "C:\Program Files (x86)\Python38\python.exe"
) do (
  if exist %%~P (
    %%~P --version >nul 2>&1
    if not errorlevel 1 (
      echo [OK] Python trouve: %%~P
      %%~P --version
      set "FOUND_PYTHON=%%~P"
      goto :python_found_std
    )
  )
)

:python_found_std
if defined FOUND_PYTHON goto :check_files

python --version >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Python trouve dans le PATH
    python --version
    set FOUND_PYTHON=python
    goto :check_files
)

echo.
echo [ERREUR] Python non trouve automatiquement
echo.
echo [SOLUTION]:
echo    1. Installez Python depuis https://www.python.org/downloads/
echo    2. Cochez "Add Python to PATH" lors de l'installation
echo    3. Redemarrez votre ordinateur apres l'installation
echo.
echo [SUPPORT]: come1.barmoy@supbiotech.fr
echo.
pause
exit /b 1

:check_files
echo.
echo [2/6] Verification des fichiers CESA...

REM Detecter si on est deja dans le dossier CESA ou un niveau au-dessus
if exist "run.py" (
    echo [OK] Fichiers detectes dans le repertoire courant
    set CESA_DIR=.
    goto :check_requirements
)

if exist "CESA\run.py" (
    echo [OK] Fichiers detectes dans le sous-dossier CESA
    set CESA_DIR=CESA
    goto :check_requirements
)

echo [ERREUR] Fichiers CESA non trouves
    echo.
    echo [SOLUTION] Assurez-vous d'etre dans le bon repertoire
echo    Lancez ce script depuis le dossier contenant run.py
echo    ou depuis le dossier parent contenant CESA\
    echo.
    echo Appuyez sur une touche pour continuer...
    pause >nul
    exit /b 1

:check_requirements
if exist "CESA\eeg_studio_fixed.py" goto :check_requirements_ok
if exist "%CESA_DIR%\eeg_studio_fixed.py" goto :check_requirements_ok
if exist "eeg_studio_fixed.py" goto :check_requirements_ok
echo [ERREUR] Fichier eeg_studio_fixed.py non trouve
echo.
echo [SOLUTION] Assurez-vous que tous les fichiers CESA sont presents
echo.
echo Appuyez sur une touche pour continuer...
pause >nul
exit /b 1

:check_requirements_ok

if not exist "requirements.txt" (
    echo [ERREUR] Fichier requirements.txt non trouve
    echo.
    echo [SOLUTION] Assurez-vous que tous les fichiers CESA sont presents
    echo.
    echo Appuyez sur une touche pour continuer...
    pause >nul
    exit /b 1
)

echo [OK] Fichiers CESA detectes

echo.
echo [3/6] Installation des dependances principales...

echo [INSTALLATION] Installation des dependances CESA...
"%FOUND_PYTHON%" -m pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo [ERREUR] Erreur lors de l'installation des dependances principales
    echo.
    echo [SOLUTION]:
    echo    1. Verifiez votre connexion Internet
    echo    2. Lancez en tant qu'administrateur
    echo    3. Essayez: "%FOUND_PYTHON%" -m pip install --user -r requirements.txt
    echo.
    echo Appuyez sur une touche pour continuer...
    pause >nul
)

echo [OK] Dependances principales installees

echo.
echo [3b/6] Viewer PyQtGraph: PySide6 + pyqtgraph ^(vue PSG par defaut^)...
"%FOUND_PYTHON%" -m pip install "PySide6>=6.5.0" "pyqtgraph>=0.13.0"
if !errorlevel! neq 0 (
    echo [ATTENTION] Echec PySide6/pyqtgraph - CESA utilisera Matplotlib pour le PSG.
    echo Verifiez la version Python ^(PySide6 exige souvent des roues precompilees 3.9-3.12^).
)

echo.
echo [4/6] Installation des modules Excel...

echo [INSTALLATION] Installation de xlrd (lecture fichiers Excel .xls)...
"%FOUND_PYTHON%" -m pip install xlrd>=2.0.0
if !errorlevel! neq 0 (
    echo [ATTENTION] Erreur lors de l'installation de xlrd
    echo    Continuation avec les autres modules...
) else (
    echo [OK] xlrd installe avec succes
)

echo [INSTALLATION] Installation d'openpyxl (lecture fichiers Excel .xlsx)...
"%FOUND_PYTHON%" -m pip install openpyxl>=3.0.0
if !errorlevel! neq 0 (
    echo [ATTENTION] Erreur lors de l'installation d'openpyxl
    echo    Continuation avec les autres modules...
) else (
    echo [OK] openpyxl installe avec succes
)

echo.
echo [5/6] Verification de l'installation...

echo [TEST] Test des modules principaux...
"%FOUND_PYTHON%" -c "import mne, numpy, matplotlib, pandas, neurokit2; print('[OK] Modules principaux OK')"
if !errorlevel! neq 0 (
    echo [ERREUR] Modules principaux non importes correctement
    echo.
    echo [SOLUTION] Relancez ce script ou contactez le support
    echo.
    pause
    exit /b 1
)

echo [TEST] Test des modules Excel...
"%FOUND_PYTHON%" -c "import xlrd, openpyxl; print('[OK] Modules Excel OK')"
if !errorlevel! neq 0 (
    echo [ATTENTION] Modules Excel non importes correctement
    echo    L'import Excel pourrait ne pas fonctionner
    echo.
    echo [SOLUTION] Relancez ce script
    echo.
)

echo [TEST] Test des modules multiscale (zarr, numcodecs)...
"%FOUND_PYTHON%" -c "import zarr, numcodecs; print('[OK] Modules multiscale OK')"
if !errorlevel! neq 0 (
    echo [ATTENTION] Modules multiscale non importes correctement
    echo    Le mode pre-calcule pourrait ne pas fonctionner
    echo.
    echo [SOLUTION] Relancez ce script
    echo.
)

echo [TEST] PySide6 + pyqtgraph ^(viewer PSG^)...
"%FOUND_PYTHON%" -c "import PySide6; import pyqtgraph; print('[OK] PySide6', PySide6.__version__, '| pyqtgraph', pyqtgraph.__version__)"
if !errorlevel! neq 0 (
    echo [ATTENTION] PySide6 ou pyqtgraph non importables - le PSG s^'ouvrira en Matplotlib.
    echo Essayez: "%FOUND_PYTHON%" -m pip install --upgrade PySide6 pyqtgraph
    echo.
)

echo.
echo [6/7] Test de lancement d'CESA...

echo [LANCEMENT] Test de lancement de CESA 0.0beta1.1...
echo    Veuillez patienter quelques secondes...
echo.

REM Test de lancement rapide - importer depuis le bon chemin
"%FOUND_PYTHON%" -c "import sys; sys.path.insert(0, '.'); from CESA.eeg_studio_fixed import main; print('[OK] CESA peut se lancer')"
if !errorlevel! neq 0 (
    echo [ERREUR] CESA ne peut pas se lancer correctement
    echo.
    echo [SOLUTION]:
    echo    1. Verifiez que tous les fichiers CESA sont presents
    echo    2. Relancez ce script d'installation
    echo    3. Consultez le README.md pour plus d'informations
    echo    4. Contactez le support: come1.barmoy@supbiotech.fr
    echo.
    echo Python utilise: %FOUND_PYTHON%
    echo.
    pause
    exit /b 1
)

echo [OK] CESA peut se lancer correctement

echo.
echo [7/7] Creation du raccourci sur le bureau...

REM Creer le raccourci sur le bureau
set DESKTOP=%USERPROFILE%\Desktop
set CURRENT_DIR=%~dp0
set SHORTCUT_PATH=%DESKTOP%\CESA 0.0beta1.1.lnk

echo [CREATION] Creation du raccourci CESA 0.0beta1.1 sur le bureau...
echo    Chemin: %SHORTCUT_PATH%
echo    Cible: %CURRENT_DIR%RUN.bat

REM Creer le raccourci avec PowerShell
powershell -Command "& {$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%'); $Shortcut.TargetPath = '%CURRENT_DIR%RUN.bat'; $Shortcut.WorkingDirectory = '%CURRENT_DIR%'; $Shortcut.Description = 'CESA 0.0beta1.1 - EEG Studio Analysis'; $Shortcut.IconLocation = '%CURRENT_DIR%CESA\logo\Icone_CESA.ico'; $Shortcut.Save()}"

if %errorlevel% == 0 (
    echo [OK] Raccourci cree avec succes sur le bureau
    echo    Nom: CESA 0.0beta1.1.lnk
    echo    Cible: RUN.bat
) else (
    echo [ATTENTION] Erreur lors de la creation du raccourci
    echo    Vous pouvez toujours utiliser RUN.bat directement
)

echo.
echo ========================================
echo    Installation CESA 0.0beta1.1 Terminee
echo ========================================
echo.
echo [OK] Installation reussie avec succes!
echo.
echo [LANCEMENT] Pour lancer CESA:
echo    1. Double-cliquez sur le raccourci "CESA 0.0beta1.1" sur le bureau
echo    2. Double-cliquez sur RUN.bat dans ce dossier
echo    3. Ou utilisez: cd CESA && "%FOUND_PYTHON%" run.py
echo.
echo [INFO] Pour plus d'informations:
echo    - Consultez le README.md
echo    - Contactez le support: come1.barmoy@supbiotech.fr
echo.
echo Python utilise: %FOUND_PYTHON%
echo.

echo Voulez-vous lancer CESA maintenant? (O/N)
set /p LAUNCH_NOW=
if /i "%LAUNCH_NOW%"=="O" (
    echo.
    echo [LANCEMENT] Lancement de CESA 0.0beta1.1...
    echo.
    "%FOUND_PYTHON%" run.py
    if !errorlevel! neq 0 (
        echo.
        echo [ERREUR] Erreur lors du lancement d'CESA
        echo.
        echo [SOLUTION] Utilisez RUN.bat pour lancer CESA
        echo.
        pause
    )
) else (
    echo.
    echo [CONSEIL] Utilisez RUN.bat pour lancer CESA quand vous voulez
    echo.
)

echo.
echo Appuyez sur une touche pour fermer...
pause >nul

exit /b 0
