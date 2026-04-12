@echo off
REM Script pour initialiser Git et pousser vers GitHub
REM Version 0.0beta1.0

echo ========================================
echo  Initialisation Git et Push - CESA 0.0beta1.0
echo ========================================
echo.

REM Vérifier si Git est installé
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Git n'est pas installe ou n'est pas dans le PATH.
    echo Veuillez installer Git depuis https://git-scm.com/
    pause
    exit /b 1
)

REM Vérifier si .git existe
if exist .git (
    echo [INFO] Depot Git deja initialise.
    goto :add_files
) else (
    echo [1/6] Initialisation du depot Git...
    git init
    echo.
)

:add_files
REM Ajouter tous les fichiers
echo [2/6] Ajout des fichiers...
git add .

REM Créer le fichier .gitignore s'il n'existe pas
if not exist .gitignore (
    echo.
    echo [INFO] Creation du fichier .gitignore...
    (
        echo __pycache__/
        echo *.pyc
        echo *.pyo
        echo *.pyd
        echo .Python
        echo *.log
        echo .pytest_cache/
        echo .mypy_cache/
        echo .vscode/
        echo .idea/
        echo *.egg-info/
        echo dist/
        echo build/
        echo .env
        echo *.swp
        echo *.swo
        echo *~
    ) > .gitignore
    git add .gitignore
)

REM Créer le commit
echo.
echo [3/6] Creation du commit...
git commit -m "chore: bump version to 0.0beta1.0

- Updated version number from 0.0alpha3.0 to 0.0beta1.0
- Updated run.py, requirements.txt, INSTALL.bat
- Updated documentation files
- Updated prepare_release.py
- Added GitHub Actions workflows and templates"

REM Demander l'URL du dépôt GitHub
echo.
echo [4/6] Configuration du remote GitHub
echo.
echo Veuillez entrer l'URL de votre depot GitHub
echo Exemple: https://github.com/votre-username/CESA.git
echo Ou: git@github.com:votre-username/CESA.git
echo.
set /p REPO_URL="URL du depot GitHub: "

if "%REPO_URL%"=="" (
    echo [INFO] Pas d'URL fournie. Configuration du remote ignoree.
    echo Vous pouvez configurer le remote plus tard avec:
    echo   git remote add origin VOTRE_URL
    goto :create_tag
)

REM Configurer le remote
echo.
echo [5/6] Ajout du remote origin...
git remote remove origin >nul 2>&1
git remote add origin %REPO_URL%

REM Pousser vers GitHub
echo.
echo [6/6] Push vers GitHub...
git branch -M main
git push -u origin main

if %errorlevel% neq 0 (
    echo.
    echo [ATTENTION] Le push a echoue.
    echo Verifiez que:
    echo   1. L'URL du depot est correcte
    echo   2. Vous avez les permissions d'ecriture
    echo   3. Le depot existe sur GitHub
    echo.
    pause
    exit /b 1
)

:create_tag
echo.
echo ========================================
echo  Configuration terminee avec succes!
echo ========================================
echo.
echo Pour creer un tag pour cette version:
echo   git tag -a v0.0beta1.0 -m "Release 0.0beta1.0"
echo   git push origin v0.0beta1.0
echo.

pause

