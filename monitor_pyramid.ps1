# Script de monitoring de la construction de la pyramide
$pyramidPath = "C:\Users\comeb\Documents\SupBiotech\Stage\BT4\Code\CESA 1.0\sample\S043_Ap_EDF+\_ms"

Write-Host "`n🔍 Monitoring de la construction de la pyramide..." -ForegroundColor Cyan
Write-Host "📁 Chemin : $pyramidPath`n" -ForegroundColor Gray

$iteration = 0
$lastSize = 0
$lastCount = 0

while ($true) {
    $iteration++
    
    if (Test-Path $pyramidPath) {
        $stats = Get-ChildItem $pyramidPath -Recurse -ErrorAction SilentlyContinue | 
                 Measure-Object -Property Length -Sum
        
        $count = $stats.Count
        $sizeMB = [math]::Round($stats.Sum / 1MB, 2)
        
        $deltaSize = $sizeMB - $lastSize
        $deltaCount = $count - $lastCount
        
        $timestamp = Get-Date -Format "HH:mm:ss"
        
        Write-Host "[$timestamp] " -NoNewline -ForegroundColor Yellow
        Write-Host "📊 Fichiers: $count " -NoNewline -ForegroundColor White
        Write-Host "(+$deltaCount) " -NoNewline -ForegroundColor Green
        Write-Host "| 💾 Taille: ${sizeMB} MB " -NoNewline -ForegroundColor White
        Write-Host "(+${deltaSize} MB)" -ForegroundColor Green
        
        # Vérifier si la construction est terminée (pas de changement)
        if ($deltaSize -eq 0 -and $iteration -gt 2) {
            Write-Host "`n✅ Construction probablement terminée (pas de changement)" -ForegroundColor Green
            
            # Vérifier l'attribut build_complete
            $zgroupPath = Join-Path $pyramidPath ".zgroup"
            if (Test-Path $zgroupPath) {
                Write-Host "✅ Fichier .zgroup trouvé" -ForegroundColor Green
            }
            
            break
        }
        
        $lastSize = $sizeMB
        $lastCount = $count
    } else {
        Write-Host "⏳ En attente de la création du dossier..." -ForegroundColor Yellow
    }
    
    Start-Sleep -Seconds 10
}

Write-Host "`n🎉 Monitoring terminé !`n" -ForegroundColor Cyan



