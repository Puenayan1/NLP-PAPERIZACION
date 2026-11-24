<#
.SYNOPSIS
    Despliegue automatizado de AbstractBot (Entorno Web Local) en Windows PowerShell.
.EXAMPLE
    .\run_abstractbot.ps1
    .\run_abstractbot.ps1 -Port 8080 -NoNli -Preload
#>
param(
    [int]$Port = 8000,
    [string]$HostIp = "127.0.0.1",
    [switch]$NoNli,
    [switch]$Preload,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Ubicarse siempre en la raíz donde vive este script
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $RootDir

Write-Host "`n=========================================================" -ForegroundColor Cyan
Write-Host "   AbstractBot · Despliegue Automatizado en Windows" -ForegroundColor Cyan
Write-Host "=========================================================`n" -ForegroundColor Cyan

# 1. Validar archivos esenciales del repositorio y de Google Drive
$RequiredPaths = @(
    "server.py",
    "web\index.html",
    "configs\generacion.yaml",
    "configs\flesch_bounds.json",
    "prompts\P-PAP-v1.txt",
    "prompts\P-BASE-v1.txt",
    "src\paperizar\generate.py",
    "src\paperizar\verify.py",
    "models\ner_hybrid_v1"
)

$Missing = @()
foreach ($p in $RequiredPaths) {
    if (-not (Test-Path $p)) { $Missing += $p }
}

if ($Missing.Count -gt 0) {
    Write-Host "[ERROR] Faltan archivos o carpetas requeridas en $RootDir :" -ForegroundColor Red
    $Missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
    Write-Host "`nDescarga las carpetas faltantes de tu Google Drive (/MyDrive/paperizar) y vuelve a ejecutar." -ForegroundColor Red
    exit 1
}
Write-Host "[1/5] Estructura de directorios y modelos verificada." -ForegroundColor Green

# 2. Crear / localizar entorno virtual Python (.venv)
$VenvDir = Join-Path $RootDir ".venv"
$VenvPy  = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $VenvPy)) {
    Write-Host "[2/5] Creando entorno virtual local (.venv)..." -ForegroundColor Yellow
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3 -m venv $VenvDir
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        python -m venv $VenvDir
    } else {
        Write-Host "[ERROR] No se encontró Python 3 instalado en el PATH de Windows." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[2/5] Entorno virtual (.venv) detectado." -ForegroundColor Green
}

# 3. Instalar dependencias (con detección automática de GPU NVIDIA para PyTorch CUDA)
if (-not $SkipInstall) {
    Write-Host "[3/5] Verificando dependencias y soporte de aceleración GPU..." -ForegroundColor Yellow
    & $VenvPy -m pip install --upgrade pip -q

    # Comprobar si torch ya está instalado y si tiene CUDA activo
    $TorchStatus = & $VenvPy -c "import importlib.util; print('installed' if importlib.util.find_spec('torch') else 'missing')" 2>$null
    $HasNvidiaGpu = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)

    if ($TorchStatus -eq "missing") {
        if ($HasNvidiaGpu) {
            Write-Host "      -> GPU NVIDIA detectada: instalando PyTorch con soporte CUDA..." -ForegroundColor Cyan
            & $VenvPy -m pip install torch --index-url https://download.pytorch.org/whl/cu124
            if ($LASTEXITCODE -ne 0) {
                Write-Host "      -> Reintentando con índice CUDA más reciente..." -ForegroundColor Yellow
                & $VenvPy -m pip install torch --index-url https://download.pytorch.org/whl/cu126
            }
        } else {
            Write-Host "      -> Sin GPU NVIDIA dedicada: instalando PyTorch CPU..." -ForegroundColor Yellow
            & $VenvPy -m pip install torch
        }
    }

    Write-Host "      -> Instalando FastAPI, Uvicorn, Transformers, spaCy y utilidades..." -ForegroundColor Cyan
    & $VenvPy -m pip install fastapi uvicorn spacy textstat pyyaml rouge-score transformers accelerate sentencepiece protobuf -q

    # Verificar modelo base de vectores en_core_web_md por si se requiere respaldo
    $HasMd = & $VenvPy -c "import spacy.util; print('ok' if spacy.util.is_package('en_core_web_md') else 'no')" 2>$null
    if ($HasMd -ne "ok") {
        Write-Host "      -> Descargando paquete spaCy en_core_web_md..." -ForegroundColor Cyan
        & $VenvPy -m spacy download en_core_web_md
    }
    Write-Host "[3/5] Dependencias listas." -ForegroundColor Green
} else {
    Write-Host "[3/5] Instalación omitida por bandera -SkipInstall." -ForegroundColor DarkGray
}

# Mostrar diagnóstico rápido de hardware antes de arrancar
$DeviceDiag = & $VenvPy -c "import torch; print(f'CUDA activo ({torch.cuda.get_device_name(0)})' if torch.cuda.is_available() else 'Modo CPU (sin CUDA)')"
Write-Host "[4/5] Motor PyTorch: $DeviceDiag" -ForegroundColor Cyan

# 4. Asegurar __init__.py en src/ y src/paperizar/ y configurar PYTHONPATH
foreach ($initFile in @("src\__init__.py", "src\paperizar\__init__.py")) {
    if (-not (Test-Path $initFile)) { New-Item -ItemType File -Path $initFile -Force | Out-Null }
}
$env:PYTHONPATH = "$RootDir;$RootDir\src"
$env:PYTHONUTF8 = "1"

# 5. Lanzar sonda en segundo plano que abre el navegador apenas responda /api/health
$TargetUrl = "http://localhost:$Port"
$HealthUrl = "$TargetUrl/api/health"

Start-Job -ScriptBlock {
    param($hUrl, $tUrl)
    for ($i = 0; $i -lt 180; $i++) {
        Start-Sleep -Seconds 1
        try {
            $resp = Invoke-RestMethod -Uri $hUrl -TimeoutSec 2 -ErrorAction Stop
            if ($resp.status -eq "ok") {
                Start-Process $tUrl
                break
            }
        } catch { }
    }
} -ArgumentList $HealthUrl, $TargetUrl | Out-Null

# Construir argumentos para server.py
$ServerArgs = @("server.py", "--host", $HostIp, "--port", "$Port")
if ($NoNli)   { $ServerArgs += "--no-nli" }
if ($Preload) { $ServerArgs += "--preload" }

Write-Host "[5/5] Iniciando servidor en $TargetUrl (Presiona Ctrl+C para detener)..." -ForegroundColor Green
& $VenvPy @ServerArgs