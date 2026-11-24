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

# Usamos Continue para evitar que PowerShell 5.1 convierta warnings de Python/pip en stderr en NativeCommandError
$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Ubicarse siempre en la raiz donde vive este script
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $RootDir

Write-Host "`n=========================================================" -ForegroundColor Cyan
Write-Host "   AbstractBot - Despliegue Automatizado en Windows" -ForegroundColor Cyan
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
        Write-Host "[ERROR] No se encontro Python 3 instalado en el PATH de Windows." -ForegroundColor Red
        exit 1
    }
    if (-not (Test-Path $VenvPy)) {
        Write-Host "[ERROR] Fallo la creacion del entorno virtual (.venv)." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[2/5] Entorno virtual (.venv) detectado." -ForegroundColor Green
}

# Funcion auxiliar: ejecuta un snippet de Python via Base64 silenciando warnings y stderr desde Python
function Invoke-PyProbe([string]$Code) {
    $Wrapped = "import os, sys, warnings`nwarnings.filterwarnings('ignore')`nsys.stderr = open(os.devnull, 'w')`n" + $Code
    $B64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($Wrapped))
    $Out = & $VenvPy -W ignore -c "import base64; exec(base64.b64decode('$B64').decode('utf-8'))"
    return ($Out | Select-Object -Last 1)
}

# 3. Instalar dependencias (con deteccion de compatibilidad real de kernels CUDA, incl. RTX 50xx sm_120)
if (-not $SkipInstall) {
    Write-Host "[3/5] Verificando dependencias y soporte de aceleracion GPU..." -ForegroundColor Yellow
    & $VenvPy -m pip install --upgrade pip --quiet --disable-pip-version-check

    $HasNvidiaGpu = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
    $ProbeCode = @"
import importlib.util
if not importlib.util.find_spec('torch'):
    print('missing')
else:
    import torch
    if not torch.cuda.is_available():
        print('cpu_only')
    else:
        try:
            _ = (torch.zeros(1, device='cuda') + 1).item()
            print('cuda_ok')
        except Exception:
            print('cuda_incompatible')
"@
    $TorchState = Invoke-PyProbe $ProbeCode

    if ($HasNvidiaGpu) {
        if ($TorchState -ne "cuda_ok") {
            # Detectar version maxima de CUDA soportada por el driver NVIDIA (ej. 13.0 para RTX 50xx sm_120)
            $SmiText = (nvidia-smi | Out-String)
            $CudaTag = "cu128"
            if ($SmiText -match "CUDA Version:\s*(\d+)\.(\d+)") {
                $Major = [int]$Matches[1]
                $Minor = [int]$Matches[2]
                if ($Major -ge 13) {
                    $CudaTag = "cu130"
                } elseif ($Major -eq 12 -and $Minor -ge 8) {
                    $CudaTag = "cu128"
                } else {
                    $CudaTag = "cu126"
                }
            }

            if ($TorchState -eq "cuda_incompatible" -or $TorchState -eq "cpu_only") {
                Write-Host "      -> PyTorch actual no tiene kernels para tu GPU. Instalando build compatible ($CudaTag, ~2 GB)..." -ForegroundColor Yellow
                & $VenvPy -m pip uninstall -y torch torchvision torchaudio --quiet
            } else {
                Write-Host "      -> GPU NVIDIA detectada: instalando PyTorch con soporte CUDA ($CudaTag, ~2 GB)..." -ForegroundColor Cyan
            }

            & $VenvPy -m pip install torch --index-url "https://download.pytorch.org/whl/$CudaTag" --disable-pip-version-check
            if ($LASTEXITCODE -ne 0 -and $CudaTag -ne "cu128") {
                Write-Host "      -> Reintentando con indice CUDA cu128..." -ForegroundColor Yellow
                & $VenvPy -m pip install torch --index-url https://download.pytorch.org/whl/cu128 --disable-pip-version-check
            }
            if ($LASTEXITCODE -ne 0) {
                Write-Host "      -> Reintentando con indice CUDA cu126..." -ForegroundColor Yellow
                & $VenvPy -m pip install torch --index-url https://download.pytorch.org/whl/cu126 --disable-pip-version-check
            }
        } else {
            Write-Host "      -> PyTorch con kernels CUDA compatibles ya instalado." -ForegroundColor Green
        }
    } elseif ($TorchState -eq "missing") {
        Write-Host "      -> Sin GPU NVIDIA dedicada: instalando PyTorch CPU..." -ForegroundColor Yellow
        & $VenvPy -m pip install torch --disable-pip-version-check
    }

    Write-Host "      -> Verificando FastAPI, Uvicorn, Transformers, spaCy y utilidades..." -ForegroundColor Cyan
    & $VenvPy -m pip install fastapi uvicorn spacy textstat pyyaml rouge-score transformers accelerate sentencepiece protobuf --quiet --disable-pip-version-check

    # Verificar modelo base de vectores en_core_web_md por si se requiere respaldo
    $HasMd = Invoke-PyProbe "import spacy.util; print('ok' if spacy.util.is_package('en_core_web_md') else 'no')"
    if ($HasMd -ne "ok") {
        Write-Host "      -> Descargando paquete spaCy en_core_web_md..." -ForegroundColor Cyan
        & $VenvPy -m spacy download en_core_web_md
    }
    Write-Host "[3/5] Dependencias listas." -ForegroundColor Green
} else {
    Write-Host "[3/5] Instalacion omitida por bandera -SkipInstall." -ForegroundColor DarkGray
}

# Mostrar diagnostico real de ejecucion de kernels en hardware antes de arrancar
$DiagCode = @"
import torch
if torch.cuda.is_available():
    try:
        _ = (torch.zeros(1, device='cuda') + 1).item()
        cap = torch.cuda.get_device_capability(0)
        print(f'CUDA activo ({torch.cuda.get_device_name(0)}, sm_{cap[0]}{cap[1]} / torch {torch.__version__})')
    except Exception as e:
        print(f'ADVERTENCIA CUDA incompatible ({e})')
else:
    print('Modo CPU (sin CUDA)')
"@
$DeviceDiag = Invoke-PyProbe $DiagCode
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
& $VenvPy -W ignore @ServerArgs
