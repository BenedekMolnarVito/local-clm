#Requires -Version 7
param([switch]$WithReference, [switch]$WithQ5)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-Location $PSScriptRoot

function Get-Verified($Url, $OutFile, $Sha256) {
    if ((Test-Path $OutFile) -and (Get-FileHash -Algorithm SHA256 $OutFile).Hash -ieq $Sha256) { return }
    Write-Host "download: $OutFile"
    New-Item -ItemType Directory -Force (Split-Path -Parent $OutFile) | Out-Null
    Invoke-WebRequest -Uri $Url -OutFile "$OutFile.part" -Resume
    if ((Get-FileHash -Algorithm SHA256 "$OutFile.part").Hash -ine $Sha256) {
        Remove-Item -Force "$OutFile.part"
        throw "sha256 mismatch: $OutFile"
    }
    Move-Item -Force "$OutFile.part" $OutFile
}

function Invoke-Native {
    & $args[0] $args[1..($args.Count - 1)]
    if ($LASTEXITCODE -ne 0) { throw "failed ($LASTEXITCODE): $args" }
}

# llama.cpp (Vulkan build)
Get-Verified 'https://github.com/ggml-org/llama.cpp/releases/download/b11829/llama-b11829-bin-win-vulkan-x64.zip' `
    'bin/llama-b11829-bin-win-vulkan-x64.zip' '839ff55ce9cbee6a099b403b640ca6b694ee5d74b815e39f77fc5d2b20962f0d'
if (-not (Test-Path 'bin/llama/llama-server.exe')) {
    Expand-Archive -Force 'bin/llama-b11829-bin-win-vulkan-x64.zip' 'bin/llama'
    if (-not (Test-Path 'bin/llama/llama-server.exe')) {
        $sub = Get-ChildItem -Directory 'bin/llama' | Where-Object { Test-Path (Join-Path $_.FullName 'llama-server.exe') } | Select-Object -First 1
        if ($sub) {
            Get-ChildItem -Force $sub.FullName | Move-Item -Destination 'bin/llama'
            Remove-Item $sub.FullName
        }
    }
    if (-not (Test-Path 'bin/llama/llama-server.exe')) { throw 'llama-server.exe missing in bin/llama' }
}

# Encoder models
$hf = 'https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main'
Get-Verified "$hf/Qwen3-8B-Q6_K.gguf" 'models/Qwen3-8B-Q6_K.gguf' 'cb042ccd76795a8830d6be6bd4165245847cc68e41797b13bd61aed4c2cfbce6'
if ($WithQ5) {
    Get-Verified "$hf/Qwen3-8B-Q5_K_M.gguf" 'models/Qwen3-8B-Q5_K_M.gguf' '068bae163faa96ad48032daf4e071a6a28fe67d8dcc95367609c2ff165e52738'
}
if ($WithReference) {
    Get-Verified "$hf/Qwen3-8B-Q8_0.gguf" 'models/Qwen3-8B-Q8_0.gguf' '408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6'
}

# venv, CLM package and head
if (-not (Test-Path '.venv/Scripts/python.exe')) { Invoke-Native uv venv .venv --python 3.13 }
Invoke-Native uv pip install --require-hashes -r requirements.txt --python .venv/Scripts/python.exe
Invoke-Native uv pip install --no-deps contrastive-lm==0.1.0 --python .venv/Scripts/python.exe
Get-Verified 'https://huggingface.co/Contrastive-LM/CLM-v0.1-8B/resolve/e939398d4556fcd9400c76fa8c5a513202f42b0a/CLM_v0.1-8B.pt' `
    "$env:USERPROFILE/.cache/clm/CLM_v0.1-8B.pt" 'b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5'

Write-Host 'setup: ok'
