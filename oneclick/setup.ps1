# ============================================================================
#  Ternary Bonsai 2 27B 一键运行（小白版）
#
#  做三件事：
#    1. 检测你的显卡（必须是 NVIDIA，显存 >= 12GB）
#    2. 第一次运行时，自动从官方源下载引擎 + 模型（约 10.5 GB）
#    3. 启动服务，并自动打开浏览器聊天界面
#
#  不会修改你系统里任何东西；所有文件都在这个文件夹内。
#  想删除：直接删掉整个文件夹即可。
#
#  双击 START.bat 即可（不要直接双击本文件）。
# ============================================================================
param(
  [int]$Port = 18200,
  [switch]$NoOpen,
  [switch]$CheckOnly
)
$ErrorActionPreference = 'Continue'
$Root    = $PSScriptRoot
$EngDir  = Join-Path $Root 'engine'
$ModDir  = Join-Path $Root 'models'
$ModelF  = Join-Path $ModDir 'Ternary-Bonsai-2-27B-ninfer-v3.ninfer'
$ModelURL = 'https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3/resolve/main/Ternary-Bonsai-2-27B-ninfer-v3.ninfer'
$RelBase  = 'https://github.com/kvmem/kvmem/releases/download/v0.18.0'

function Line($s) { Write-Host $s }

Line ''
Line '============================================================'
Line '  Ternary Bonsai 2 27B  一键运行'
Line '============================================================'
Line ''

# ---------------------------------------------------------------- 1. 显卡检测
Line '[1/4] 检测显卡 ...'
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) {
  Line '  [错误] 没找到 nvidia-smi。'
  Line '  这套东西需要一块 NVIDIA 显卡（RTX 30/40/50 系）和官方驱动。'
  Line '  如果你用的是 AMD 或 Intel 显卡，或者只有核显，跑不了。'
  exit 1
}
$raw = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null
$gpuName = ''; $gpuMem = 0
foreach ($ln in $raw) {
  $p = $ln -split ','
  if ($p.Count -ge 2) {
    $m = [int]($p[1].Trim())
    if ($m -gt $gpuMem) { $gpuMem = $m; $gpuName = $p[0].Trim() }
  }
}
if ($gpuName -eq '') {
  Line '  [错误] nvidia-smi 没有返回显卡信息，驱动可能有问题。'
  exit 1
}
Line ("  显卡: {0}   显存: {1} MB" -f $gpuName, $gpuMem)
if ($gpuMem -lt 11800) {
  Line ''
  Line '  [错误] 显存不够。'
  Line '  这个 27B 模型需要大约 11 GB 显存，所以至少要 12 GB 的卡。'
  Line '  能跑的常见型号：RTX 3060 12G / 4070 Ti Super 16G / 4080 / 4090 / 5070 及以上。'
  Line '  8G 的卡（3060 Ti / 4060 / 3070 等）跑不了这个模型。'
  exit 1
}

# ---------------------------------------------------------------- 2. 选引擎包
Line ''
Line '[2/4] 选择匹配的引擎 ...'
if ($gpuName -match 'RTX\s*50') {
  $zipName = 'kvmem-v0.18.0-ninfer-windows-rtx50-sm120a.zip'; $arch = 'RTX 50 系 (sm120a)'
} elseif ($gpuName -match 'RTX\s*40') {
  $zipName = 'kvmem-v0.18.0-ninfer-windows-rtx40-sm89.zip';  $arch = 'RTX 40 系 (sm89)'
} elseif ($gpuName -match 'RTX\s*30') {
  $zipName = 'kvmem-v0.18.0-ninfer-windows-rtx30-sm86.zip';  $arch = 'RTX 30 系 (sm86)'
} else {
  Line ("  [错误] 暂时只支持 RTX 30 / 40 / 50 系，检测到的是: {0}" -f $gpuName)
  exit 1
}
Line ("  匹配: {0}  ->  {1}" -f $arch, $zipName)

# ---------------------------------------------------------------- 3. 磁盘检查
Line ''
Line '[3/4] 检查磁盘空间 ...'
$drive = (Split-Path $Root -Qualifier)
$free = (Get-PSDrive -Name $drive.TrimEnd(':') -ErrorAction SilentlyContinue).Free
if ($free) {
  $freeGB = [math]::Round($free / 1GB, 1)
  Line ("  可用: {0} GB（需要约 13 GB：引擎 1 GB + 模型 9 GB + 解压）" -f $freeGB)
  if ($freeGB -lt 13) {
    Line '  [错误] 磁盘空间不足，请先清理，或把整个文件夹挪到空间够的盘。'
    exit 1
  }
} else {
  Line '  （无法读取磁盘剩余空间，跳过检查）'
}

if ($CheckOnly) {
  Line ''
  Line '  [只检查] 环境没问题，可以运行。'
  Line ("  将会下载引擎: {0}（约 1 GB）" -f $zipName)
  Line '  将会下载模型: Ternary-Bonsai-2-27B-ninfer-v3.ninfer（8.87 GB）'
  Line '  下载完会自动启动并打开浏览器。'
  exit 0
}

# ---------------------------------------------------------------- 4. 下载引擎
function Get-File($url, $out) {
  Line ("  下载: {0}" -f (Split-Path $out -Leaf))
  $args = @('-L', '-C', '-', '--retry', '5', '--retry-delay', '5', '--fail', '-o', $out, $url)
  & curl.exe @args
  return ($LASTEXITCODE -eq 0 -and (Test-Path $out))
}

$serverExe = $null
if (Test-Path $EngDir) {
  $found = Get-ChildItem $EngDir -Recurse -Filter 'ninfer-kvmem-server.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($found) { $serverExe = $found.FullName }
}
if (-not $serverExe) {
  Line ''
  Line '[4/4] 第一次运行，准备下载（共约 10.5 GB，看网速可能要 30~60 分钟）...'
  New-Item -ItemType Directory -Force -Path $EngDir | Out-Null
  $zipPath = Join-Path $EngDir $zipName
  if (-not (Test-Path $zipPath)) {
    if (-not (Get-File "$RelBase/$zipName" $zipPath)) { Line '  [错误] 引擎下载失败，请检查网络后重试（再运行一次会自动续传）。'; exit 1 }
  }
  Line '  解压引擎 ...'
  Expand-Archive -Path $zipPath -DestinationPath $EngDir -Force
  $found = Get-ChildItem $EngDir -Recurse -Filter 'ninfer-kvmem-server.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $found) { Line '  [错误] 解压后没找到引擎程序。'; exit 1 }
  $serverExe = $found.FullName
  Line ("  引擎就绪: {0}" -f $serverExe)
} else {
  Line ''
  Line '[4/4] 引擎已存在，跳过下载。'
}

# ---------------------------------------------------------------- 模型
if (-not (Test-Path $ModelF)) {
  New-Item -ItemType Directory -Force -Path $ModDir | Out-Null
  Line ''
  Line '  下载模型（8.87 GB，最大的一步）...'
  if (-not (Get-File $ModelURL $ModelF)) { Line '  [错误] 模型下载失败。再运行一次会自动从断点续传。'; exit 1 }
} else {
  Line '  模型已存在，跳过下载。'
}
$sz = [math]::Round((Get-Item $ModelF).Length / 1GB, 2)
Line ("  模型就绪: {0} GB" -f $sz)

# ---------------------------------------------------------------- 启动
Line ''
Line '  启动引擎（第一次会热身，约 1~3 分钟，请耐心等）...'
$argv = @(
  '--host', '127.0.0.1', '--port', "$Port",
  '--max-context', '262144', '--max-concurrency', '1',
  '--prefill-chunk', '256', '--default-max-tokens', '16384',
  '--kvmem-budget', '57344', '--kvmem-gen-reserve', '16384',
  '--kvmem-host-mib', '12288', '--kvmem-sink-tokens', '5120',
  '--kvmem-recent-tokens', '5120', '--kvmem-sessions', '1',
  '--kv-dtype', 'int8', '--device-profile', 'auto', '--device', '0',
  '--spec', 'mtp', '--draft-tokens', '4',
  '--ngram-draft-tokens', '31', '--ngram-min-match', '12', '--adaptive-mtp',
  '--vision', '--vision-residency', 'resident', '--vision-max-merged', '1024',
  '--default-thinking-budget', '8192', '--kvmem-score-agg', 'max',
  '--fast-prefill-kernel'
)
$elog = Join-Path $Root 'engine.log'
Remove-Item $elog, ($elog + '.err') -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath $serverExe -ArgumentList (@($ModelF) + $argv) -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $elog -RedirectStandardError ($elog + '.err')

$ok = $false
for ($i = 0; $i -lt 150; $i++) {
  Start-Sleep -Seconds 2
  if ($proc.HasExited) { break }
  try {
    $r = Invoke-RestMethod -Uri ("http://127.0.0.1:{0}/v1/models" -f $Port) -TimeoutSec 4
    if ($r.data) { $ok = $true; break }
  } catch {}
}
if (-not $ok) {
  Line ''
  Line '  [错误] 引擎没起来。最后几行日志：'
  Get-Content ($elog + '.err') -Tail 8 -ErrorAction SilentlyContinue | ForEach-Object { Line ('    ' + $_) }
  Line ''
  Line '  常见原因：驱动太旧（需要支持 CUDA 13.2 的驱动）、显存被别的程序占了、杀毒软件拦截。'
  exit 1
}

Line ''
Line '============================================================'
Line '  好了！浏览器会自动打开聊天界面。'
Line ''
Line ("  聊天地址: http://127.0.0.1:{0}/" -f $Port)
Line ("  接口地址: http://127.0.0.1:{0}/v1   (模型名 bonsai2-27b)" -f $Port)
Line ''
Line '  关掉服务：双击 STOP.bat'
Line '============================================================'
if (-not $NoOpen) { Start-Process ("http://127.0.0.1:{0}/" -f $Port) }
