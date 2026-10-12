# Start the serving stack: engine + keepalive proxy.
#
# Two processes on purpose:
#   engine  on 8082   RC3, upstream defaults + --fast-prefill-kernel, LRFU off
#   proxy   on 8081   the port clients point at; injects `: keepalive` SSE comments
#                     whenever the engine is silent, so a multi-minute prefill no
#                     longer trips a client's stream idle timeout
#
#   powershell -File start_stack.ps1            # start both
#   powershell -File start_stack.ps1 -Status    # show what is listening
#   powershell -File rc3_stop.ps1               # stop the engine (and the proxy is
#                                               # killed with it via the command match)
param([switch]$Status, [int]$KeepaliveSeconds = 15, [int]$EnginePort = 8082, [int]$ProxyPort = 8081)
$ROOT = 'C:\Users\Administrator\AppData\Local\Temp\opencode'

if ($Status) {
  Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalPort -in $ProxyPort, $EnginePort } |
    ForEach-Object {
      $p = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue
      Write-Output ('  port {0}  pid {1}  {2}' -f $_.LocalPort, $_.OwningProcess, $p.Name)
    }
  try {
    $m = Invoke-RestMethod -Uri ("http://127.0.0.1:{0}/v1/models" -f $ProxyPort) -TimeoutSec 8
    Write-Output ('  clients path :{0} -> {1}' -f $ProxyPort, $m.data[0].id)
  } catch { Write-Output ('  clients path :{0} NOT ANSWERING' -f $ProxyPort) }
  return
}

Write-Output 'stopping anything already on those ports ...'
Get-Process -Name 'ninfer-kvmem-server' -ErrorAction SilentlyContinue | Stop-Process -Force
Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -match 'rc3_prod|keepalive_proxy' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 7

Write-Output ("starting engine on {0} ..." -f $EnginePort)
Start-Process -FilePath 'powershell' -WindowStyle Hidden -PassThru `
  -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "$ROOT\rc3_prod.ps1",
                  '-Port', "$EnginePort", '-NoLrfu', '-FastPrefill') `
  -RedirectStandardOutput "$ROOT\stack-engine.out" -RedirectStandardError "$ROOT\stack-engine.err" | Out-Null

$ok = $false
for ($i = 0; $i -lt 40; $i++) {
  Start-Sleep -Seconds 3
  try {
    $r = Invoke-RestMethod -Uri ("http://127.0.0.1:{0}/v1/models" -f $EnginePort) -TimeoutSec 4
    if ($r.data) { $ok = $true; break }
  } catch {}
}
if (-not $ok) { Write-Output 'ENGINE NOT READY'; Get-Content "$ROOT\stack-engine.err" -Tail 5; exit 1 }
Write-Output ("engine ready on {0}" -f $EnginePort)

Write-Output ("starting keepalive proxy on {0} -> {1} ..." -f $ProxyPort, $EnginePort)
Start-Process -FilePath 'py' -WindowStyle Hidden -PassThru `
  -ArgumentList @("$ROOT\keepalive_proxy.py", "$ProxyPort", "$EnginePort", "$KeepaliveSeconds") `
  -RedirectStandardOutput "$ROOT\stack-proxy.out" -RedirectStandardError "$ROOT\stack-proxy.err" | Out-Null
Start-Sleep -Seconds 4

try {
  $m = Invoke-RestMethod -Uri ("http://127.0.0.1:{0}/v1/models" -f $ProxyPort) -TimeoutSec 8
  Write-Output ("READY: clients use http://127.0.0.1:{0}/v1  (model {1})" -f $ProxyPort, $m.data[0].id)
} catch {
  Write-Output ("proxy not answering on {0}" -f $ProxyPort)
  Get-Content "$ROOT\stack-proxy.err" -Tail 5
  exit 1
}
Write-Output "DONE"
