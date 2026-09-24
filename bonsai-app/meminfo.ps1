$os = Get-CimInstance Win32_OperatingSystem
$total = [math]::Round($os.TotalVisibleMemorySize/1MB,1)
$free  = [math]::Round($os.FreePhysicalMemory/1MB,1)
Write-Output ("TotalGB=" + $total)
Write-Output ("FreeGB=" + $free)
