#python ./src/config.py


Write-Host "The value of PYTHONPATH was: $env:PYTHONPATH"
# $paths = python ./src/config.py
$paths = Invoke-Expression -Command "& python ./src/config.py"
# Assign the output to an environment variable
$env:PYTHONPATH = $paths
Write-Host "The current value of PYTHONPATH is: $env:PYTHONPATH"
