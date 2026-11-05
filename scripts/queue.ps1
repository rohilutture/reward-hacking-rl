# Overnight queue: strictly one GPU job at a time. Resumable: finished runs (eval.json present) are skipped.
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = '.\.venv\Scripts\python.exe'
$env:PYTHONUNBUFFERED = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$log = 'runs\queue.log'
New-Item -ItemType Directory -Force runs | Out-Null
function Log($m) { "$(Get-Date -Format 'MM-dd HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $log }

function Run-Experiment($name, $reward, $seed, $extra) {
    if (Test-Path "runs\$name\eval.json") { Log "skip $name (done)"; return }
    Log "train $name start"
    if (-not (Test-Path "runs\$name\adapter\adapter_config.json")) {
        & $py scripts\train_grpo.py --reward $reward --seed $seed --steps 80 --name $name @extra *> "runs\$name.train.log"
    }
    Log "train $name end: adapter $(Test-Path "runs\$name\adapter\adapter_config.json")"
    if (Test-Path "runs\$name\adapter\adapter_config.json") {
        & $py scripts\eval_model.py --name $name --adapter "runs\$name\adapter" --n 300 *> "runs\$name.eval.log"
        Log "eval $name end: $(Test-Path "runs\$name\eval.json")"
    }
}

if (-not (Test-Path 'runs\base\eval.json')) {
    Log 'eval base'; & $py scripts\eval_model.py --name base --n 300 *> 'runs\base.eval.log'
}
if (-not (Test-Path 'runs\difficulty\hard_ids.txt')) {
    Log 'difficulty'; & $py scripts\difficulty.py *> 'runs\difficulty.log'
}
$hard = @('--problem_ids', 'runs\difficulty\hard_ids.txt')

# seed 0 first (all three rewards on the full training set, then the labeled hard-subset stress test)
Run-Experiment 'A_s0' 'A' 0 @()
Run-Experiment 'B_s0' 'B' 0 @()
Run-Experiment 'C_s0' 'C' 0 @()
Run-Experiment 'hardA_s0' 'A' 0 $hard
Run-Experiment 'hardC_s0' 'C' 0 $hard
# seeds 1 and 2
foreach ($s in 1, 2) {
    Run-Experiment "A_s$s" 'A' $s @()
    Run-Experiment "B_s$s" 'B' $s @()
    Run-Experiment "C_s$s" 'C' $s @()
    Run-Experiment "hardA_s$s" 'A' $s $hard
    Run-Experiment "hardC_s$s" 'C' $s $hard
}
Log 'QUEUE DONE'
