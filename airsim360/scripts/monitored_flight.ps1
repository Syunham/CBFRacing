param([Parameter(Mandatory)][ValidatePattern('^[a-z0-9_]+$')][string]$Name,[switch]$PairedCameras,[switch]$CollectCrazyflie,[switch]$HighResolution,[ValidateRange(0,2147483647)][int]$ThreatWeight=350,[ValidatePattern('^[a-z0-9_]+$')][string]$ReplayField,[ValidateSet('pose','trigger')][string]$ProbePeerDepth,[switch]$TimeQuery,[switch]$RtiColdStart,[switch]$CollectTimeHistory,[switch]$TrainTimeHistory,[ValidatePattern('^[a-z0-9_]+$')][string]$EvaluateTimeRun,[ValidatePattern('^[a-zA-Z0-9_./-]+$')][string]$TimeCheckpoint,[ValidatePattern('^[a-z0-9_]+$')][string]$ResumeTimeRun,[ValidateRange(0.001,1.2)][double]$StopSeparation,[switch]$GatePairs,[switch]$TrainFlow,[ValidatePattern('^[a-zA-Z0-9_./-]+$')][string]$FlowPrior,[switch]$FlowGuidance,[switch]$RecoverFrontend,[switch]$DiagnoseLaunch,[switch]$LiveLaunchPanel,[switch]$NeutralLaunch,[ValidatePattern('^[a-z0-9_]+$')][string]$GatePotentialSource,[ValidateRange(0.01,2.5)][double]$PassageGain=1.0,[ValidatePattern('^[a-zA-Z0-9_./-]+$')][string]$LocalFlow,[ValidatePattern('^[a-zA-Z0-9_./-]+$')][string]$PassageHead,[switch]$BendLaunchSwap,[switch]$BendLaunchReferenceSwap,[string]$BendVideoFfmpeg,[ValidatePattern('^[a-zA-Z0-9_./-]+$')][string]$FutureCheckpoint,
    [ValidateSet('smoke','train')][string]$ImplicitExperiment,
    [ValidateSet('P','A','B','C')][string[]]$ImplicitVariants=@('A','B','C'),
    [ValidateSet('short','long')][string]$ImplicitHistory='short',
    [ValidatePattern('^[a-zA-Z0-9_./-]+$')][string]$ImplicitCheckpoint,
    [ValidatePattern('^[a-zA-Z0-9_./-]+$')][string]$VerifyImplicitCheckpoint,
    [switch]$VerifyFlowMLP,
    [ValidateSet('nominal','filtered')][string]$CbfStudy,
    [switch]$OriginalAllocation,
    [switch]$VisualPeer,
    [switch]$DepthOnly,
    [switch]$CaptureStartImages,
    [switch]$RevisedStills,
    [ValidateSet('label','train','external','transition-label','transition-train','guan')][string]$SnapTimeStudy,
    [ValidatePattern('^[a-z0-9_]+$')][string]$SnapDataset,
    [switch]$SnapLowDuty,
    [switch]$OptimizeGateTimes,
    [ValidatePattern('^[a-zA-Z0-9_./-]+$')][string]$SnapTimeNetwork,
    [ValidatePattern('^[a-zA-Z0-9_./-]+$')][string]$GuanTimeNetwork,
    [switch]$ContinueInfeasibleCbf,
    [switch]$ReuseLastMpcc,
    [switch]$Race,
    [switch]$CbfAfterFirstGate,
    [ValidateSet('minimum-snap','guarded-fast-fly')][string]$Planner='minimum-snap',
    [ValidatePattern('^[a-z0-9_]+$')][string]$DspReplay,
    [ValidatePattern('^[a-z0-9_]+$')][string[]]$OccFlowReplay,
    [ValidatePattern('^[a-z0-9_]+$')][string[]]$OccFlowTopdown,
    [ValidatePattern('^[a-z0-9_]+$')][string]$ReplayFlight,
    [string]$CbfReplayFfmpeg)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot
Set-Location $root
$log = Join-Path $root "logs/$Name-telemetry.jsonl"
if ((Test-Path $log) -or (Test-Path "reports/spatial_potential/$Name")) {throw 'Use a fresh trial name'}
if (Get-Process AirSim360-Win64-Shipping -ErrorAction SilentlyContinue) {throw 'Simulator is already running'}
[void][Reflection.Assembly]::LoadFrom('C:/Program Files/SystemTemperatureMonitoring/XKWLib.dll')
$udp = [Net.Sockets.UdpClient]::new()
$udp.Client.ReceiveTimeout = 1400
$ep = [Net.IPEndPoint]::new([Net.IPAddress]::Any,0)
function Get-LatestWhea {
    try {
        Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-WHEA-Logger'} -MaxEvents 1 -ErrorAction Stop
    } catch {
        if ($_.FullyQualifiedErrorId -notlike 'NoMatchingEventsFound,*') {throw}
        [pscustomobject]@{RecordId=0;Id=$null}
    }
}
$baseline = (Get-LatestWhea).RecordId
function Check-Health {
    $packet = [XKWLib.UdpCommand]::new('rundata','').GetCmdBytes()
    [void]$udp.Send($packet,$packet.Length,'127.0.0.1',8890)
    $reply = $udp.Receive([ref]$ep)
    $groups = [XKWLib.UdpCommand]::Analysis($reply).data | ConvertFrom-Json
    $cpu = ($groups | Where-Object type -eq Cpu).data | ConvertFrom-Json
    $ram = ($groups | Where-Object type -eq Ram).data | ConvertFrom-Json
    $temp = [double]$cpu.temperature
    $last = Get-LatestWhea
    [ordered]@{time=(Get-Date -Format o);cpu_c=$temp;cpu_load_pct=$cpu.load;ram_available_gb=$ram.'Available Memory';whea_record=$last.RecordId;whea_id=$last.Id;whea_baseline=$baseline} | ConvertTo-Json -Compress | Add-Content -LiteralPath $log
    if ($temp -le 0 -or $temp -ge 85 -or $last.RecordId -gt $baseline) {throw "Health stop: CPU $temp C, WHEA $($last.RecordId) vs $baseline"}
    if ($ram.'Available Memory' -lt 1) {throw 'Health stop: less than 1 GB available RAM'}
}
function Wait-Checked($child) {
    # Retain the native handle while the child is alive (Windows PowerShell 5.1).
    $null = $child.Handle
    while (!$child.HasExited) {Check-Health; Start-Sleep -Seconds 2}
    $child.WaitForExit()
    if ($child.ExitCode -ne 0) {throw "Child process exited with $($child.ExitCode)"}
    Check-Health
}
$sim = $null
$child = $null
try {
    Check-Health
    foreach ($key in 'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS') {[Environment]::SetEnvironmentVariable($key,'1','Process')}
    if ($SnapTimeStudy) {
        if (!$SnapDataset) {throw 'Provide the synthetic time dataset directory name'}
        $script = if ($SnapTimeStudy -eq 'train') {'scripts.training.train_snap_times'} else {'scripts.training.synthetic_snap_teacher'}
        $jobArgs=@('-u','-m',$script,"reports/spatial_potential/$SnapDataset")
        if ($SnapTimeStudy -like 'transition-*') {$jobArgs=@('-u','-m','scripts.training.snap_transition_study',$SnapTimeStudy,"reports/spatial_potential/$SnapDataset")}
        if ($SnapTimeStudy -eq 'guan') {$jobArgs=@('-u','-m','scripts.planning.minsnap.guan_minjerk',"reports/spatial_potential/$SnapDataset")}
        if ($SnapTimeStudy -eq 'external') {$jobArgs+=@('--start','36','--groups-per-gate','6','--holdout')}
        if ($SnapLowDuty) {$env:SNAP_LOW_DUTY='1'}
        $child=Start-Process '.\.conda-envs\risk-training\python.exe' -ArgumentList $jobArgs -WindowStyle Hidden -RedirectStandardOutput "logs/$Name.log" -RedirectStandardError "logs/$Name.err" -PassThru
        $child.ProcessorAffinity=[IntPtr]0x100
        $child.PriorityClass=if ($SnapLowDuty) {'Idle'} else {'BelowNormal'}
        Wait-Checked $child
        return
    }
    if ($ImplicitExperiment -or $VerifyImplicitCheckpoint -or $VerifyFlowMLP) {
        $out = "reports/spatial_potential/$Name"
        $params = @('-u','-m','scripts.training.train_implicit_rgbd','--output',$out,'--history',$ImplicitHistory,'--variants')+$ImplicitVariants
        if ($VerifyFlowMLP) {
            $params = @('-u','-m','scripts.evaluation.verify_flow_mlp',$out)
        } elseif ($VerifyImplicitCheckpoint) {
            $params = @('-u','-m','scripts.evaluation.check_implicit_mpcc','--checkpoint',$VerifyImplicitCheckpoint,'--output',$out)
        } elseif ($ImplicitExperiment -eq 'smoke') {
            $params += @('--epochs','1','--samples','40','--val-samples','10','--batch-size','1')
        } else {
            $params += @('--device','cuda')
        }
        $child = Start-Process '.\.conda-envs\risk-training\python.exe' -ArgumentList $params -WindowStyle Hidden -RedirectStandardOutput "logs/$Name.log" -RedirectStandardError "logs/$Name.err" -PassThru
        $child.ProcessorAffinity=[IntPtr]0x100
        $child.PriorityClass='Idle'
        Wait-Checked $child
        return
    }
    if ($BendVideoFfmpeg) {
        $exe=(Resolve-Path -LiteralPath $BendVideoFfmpeg).Path
        $params=@('-u','-m','scripts.replay.compose_bendhead_comparison',('"{0}"' -f $exe))
        $child=Start-Process '.\.venv\python.exe' -ArgumentList $params -WindowStyle Hidden -RedirectStandardOutput "logs/$Name.log" -RedirectStandardError "logs/$Name.err" -PassThru
        $child.ProcessorAffinity=[IntPtr]0x300
        $child.PriorityClass='BelowNormal'
        Wait-Checked $child
        return
    }
    if ($BendLaunchSwap -or $BendLaunchReferenceSwap) {
        foreach ($history in 1,2) {
            foreach ($embedding in 1,2) {
                $trial="${Name}_h${history}_v${embedding}"
                $params=@('-u','-m','scripts.replay.replay_bendhead_launch',$history,$embedding,$trial)
                if ($BendLaunchReferenceSwap) {
                    if ($history -eq $embedding) {continue}
                    $params=@('-u','-m','scripts.replay.replay_bendhead_launch',$history,$history,$trial,'--reference',$embedding)
                }
                $child=Start-Process '.\.conda-envs\risk-training\python.exe' -ArgumentList $params -WindowStyle Hidden -RedirectStandardOutput "logs/$trial.log" -RedirectStandardError "logs/$trial.err" -PassThru
                $child.ProcessorAffinity=[IntPtr]0x300
                $child.PriorityClass='BelowNormal'
                Wait-Checked $child
            }
        }
        return
    }
    if ($GatePotentialSource) {
        $params = @('-u','-m','scripts.replay.export_gate_potential',"reports/spatial_potential/$GatePotentialSource",$Name)
        $child = Start-Process '.\.conda-envs\risk-training\python.exe' -ArgumentList $params -WindowStyle Hidden -RedirectStandardOutput "logs/$Name.log" -RedirectStandardError "logs/$Name.err" -PassThru
        $child.ProcessorAffinity=[IntPtr]0x300
        $child.PriorityClass='BelowNormal'
        Wait-Checked $child
        return
    }
    if ($DiagnoseLaunch) {
        $python = '.\.conda-envs\risk-training\python.exe'
        $cases = @('2800_1_learned','0_0_learned','2800_0_learned','0_1_learned','350_1_learned','2800_1_true_motion','2800_1_true_peer')
        $recordings = @('cf21_flow_recover_01','cf21_flow_recover_repeat02')
        $seconds = '1.36'
        if ($NeutralLaunch) {
            $recordings += 'cf21_flow_recover_off01'
            $cases = @('2800_1_learned','2800_1_true_motion','2800_1_true_peer','350_1_true_peer','0_0_learned')
            $seconds = '1.70'
        }
        foreach ($recording in $recordings) {
            $tag = switch ($recording) {
                'cf21_flow_recover_01' {'a'}
                'cf21_flow_recover_repeat02' {'b'}
                'cf21_flow_recover_off01' {'neutral'}
            }
            $selected = $cases
            if ($NeutralLaunch -and $tag -ne 'neutral') {
                $selected = @('2800_1_static_only','2800_1_dynamic_only','2800_1_wide_lane')
                if ($tag -eq 'a') {$selected += '2800_1_true_peer'}
            }
            foreach ($case in $selected) {
                $parts = $case -split '_',3
                $trial = "${Name}_${tag}_$case"
                $cmdArgs = @('-u','-m','scripts.evaluation.diagnose_launch',"reports/spatial_potential/$recording",$trial) + $parts + @('--seconds',$seconds)
                $child = Start-Process $python -ArgumentList $cmdArgs -WindowStyle Hidden -RedirectStandardOutput "logs/$trial.log" -RedirectStandardError "logs/$trial.err" -PassThru
                $child.ProcessorAffinity=[IntPtr]0x300
                $child.PriorityClass='BelowNormal'
                Wait-Checked $child
                Write-Output "Completed $trial"
            }
        }
        return
    }
    if ($TrainFlow) {
        $out = "reports/spatial_potential/$Name"
        $py = '.\.conda-envs\risk-training\python.exe'
        $child = Start-Process $py -ArgumentList @('-u','-m','scripts.training.train_occupancy_flow',$out) -WindowStyle Hidden -RedirectStandardOutput "logs/$Name.log" -RedirectStandardError "logs/$Name.err" -PassThru
        $child.ProcessorAffinity=[IntPtr]0x100
        $child.PriorityClass='Idle'
        Wait-Checked $child
        $child = Start-Process $py -ArgumentList @('-u','-m','scripts.evaluation.evaluate_flow','validation',"$out/validation.json",'--prior',"$out/flow.npz") -WindowStyle Hidden -RedirectStandardOutput "logs/$Name-eval.log" -RedirectStandardError "logs/$Name-eval.err" -PassThru
        $child.ProcessorAffinity=[IntPtr]0x100
        $child.PriorityClass='Idle'
        Wait-Checked $child
        return
    }
    if ($EvaluateTimeRun) {
        $env:CUDA_VISIBLE_DEVICES='-1'
        $evalArgs=@('-u','-m','scripts.evaluation.evaluate_flight_count','--experiment','reports/flight_count_comparison','--manifest','captures/cf21_512_double_50hz/manifest.json','--retrained-run',"reports/spatial_potential/$EvaluateTimeRun",'--original-run','reports/flight_count_comparison/run_8','--device','cpu')
        $child=Start-Process '.\.conda-envs\risk-training\python.exe' -ArgumentList $evalArgs -WindowStyle Hidden -RedirectStandardOutput "logs/$Name.log" -RedirectStandardError "logs/$Name.err" -PassThru
        $child.ProcessorAffinity=[IntPtr]0x100
        $child.PriorityClass='Idle'
        Wait-Checked $child
        return
    }
    if ($TrainTimeHistory -or $ResumeTimeRun) {
        $manifest = 'captures/cf21_512_double_50hz/manifest.json'
        $out = "reports/spatial_potential/$Name"
        $trainArgs = @('-u','-m','scripts.training.train_rgbd_field','--manifest',$manifest,'--output',$out,'--device','cuda')
        if ($GatePairs) {
            if (!$HighResolution) {throw 'Gate pair comparison expects native resolution'}
            $trainArgs+='--gate-pairs'
        }
        if (!$HighResolution) {$trainArgs+='--half-size'}
        $affinity=[IntPtr]0x300
        $priority='BelowNormal'
        if ($ResumeTimeRun) {$trainArgs+=@('--resume',"reports/spatial_potential/$ResumeTimeRun/last.pt",'--total-epochs','30')}
        if ($ResumeTimeRun -or $HighResolution) {
            $trainArgs+=@('--batch-rest-ms','20','--lazy-windows')
            $affinity=[IntPtr]0x100
            $priority='Idle'
        }
        $child = Start-Process '.\.conda-envs\risk-training\python.exe' -ArgumentList $trainArgs -WindowStyle Hidden -RedirectStandardOutput "logs/$Name.log" -RedirectStandardError "logs/$Name.err" -PassThru
        $child.ProcessorAffinity=$affinity
        $child.PriorityClass=$priority
        Wait-Checked $child
        $evalArgs = @('-u','-m','scripts.evaluation.evaluate_flight_count','--experiment','reports/flight_count_comparison','--manifest',$manifest,'--retrained-run',$out,'--original-run','reports/flight_count_comparison/run_8')
        if ($HighResolution) {
            $evalArgs=@('-u','-m','scripts.evaluation.evaluate_resolution','--manifest',$manifest,'--native',$out,'--baseline','reports/spatial_potential/cf21_time8_rgbd_50hz_resume02')
            if ($GatePairs) {
                $evalArgs[-1]='reports/spatial_potential/cf21_time8_rgbd_native512_01'
                $evalArgs+='--sampling-comparison'
            }
        } elseif ($ResumeTimeRun) {
            $env:CUDA_VISIBLE_DEVICES='-1'
            $evalArgs+=@('--device','cpu')
        }
        $child = Start-Process '.\.conda-envs\risk-training\python.exe' -ArgumentList $evalArgs -WindowStyle Hidden -RedirectStandardOutput "logs/$Name-eval.log" -RedirectStandardError "logs/$Name-eval.err" -PassThru
        $child.ProcessorAffinity=$affinity
        $child.PriorityClass=$priority
        Wait-Checked $child
        return
    }
    $env:SAVE_FLIGHT_VIEWS='1'
    $env:SINGLE_FLIGHT_CAMERA=if ($PairedCameras) {'0'} else {'1'}
    $exe = @(Get-ChildItem 'simulator/AirSim360_Pro_FactoryExteriorInteriorBundle' -Filter '*-Win64-Shipping.exe' -Recurse)
    if ($exe.Count -ne 1) {throw 'Expected one simulator executable'}
    $sim = Start-Process $exe[0].FullName -WorkingDirectory $exe[0].DirectoryName -ArgumentList @('-RenderOffscreen','-windowed','-ResX=640','-ResY=360','-nosound','-unattended','-ExecCmds="t.MaxFPS 18,sg.TextureQuality 0,r.Streaming.PoolSize 512,sg.ShadowQuality 0,sg.EffectsQuality 0"',"-settings=`"$root/settings.json`"","-abslog=`"$root/logs/$Name-sim.log`"") -WindowStyle Hidden -PassThru
    $sim.ProcessorAffinity = if ($CbfReplayFfmpeg -and $ReplayFlight) {[IntPtr]0xC0} else {[IntPtr]0xF0}
    $sim.PriorityClass = 'BelowNormal'
    $ready=$false
    for ($i=0;$i -lt 90;$i++) {
        Check-Health
        if ($sim.HasExited) {throw 'Simulator exited'}
        $test=[Net.Sockets.TcpClient]::new()
        try {$test.Connect('127.0.0.1',41451);$ready=$true} catch {} finally {$test.Dispose()}
        if ($ready) {break}
        Start-Sleep 1
    }
    if (!$ready) {throw 'Simulator not ready'}
    if ($Race) {
        $env:SAVE_FLIGHT_VIEWS='0'
        $child=Start-Process '.\.conda-envs\risk-training\python.exe' -ArgumentList @('-u','-m','scripts.control.race',$Name) -WindowStyle Hidden -RedirectStandardOutput "logs/$Name.log" -RedirectStandardError "logs/$Name.err" -PassThru
        $child.ProcessorAffinity=[IntPtr]0x300
        $child.PriorityClass='BelowNormal'
        Wait-Checked $child
        return
    }
    if ($CaptureStartImages -or $RevisedStills) {
        $params=@('-u','-m','scripts.simulation.capture_split_s_stills',"reports/spatial_potential/$Name")
        if ($RevisedStills) {$params+='--revised'}
        $child=Start-Process '.\.venv\python.exe' -ArgumentList $params -WindowStyle Hidden -RedirectStandardOutput "logs/$Name.log" -RedirectStandardError "logs/$Name.err" -PassThru
        $child.ProcessorAffinity=[IntPtr]0x300
        $child.PriorityClass='BelowNormal'
        Wait-Checked $child
        return
    }
    if ($DspReplay) {
        $params=@('-u','-m','scripts.replay.replay_dsp_points',"reports/spatial_potential/$DspReplay","reports/spatial_potential/$Name")
        if ($DepthOnly) {$params+='--depth-only'}
        $child=Start-Process '.\.conda-envs\risk-training\python.exe' -ArgumentList $params -WindowStyle Hidden -RedirectStandardOutput "logs/$Name.log" -RedirectStandardError "logs/$Name.err" -PassThru
        $child.ProcessorAffinity=[IntPtr]0x300
        $child.PriorityClass='BelowNormal'
        Wait-Checked $child
        return
    }
    if ($OccFlowTopdown) {
        $params=@('-u','-m','scripts.replay.plot_occflow_topdown')
        foreach ($run in $OccFlowTopdown) {$params+="reports/spatial_potential/$run"}
        $child=Start-Process '.\.conda-envs\risk-training\python.exe' -ArgumentList $params -WindowStyle Hidden -RedirectStandardOutput "logs/$Name.log" -RedirectStandardError "logs/$Name.err" -PassThru
        $child.ProcessorAffinity=[IntPtr]0x300
        $child.PriorityClass='BelowNormal'
        Wait-Checked $child
        return
    }
    if ($CbfReplayFfmpeg) {
        $exe=(Resolve-Path -LiteralPath $CbfReplayFfmpeg).Path
        $params=@('-u','-m','scripts.replay.export_cbf_replays',('"{0}"' -f $exe))
        if ($ReplayFlight) {$params+=@('--run',$ReplayFlight)}
        if ($OccFlowReplay) {
            $params=@('-u','-m','scripts.replay.export_occflow_chase',('"{0}"' -f $exe))
            foreach ($run in $OccFlowReplay) {$params+="reports/spatial_potential/$run"}
        }
        $child=Start-Process '.\.venv\python.exe' -ArgumentList $params -WindowStyle Hidden -RedirectStandardOutput "logs/$Name.log" -RedirectStandardError "logs/$Name.err" -PassThru
        $child.ProcessorAffinity=if ($ReplayFlight) {[IntPtr]0x100} else {[IntPtr]0x300}
        $child.PriorityClass='BelowNormal'
        Wait-Checked $child
        return
    }
    if ($LiveLaunchPanel) {
        $python = '.\.conda-envs\risk-training\python.exe'
        $env:SAVE_FLIGHT_VIEWS='0'
        $treatments = @('2800_0','350_1','0_1','0_1','350_1','2800_0')
        for ($i=0;$i -lt $treatments.Count;$i++) {
            $case = $treatments[$i]
            $trial = "${Name}_${i}_$case"
            $cmdArgs = @('-u','-m','scripts.evaluation.diagnose_launch','reports/spatial_potential/cf21_flow_recover_01',$trial) + ($case -split '_') + @('learned','--live','--seconds','1.56')
            $child = Start-Process $python -ArgumentList $cmdArgs -WindowStyle Hidden -RedirectStandardOutput "logs/$trial.log" -RedirectStandardError "logs/$trial.err" -PassThru
            $child.ProcessorAffinity=[IntPtr]0x300
            $child.PriorityClass='BelowNormal'
            Wait-Checked $child
            Write-Output "Completed $trial"
        }
        return
    }
    if ($CollectTimeHistory) {
        $child=Start-Process '.\.venv\python.exe' -ArgumentList @('-u','-m','scripts.simulation.collect_crazyflie','captures/cf21_512_double_50hz','--width','512','--geometry','captures/crazyflie21_split_s/geometry.json','--trajectory-stride','1') -WindowStyle Hidden -RedirectStandardOutput "logs/$Name.log" -RedirectStandardError "logs/$Name.err" -PassThru
        $child.ProcessorAffinity=[IntPtr]0x300
        $child.PriorityClass='BelowNormal'
        Wait-Checked $child
        return
    }
    if ($ReplayField) {
        $replayScript = if ($ProbePeerDepth) {'scripts.evaluation.probe_peer_depth'} else {'scripts.replay.replay_field_inputs'}
        $replayArgs = @('-u','-m',$replayScript,"reports/spatial_potential/$ReplayField","reports/spatial_potential/$Name")
        if ($ProbePeerDepth -eq 'trigger') {$replayArgs += '--trigger-check'}
        $child = Start-Process '.\.venv\python.exe' -ArgumentList $replayArgs -WindowStyle Hidden -RedirectStandardOutput "logs/$Name.log" -RedirectStandardError "logs/$Name.err" -PassThru
        $child.ProcessorAffinity=[IntPtr]0x300
        $child.PriorityClass='BelowNormal'
        Wait-Checked $child
        return
    }
    if ($CollectCrazyflie) {
        $data='captures/crazyflie21_split_s'
        $out='reports/spatial_potential/crazyflie21'
        $collection=@('-u','-m','scripts.simulation.collect_crazyflie',$data)
        $compare=@('--paired-manifest','reports/flight_count_comparison/manifest_8.json')
        if ($HighResolution) {
            $data += '_512'; $out += '_512'
            $collection=@('-u','-m','scripts.simulation.collect_crazyflie',$data,'--width','512','--geometry','captures/crazyflie21_split_s/geometry.json')
            $compare=@()
        }
        $child=Start-Process '.\.venv\python.exe' -ArgumentList $collection -WindowStyle Hidden -RedirectStandardOutput "logs/$Name.log" -RedirectStandardError "logs/$Name.err" -PassThru
        $child.ProcessorAffinity=[IntPtr]0x300
        $child.PriorityClass='BelowNormal'
        Wait-Checked $child
        Stop-Process -Id $sim.Id
        $sim.WaitForExit()
        $child=Start-Process '.\.conda-envs\risk-training\python.exe' -ArgumentList (@('-u','-m','scripts.training.train_spatial_potential','--manifest',"$data/manifest.json",'--out',$out,'--device','cuda')+$compare) -WindowStyle Hidden -RedirectStandardOutput "logs/$Name-train.log" -RedirectStandardError "logs/$Name-train.err" -PassThru
        $child.ProcessorAffinity=[IntPtr]0x300
        $child.PriorityClass='BelowNormal'
        Wait-Checked $child
        return
    }
    $weights=@()
    if ($OriginalAllocation) {$weights+='--original-allocation'}
    if ($DepthOnly) {$weights+='--depth-pair'}
    if ($OptimizeGateTimes) {$weights+='--optimize-gate-times'}
    if ($SnapTimeNetwork) {$weights+=@('--snap-time-network',$SnapTimeNetwork)}
    if ($GuanTimeNetwork) {$weights+=@('--guan-time-network',$GuanTimeNetwork)}
    if ($ContinueInfeasibleCbf) {$weights+='--continue-infeasible-cbf'}
    if ($ReuseLastMpcc) {$weights+='--reuse-last-mpcc'}
    if ($CbfAfterFirstGate) {$weights+='--cbf-after-first-gate'}
    if ($VisualPeer) {$weights+='--visual-peer'}
    if ($CbfStudy) {
        $weights+=@('--cbf-study',$CbfStudy)
        $env:SAVE_FLIGHT_VIEWS='0'
    }
    if ($HighResolution -or $ImplicitCheckpoint) {$weights+=@('--checkpoint','reports/spatial_potential/crazyflie21_512/best.pt')}
    if ($PSBoundParameters.ContainsKey('StopSeparation')) {
        $weights+=@('--stop-separation',$StopSeparation.ToString([Globalization.CultureInfo]::InvariantCulture))
    }
    if ($PSBoundParameters.ContainsKey('PassageGain')) {$weights+=@('--passage-gain',$PassageGain.ToString([Globalization.CultureInfo]::InvariantCulture))}
    if ($RecoverFrontend) {$weights+='--recover-frontend'}
    if ($FlowGuidance) {$weights+='--flow-guidance'}
    if ($PassageHead) {$weights+=@('--passage-head',$PassageHead)}
    if ($LocalFlow) {$weights+=@('--local-flow',$LocalFlow)}
    if ($FlowPrior) {$weights+=@('--flow-prior',$FlowPrior)}
    if ($FutureCheckpoint) {$weights+=@('--future-checkpoint',$FutureCheckpoint)}
    if ($ImplicitCheckpoint) {$weights+=@('--implicit-checkpoint',$ImplicitCheckpoint)}
    if ($TimeCheckpoint) {$weights+=@('--time-checkpoint',$TimeCheckpoint)}
    elseif ($TimeQuery) {$weights+=@('--time-checkpoint','reports/flight_count_comparison/run_8/best.pt')}
    if ($RtiColdStart) {$weights+='--rti-cold-start'}
    $child = Start-Process '.\.conda-envs\risk-training\python.exe' -ArgumentList (@('-u','-m','scripts.control.run_spatial_mpcc','--name',$Name,'--separation','1.2',"--$Planner",'--weight',$ThreatWeight.ToString())+$weights) -WindowStyle Hidden -RedirectStandardOutput "logs/$Name.log" -RedirectStandardError "logs/$Name.err" -PassThru
    $child.ProcessorAffinity=[IntPtr]0x300
    $child.PriorityClass='BelowNormal'
    Wait-Checked $child
    if ($CbfStudy) {return}
    $child = Start-Process '.\.venv\python.exe' -ArgumentList @('-u','-m','scripts.replay.flight_movie',$Name) -WindowStyle Hidden -RedirectStandardOutput "logs/$Name-movie.log" -RedirectStandardError "logs/$Name-movie.err" -PassThru
    $child.ProcessorAffinity=[IntPtr]0x300
    $child.PriorityClass='BelowNormal'
    Wait-Checked $child
} finally {
    if ($child -and !$child.HasExited) {taskkill /T /F /PID $child.Id}
    if ($sim -and !$sim.HasExited) {Stop-Process -Id $sim.Id}
    $udp.Close()
}
