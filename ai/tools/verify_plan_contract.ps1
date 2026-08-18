param(
    [string]$Root = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
)

$ErrorActionPreference = 'Stop'
$contractPath = Join-Path $Root 'contracts\forge200_plan_contract.v1.json'
$rosterPath = Join-Path $Root 'contracts\model_roster_200.v1.tsv'
$taskContractPath = Join-Path $Root 'contracts\model_task_contracts_200.v1.tsv'
$candidatePath = Join-Path $Root 'contracts\candidate_pool_244.v1.tsv'
$candidateTaskPath = Join-Path $Root 'contracts\candidate_task_contracts_244.v1.tsv'
$requiredDocs = @(
    'AGENTS.md',
    'docs\CIMC_ICMAT_FORGE200_FINAL_MASTER_PLAN.md',
    'docs\MODEL_ROSTER_200.md',
    'docs\ICMAT_COUNCIL_LLM_RAG_PLAN.md',
    'docs\DUAL_5090_MODEL_FACTORY_AND_ACCEPTANCE.md',
    'docs\POST_MODEL_VERIPROCESS_UPGRADE.md',
    'docs\RESEARCH_BASIS_AND_SCOPE_LOCK.md',
    'contracts\forge200_plan_contract.v1.json',
    'contracts\model_roster_200.v1.tsv',
    'contracts\model_task_contracts_200.v1.tsv',
    'contracts\candidate_pool_244.v1.tsv',
    'contracts\candidate_task_contracts_244.v1.tsv'
)

foreach ($relative in $requiredDocs) {
    $path = Join-Path $Root $relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing required artifact: $relative"
    }
}

$contract = Get-Content -LiteralPath $contractPath -Raw -Encoding UTF8 | ConvertFrom-Json
$rows = Import-Csv -LiteralPath $rosterPath -Delimiter "`t" -Encoding UTF8
$taskRows = Import-Csv -LiteralPath $taskContractPath -Delimiter "`t" -Encoding UTF8
$candidates = Import-Csv -LiteralPath $candidatePath -Delimiter "`t" -Encoding UTF8
$candidateTasks = Import-Csv -LiteralPath $candidateTaskPath -Delimiter "`t" -Encoding UTF8

if ($contract.project -ne 'CIMC') { throw 'Contract project must be CIMC' }
if ($contract.execution_started -ne $false) { throw 'Plan unexpectedly marked executed' }
if ($contract.training_started -ne $false) { throw 'Plan unexpectedly marked trained' }
if ($contract.production_modified -ne $false) { throw 'Plan unexpectedly marks production modified' }
if ($rows.Count -ne 200) { throw "Expected 200 roster rows, got $($rows.Count)" }

$expectedIds = 1..200 | ForEach-Object { 'ICM-{0:D3}' -f $_ }
$actualIds = @($rows.asset_id)
if ((Compare-Object $expectedIds $actualIds).Count -ne 0) { throw 'Roster IDs are not exactly ICM-001..ICM-200' }
if (($actualIds | Sort-Object -Unique).Count -ne 200) { throw 'Duplicate asset_id found' }
if (($rows.name | Sort-Object -Unique).Count -ne 200) { throw 'Duplicate model name found' }
if (($rows | Where-Object { [int]$_.authority -ne 0 }).Count -ne 0) { throw 'A model has nonzero authority' }
if (($rows | Where-Object { $_.asset_id -eq 'ICM-072' }).name -ne 'SinterGraph-PSP') {
    throw 'ICM-072 must be SinterGraph-PSP'
}

$baseline = @($rows | Where-Object { $_.type -like 'BASELINE_*' })
$predictive = @($rows | Where-Object { $_.type -eq 'NEW_PREDICTIVE' })
$generative = @($rows | Where-Object { $_.type -eq 'NEW_GENERATIVE' })
$support = @($rows | Where-Object { $_.type -eq 'NEW_SUPPORT_MODEL' })
$logical = @($rows.logical_family | Sort-Object -Unique)

if ($baseline.Count -ne 30) { throw "Expected 30 baseline assets, got $($baseline.Count)" }
if ($predictive.Count -ne 112) { throw "Expected 112 predictive models, got $($predictive.Count)" }
if ($generative.Count -ne 30) { throw "Expected 30 new generative models, got $($generative.Count)" }
if ($support.Count -ne 28) { throw "Expected 28 support models, got $($support.Count)" }
if ($logical.Count -ne 198) { throw "Expected 198 logical models, got $($logical.Count)" }
if (@($rows | Where-Object { $_.type -like 'NEW_*' -and $_.countable_now -ne '0' }).Count -ne 0) {
    throw 'A planned new model is prematurely countable'
}

if ($contract.target.runtime_assets_max -ne 200 -or
    $contract.target.logical_models_max -ne 198 -or
    $contract.target.new_predictive -ne 112 -or
    $contract.target.new_generative -ne 30 -or
    $contract.target.new_support -ne 28) {
    throw 'Contract counts do not match roster'
}

if ($contract.resource_gates.provisional_new_runtime_code_bytes_target_max -ne 139264 -or
    $contract.resource_gates.provisional_new_runtime_code_bytes_hard_max -ne 142804 -or
    $contract.resource_gates.rom_release_percent_max -ne 88) {
    throw 'Flash growth budget is not the frozen 136 KiB target / 142804 B absolute gate'
}
if ($contract.resource_gates.baseline_map_status -ne 'P0_MERGED_30_MODELS_FATFS_CURRENT_DRIVERS_REQUIRED') {
    throw 'The authoritative merged P0 map is not correctly marked as required'
}
if ($contract.quantization.nano_lm -ne 'W8' -or $contract.quantization.int4_allowed -ne $false) {
    throw 'The nano-LM quantization contract must be W8 with INT4 disabled'
}
if ($contract.target.max_executing_models -ne 1 -or $contract.target.max_resident_packages -ne 2) {
    throw 'Global inference execution/residency limits are not frozen'
}
if ($contract.rag_runtime_topology.separate_resident_guard_pack -ne $false -or
    $contract.rag_runtime_topology.slot_a -ne 'SUPPORT_BUNDLE' -or
    $contract.rag_runtime_topology.slot_b -ne 'ACTIVE_NANO_LM' -or
    $contract.rag_runtime_topology.support_bundle_bytes_max -ne 1048576 -or
    $contract.rag_runtime_topology.support_bundle_model_count -ne 13 -or
    $contract.rag_runtime_topology.active_nano_lm_bytes_max -ne 2097152 -or
    $contract.rag_runtime_topology.resident_packages_max -ne 2 -or
    $contract.rag_runtime_topology.same_query_support_reload_allowed -ne $false -or
    $contract.rag_runtime_topology.same_query_implicit_retry_allowed -ne $false -or
    $contract.rag_runtime_topology.query_total_sd_read_bytes_max -ne 4194304 -or
    $contract.rag_runtime_topology.load_failure_action -ne 'ROLLBACK_AND_REFUSE') {
    throw 'RAG two-resident-package topology is not frozen'
}
$expectedSharedSupport = @(
    'ICM-173', 'ICM-174', 'ICM-175', 'ICM-176', 'ICM-177',
    'ICM-178', 'ICM-179', 'ICM-180', 'ICM-199', 'ICM-200'
)
if (($expectedSharedSupport -join '|') -ne (@($contract.rag_runtime_topology.support_bundle_required_shared_models) -join '|')) {
    throw 'Support bundle shared model membership or order is not frozen'
}
$expectedDomainTriplets = [ordered]@{
    PHOSPHOR = @('ICM-181', 'ICM-187', 'ICM-193')
    FURNACE = @('ICM-182', 'ICM-188', 'ICM-194')
    SEMIMAT = @('ICM-183', 'ICM-189', 'ICM-195')
    METROLOGY = @('ICM-184', 'ICM-190', 'ICM-196')
    PACKAGING = @('ICM-185', 'ICM-191', 'ICM-197')
    FAB_QUALITY = @('ICM-186', 'ICM-192', 'ICM-198')
}
foreach ($domain in $expectedDomainTriplets.Keys) {
    $actualTriplet = @($contract.rag_runtime_topology.support_bundle_domain_triplets.$domain)
    if (($expectedDomainTriplets[$domain] -join '|') -ne ($actualTriplet -join '|')) {
        throw "Support bundle domain triplet is not frozen for $domain"
    }
}
$expectedLoadValidation = @('SCHEMA', 'SHA256', 'GENERATION_COUNTER', 'GOLDEN')
if (($expectedLoadValidation -join '|') -ne (@($contract.rag_runtime_topology.load_validation_order) -join '|')) {
    throw 'Support bundle load validation order is not frozen'
}
$expectedRagStates = @(
    'LOAD_SUPPORT_A', 'ROUTE_ENCODE_RETRIEVE_RERANK', 'LOAD_LM_B', 'GENERATE',
    'UNLOAD_LM_B', 'NLI_QUALITY_A', 'COMMIT_OR_REFUSE', 'ZEROIZE'
)
if (($expectedRagStates -join '|') -ne (@($contract.rag_runtime_topology.state_machine) -join '|')) {
    throw 'RAG state machine is incomplete or reordered'
}

if ($taskRows.Count -ne 200) { throw "Expected 200 task-contract rows, got $($taskRows.Count)" }
if ((Compare-Object $expectedIds @($taskRows.asset_id)).Count -ne 0) {
    throw 'Task-contract IDs are not exactly ICM-001..ICM-200'
}
if (($taskRows.objective_id | Sort-Object -Unique).Count -ne 200) {
    throw 'Task-contract objective_id values are not unique'
}
$requiredTaskFields = @(
    'asset_id', 'objective_id', 'input_contract', 'target_label', 'source_gate',
    'baseline', 'primary_metric', 'parameter_cap', 'consumer', 'overlap_guard',
    'status', 'authority'
)
foreach ($taskRow in $taskRows) {
    foreach ($field in $requiredTaskFields) {
        if ([string]::IsNullOrWhiteSpace([string]$taskRow.$field)) {
            throw "Empty task-contract field $field for $($taskRow.asset_id)"
        }
    }
    if ([int]$taskRow.authority -ne 0) { throw "Nonzero task authority for $($taskRow.asset_id)" }
}
$baselineTasks = @($taskRows | Where-Object { [int]$_.asset_id.Substring(4) -le 30 })
$newTasks = @($taskRows | Where-Object { [int]$_.asset_id.Substring(4) -ge 31 })
if (@($baselineTasks | Where-Object { $_.status -ne 'FROZEN_BASELINE_CONTRACT' }).Count -ne 0) {
    throw 'A baseline task is not frozen'
}
if (@($newTasks | Where-Object { $_.status -ne 'PLAN_ONLY_PRE_FREEZE' }).Count -ne 0) {
    throw 'A new task is not marked plan-only pre-freeze'
}
if (($newTasks.target_label | Sort-Object -Unique).Count -ne 170) {
    throw 'New model target labels are not independently identified'
}
$nanoLmTasks = @($taskRows | Where-Object {
    $id = [int]$_.asset_id.Substring(4)
    $id -ge 143 -and $id -le 172
})
if ($nanoLmTasks.Count -ne 30 -or
    @($nanoLmTasks | Where-Object { $_.parameter_cap -notmatch '_W8_' -or $_.parameter_cap -match 'INT4' }).Count -ne 0) {
    throw 'All 30 new nano-LM task contracts must use the frozen W8 path'
}

if ($candidates.Count -ne 244) { throw "Expected 244 candidate rows, got $($candidates.Count)" }
if (($candidates.candidate_id | Sort-Object -Unique).Count -ne 244) { throw 'Duplicate candidate_id found' }
if (($candidates.name | Sort-Object -Unique).Count -ne 244) { throw 'Duplicate candidate name found' }
if (@($candidates | Where-Object { $_.status -ne 'PRE_REGISTERED_PLAN_ONLY' }).Count -ne 0) {
    throw 'A candidate is not marked plan-only'
}
if (@($candidates | Where-Object { [int]$_.authority -ne 0 }).Count -ne 0) {
    throw 'A candidate has nonzero authority'
}
$candidatePredictive = @($candidates | Where-Object { $_.category -eq 'PREDICTIVE' })
$candidateGenerative = @($candidates | Where-Object { $_.category -eq 'GENERATIVE' })
$candidateSupport = @($candidates | Where-Object { $_.category -eq 'SUPPORT' })
if ($candidatePredictive.Count -ne 148 -or $candidateGenerative.Count -ne 48 -or $candidateSupport.Count -ne 48) {
    throw 'Candidate category counts must be 148 predictive / 48 generative / 48 support'
}
$targetCandidates = @($candidates | Where-Object { $_.target_slot -ne 'SPARE' })
$spareCandidates = @($candidates | Where-Object { $_.target_slot -eq 'SPARE' })
$expectedNewIds = 31..200 | ForEach-Object { 'ICM-{0:D3}' -f $_ }
if ($targetCandidates.Count -ne 170 -or $spareCandidates.Count -ne 74) {
    throw 'Candidate target/spare split must be 170/74'
}
if ((Compare-Object $expectedNewIds @($targetCandidates.target_slot)).Count -ne 0) {
    throw 'Candidate pool does not cover ICM-031..ICM-200 exactly once'
}
foreach ($candidate in $targetCandidates) {
    $rosterRow = $rows | Where-Object { $_.asset_id -eq $candidate.target_slot }
    if ($null -eq $rosterRow -or $rosterRow.name -ne $candidate.name) {
        throw "Candidate/roster mismatch at $($candidate.target_slot)"
    }
}
$modelLoad = @($candidates | Where-Object { $_.name -eq 'ModelLoad-LatencyNet' })
if ($modelLoad.Count -ne 1 -or $modelLoad[0].target_slot -ne 'SPARE') {
    throw 'ModelLoad-LatencyNet must remain a single spare candidate'
}

if ($candidateTasks.Count -ne 244) {
    throw "Expected 244 candidate task-contract rows, got $($candidateTasks.Count)"
}
if ((Compare-Object @($candidates.candidate_id) @($candidateTasks.candidate_id)).Count -ne 0) {
    throw 'Candidate task-contract IDs do not exactly match the candidate pool'
}
if (($candidateTasks.candidate_id | Sort-Object -Unique).Count -ne 244) {
    throw 'Duplicate candidate task-contract candidate_id found'
}
if (($candidateTasks.objective_id | Sort-Object -Unique).Count -ne 244) {
    throw 'Candidate task-contract objective_id values are not unique'
}
if (($candidateTasks.target_label | Sort-Object -Unique).Count -ne 244) {
    throw 'Candidate task-contract target_label values are not unique'
}
$requiredCandidateTaskFields = @(
    'candidate_id', 'objective_id', 'input_contract', 'target_label', 'source_gate',
    'baseline', 'primary_metric', 'parameter_cap', 'consumer', 'replacement_for',
    'status', 'authority'
)
foreach ($candidateTask in $candidateTasks) {
    foreach ($field in $requiredCandidateTaskFields) {
        if ([string]::IsNullOrWhiteSpace([string]$candidateTask.$field)) {
            throw "Empty candidate task-contract field $field for $($candidateTask.candidate_id)"
        }
    }
    if ($candidateTask.status -ne 'PRE_REGISTERED_PLAN_ONLY') {
        throw "Candidate task is not plan-only: $($candidateTask.candidate_id)"
    }
    if ([int]$candidateTask.authority -ne 0) {
        throw "Nonzero candidate task authority for $($candidateTask.candidate_id)"
    }
    $candidate = $candidates | Where-Object { $_.candidate_id -eq $candidateTask.candidate_id }
    if ($candidateTask.replacement_for -ne $candidate.replacement_for) {
        throw "Candidate task replacement mismatch for $($candidateTask.candidate_id)"
    }
}
$candidateTaskById = @{}
foreach ($candidateTask in $candidateTasks) { $candidateTaskById[$candidateTask.candidate_id] = $candidateTask }
foreach ($candidate in $targetCandidates) {
    $formalTask = $taskRows | Where-Object { $_.asset_id -eq $candidate.target_slot }
    $candidateTask = $candidateTaskById[$candidate.candidate_id]
    $mappedFields = @('objective_id', 'input_contract', 'target_label', 'source_gate', 'baseline', 'primary_metric', 'consumer')
    if ($candidate.category -ne 'GENERATIVE') { $mappedFields += 'parameter_cap' }
    foreach ($field in $mappedFields) {
        if ([string]$candidateTask.$field -ne [string]$formalTask.$field) {
            throw "Mapped candidate task field $field disagrees with $($candidate.target_slot) for $($candidate.candidate_id)"
        }
    }
}
$candidateNonGenerativeTasks = @($candidateTasks | Where-Object { $_.candidate_id -notlike 'CAND-G-*' })
if ($candidateNonGenerativeTasks.Count -ne 196 -or
    @($candidateNonGenerativeTasks | Where-Object { $_.parameter_cap -notmatch '(INT8|W8A8)' }).Count -ne 0) {
    throw 'All predictive/support candidates must use the W8A8/INT8 deployment path'
}
$candidateGenerativeTasks = @($candidateTasks | Where-Object { $_.candidate_id -like 'CAND-G-*' })
if ($candidateGenerativeTasks.Count -ne 48 -or
    @($candidateGenerativeTasks | Where-Object { $_.parameter_cap -notmatch '_W8_' -or $_.parameter_cap -match 'INT4' }).Count -ne 0) {
    throw 'All 48 generative candidates must use the frozen W8 path without INT4'
}
foreach ($candidate in $candidateGenerative) {
    $candidateTask = $candidateTaskById[$candidate.candidate_id]
    $slotIdText = if ($candidate.target_slot -ne 'SPARE') {
        $candidate.target_slot
    } else {
        @($candidate.replacement_for -split ',')[0].Trim()
    }
    $targetRow = $rows | Where-Object { $_.asset_id -eq $slotIdText }
    if ($null -eq $targetRow) { throw "Cannot resolve generative slot class for $($candidate.candidate_id)" }
    $slotId = [int]$targetRow.asset_id.Substring(4)
    $capMiB = if ($candidateTask.parameter_cap -match 'MAX_1P8M_') {
        1.8
    } elseif ($candidateTask.parameter_cap -match 'MAX_1P6M_') {
        1.6
    } elseif ($candidateTask.parameter_cap -match 'MAX_1P2M_') {
        1.2
    } elseif ($candidateTask.parameter_cap -match 'MAX_0P8M_') {
        0.8
    } else {
        throw "Unrecognized generative parameter cap for $($candidate.candidate_id)"
    }
    $maxCapMiB = if ($slotId -le 148) { 1.8 } elseif ($slotId -le 168) { 1.2 } else { 0.8 }
    if ($capMiB -gt $maxCapMiB) {
        throw "Generative parameter cap exceeds slot class for $($candidate.candidate_id): $capMiB M > $maxCapMiB M"
    }
}
$foundation16 = @('CAND-G-004', 'CAND-G-005', 'CAND-G-006')
foreach ($candidateId in $foundation16) {
    if ($candidateTaskById[$candidateId].parameter_cap -notmatch 'MAX_1P6M_') {
        throw "$candidateId must retain the 1.6M W8 formal-slot cap"
    }
}
foreach ($candidate in $spareCandidates) {
    $candidateTask = $candidateTaskById[$candidate.candidate_id]
    $allowedSlots = @($candidate.replacement_for -split ',' | ForEach-Object { $_.Trim() })
    if ($allowedSlots.Count -eq 0 -or @($allowedSlots | Where-Object { $_ -notmatch '^ICM-\d{3}$' }).Count -ne 0) {
        throw "Spare candidate lacks an explicit ICM slot whitelist: $($candidate.candidate_id)"
    }
    if (($allowedSlots | Sort-Object -Unique).Count -ne $allowedSlots.Count) {
        throw "Duplicate slot in spare whitelist: $($candidate.candidate_id)"
    }
    $expectedType = switch ($candidate.category) {
        'PREDICTIVE' { 'NEW_PREDICTIVE' }
        'GENERATIVE' { 'NEW_GENERATIVE' }
        'SUPPORT' { 'NEW_SUPPORT_MODEL' }
        default { throw "Unknown candidate category $($candidate.category)" }
    }
    foreach ($slot in $allowedSlots) {
        $slotRow = $rows | Where-Object { $_.asset_id -eq $slot }
        if ($null -eq $slotRow -or $slotRow.type -ne $expectedType) {
            throw "Invalid or cross-category spare whitelist slot $slot for $($candidate.candidate_id)"
        }
    }
    if ($candidateTask.replacement_for -ne ($allowedSlots -join ',')) {
        throw "Candidate task whitelist is not normalized for $($candidate.candidate_id)"
    }
}

$result = [ordered]@{
    status = 'PASS'
    project = $contract.project
    runtime_assets = $rows.Count
    logical_models = $logical.Count
    baseline_assets = $baseline.Count
    new_predictive = $predictive.Count
    new_generative = $generative.Count
    new_support = $support.Count
    task_contracts = $taskRows.Count
    candidate_pool = $candidates.Count
    candidate_task_contracts = $candidateTasks.Count
    candidate_targets = $targetCandidates.Count
    candidate_spares = $spareCandidates.Count
    flash_growth_target_bytes = $contract.resource_gates.provisional_new_runtime_code_bytes_target_max
    flash_growth_hard_bytes = $contract.resource_gates.provisional_new_runtime_code_bytes_hard_max
    nano_lm_quantization = $contract.quantization.nano_lm
    max_executing_models = $contract.target.max_executing_models
    max_resident_packages = $contract.target.max_resident_packages
    execution_started = $contract.execution_started
    training_started = $contract.training_started
    production_modified = $contract.production_modified
}

$result | ConvertTo-Json -Depth 4
