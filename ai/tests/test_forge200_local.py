from __future__ import annotations

import csv
import hashlib
import json
import re
import struct
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))


def load_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(relative: str):
    with (ROOT / relative).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class Forge200LocalAcceptance(unittest.TestCase):
    def test_244_task_contracts_are_complete_and_authority_zero(self):
        tasks = read_tsv("contracts/candidate_task_contracts_244.v1.tsv")
        pool = read_tsv("contracts/candidate_pool_244.v1.tsv")
        self.assertEqual(len(tasks), 244)
        self.assertEqual(len(pool), 244)
        self.assertEqual({row["candidate_id"] for row in tasks}, {row["candidate_id"] for row in pool})
        for row in tasks + pool:
            self.assertTrue(all(row.values()))
            self.assertEqual(row["authority"], "0")

    def test_split_and_truth_gates(self):
        split = load_json("data/ledgers/split_ledger.v1.json")
        leakage = load_json("data/ledgers/leakage_audit.v1.json")
        truth = load_json("data/ledgers/truth_ledger.v1.json")
        self.assertEqual(split["status"], "PASS")
        self.assertEqual(leakage["status"], "PASS")
        self.assertEqual(leakage["cross_split_group_overlap_total"], 0)
        self.assertIn("TEACHER_CANDIDATE_TO_GROUND_TRUTH", truth["forbidden_promotions"])

    def test_queue_is_complete_recoverable_and_fail_closed(self):
        queue = load_json("queue/dual_5090_queue.v1.json")
        jobs = queue["jobs"]["GPU_A"] + queue["jobs"]["GPU_B"]
        self.assertEqual(len(queue["jobs"]["GPU_A"]), 148)
        self.assertEqual(len(queue["jobs"]["GPU_B"]), 96)
        self.assertEqual(len(jobs), 244)
        self.assertTrue(queue["no_cross_public_network_ddp"])
        self.assertTrue(all(job["authority"] == 0 for job in jobs))
        self.assertTrue(all(job["max_retries"] == 2 for job in jobs))
        self.assertTrue(all(job["heartbeat_seconds"] == 30 for job in jobs))
        self.assertEqual({job["admission_state"] for job in jobs}, {"ADMITTED", "PRE_GPU_REJECTED_WITH_EVIDENCE"})
        for job in jobs:
            if job["admission_state"] == "ADMITTED":
                dataset = ROOT / job["staged_dataset"]
                self.assertTrue(dataset.is_file())
                self.assertEqual(sha256_file(dataset), job["staged_dataset_sha256"])
            else:
                self.assertNotIn("staged_dataset", job)

    def test_ccby_corpus_is_per_document_licensed_and_leak_free(self):
        ledger = load_json("data/ledgers/ccby_multidomain_corpus.v1.json")
        corpus = ROOT / ledger["corpus_path"]
        self.assertEqual(ledger["status"], "PASS")
        self.assertEqual(ledger["document_count"], 47)
        self.assertEqual(ledger["chunk_count"], 2237)
        self.assertEqual(ledger["document_family_overlap"], 0)
        self.assertEqual(ledger["teacher_outputs"], 0)
        self.assertEqual(sha256_file(corpus), ledger["corpus_sha256"])
        self.assertTrue(all(item["license_metadata_value"].lower() == "cc by" for item in ledger["documents"]))
        for domain, counts in ledger["domain_split_counts"].items():
            self.assertEqual(set(counts), {"train", "validation", "test"}, domain)
            self.assertTrue(all(value >= 16 for value in counts.values()), domain)

    def test_all_candidates_have_final_pre_gpu_disposition(self):
        receipt = load_json("evidence/pre_gpu_disposition_244.v1.json")
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["candidate_count"], 244)
        self.assertEqual(receipt["admitted"], 69)
        self.assertEqual(receipt["pre_gpu_rejected_with_evidence"], 175)
        self.assertEqual(receipt["unresolved"], 0)
        self.assertEqual(receipt["admitted_by_category"], {"PREDICTIVE": 12, "GENERATIVE": 26, "SUPPORT": 31})
        self.assertTrue(all(item["authority"] == 0 for item in receipt["records"]))

    def test_rag_staging_and_new_engine_fixture_packages(self):
        staging = load_json("data/staged/rag_staging_manifest.v1.json")
        self.assertEqual(staging["status"], "PASS_WITH_FAIL_CLOSED_EXPERT_TASKS")
        self.assertEqual(staging["staged_pass"], 57)
        self.assertEqual(staging["staged_blocked"], 0)
        self.assertEqual(staging["teacher_outputs"], 0)
        receipt = load_json("evidence/rag_fixture_dry_run_v1/receipt.json")
        self.assertEqual(receipt["status"], "PASS")
        self.assertFalse(receipt["gpu_used"])
        expected_engines = {"CAND-G-001": 5, "CAND-S-009": 4}
        for record in receipt["records"]:
            package = ROOT / "evidence" / "rag_fixture_dry_run_v1" / record["candidate_id"] / record["package"]["path"]
            magic, schema, header, engine, opset, authority = struct.unpack_from("<4sHHHHB", package.read_bytes(), 0)
            self.assertEqual((magic, schema, header, engine, opset, authority), (b"ICMF", 1, 256, expected_engines[record["candidate_id"]], 1, 0))
            self.assertEqual(record["onnx_checker"], "PASS")

    def test_abi_is_contiguous_and_fixture_packages_parse(self):
        abi = load_json("contracts/model_package_abi.v1.json")
        offset = 0
        for field in abi["fields"]:
            self.assertEqual(field["offset"], offset)
            offset += field["bytes"]
        self.assertEqual(offset, 256)
        receipt = load_json("artifacts/fixture_dry_run/dry_run_receipt.v1.json")
        self.assertEqual(receipt["status"], "PASS")
        for task in receipt["tasks"]:
            package = ROOT / "artifacts" / "fixture_dry_run" / task["candidate_id"] / task["package"]["path"]
            raw = package.read_bytes()
            magic, schema, header, engine, opset, authority = struct.unpack_from("<4sHHHHB", raw, 0)
            self.assertEqual((magic, schema, header, engine, opset, authority), (b"ICMF", 1, 256, 240, 1, 0))
            self.assertEqual(task["onnx"]["status"], "ONNX_CHECKER_PASS")

    def test_interface_freeze_authority_zero(self):
        evidence = load_json("contracts/schemas/evidence_card_v2.schema.json")
        sinter = load_json("contracts/schemas/sintergraph_psp_r1.schema.json")
        chrono = load_json("contracts/schemas/chronospec_r4.events.v1.json")
        self.assertEqual(evidence["properties"]["authority"]["const"], 0)
        self.assertEqual(sinter["properties"]["authority"]["const"], 0)
        self.assertEqual(chrono["authority"], 0)
        self.assertIn("SINTERGRAPH_PREDICTION_FROZEN", {item["name"] for item in chrono["events"]})

    def test_team_board_evidence_is_not_promoted_to_training_truth(self):
        ledger = load_json("data/ledgers/team_hardware_evidence_records.v1.json")
        source = load_json("data/ledgers/source_ledger.v1.json")
        self.assertEqual(ledger["status"], "METADATA_ONLY_NO_TRAINING_LABELS")
        self.assertTrue(all(not row["training_label_present"] for row in ledger["records"]))
        team = next(item for item in source["records"] if item["source_id"] == "cimc_team_hardware_evidence_20260730")
        self.assertFalse(team["training_allowed"])
        self.assertEqual(team["truth_class"], "METADATA_ONLY")

    def test_pipeline_code_has_no_device_or_remote_control_calls(self):
        forbidden = ("paramiko", "netsh", "openocd", "jlink", "pyserial", "serial.serial", "fabric.connection")
        for path in (ROOT / "pipeline").glob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            for token in forbidden:
                self.assertNotIn(token, text, f"{path.name}: {token}")

    def test_corrective_nanolm_architecture_and_staging_contract(self):
        from nanolm_architecture import config_for_candidate, parameter_count

        expected = {
            "CAND-G-001": ("FOUNDATION", 800_000, 1_600_000),
            "CAND-G-007": ("EXPERT", 400_000, 1_200_000),
            "CAND-G-027": ("COLLABORATION", 400_000, 800_000),
        }
        for candidate_id, (family, lower, upper) in expected.items():
            config = config_for_candidate(candidate_id)
            self.assertEqual(config.family, family)
            self.assertEqual(config.quantization, "W8")
            self.assertEqual(config.vocab_size, 2048)
            self.assertEqual(config.context_tokens, 192)
            self.assertEqual(config.max_generation_tokens, 24)
            self.assertGreaterEqual(parameter_count(config), lower)
            self.assertLessEqual(parameter_count(config), upper)

        manifest = load_json("data/staged_nanolm_v2/manifest.v2.json")
        self.assertEqual(manifest["status"], "PASS_CORRECTIVE_DATASETS_TEACHER_PENDING")
        self.assertEqual(manifest["candidate_count"], 26)
        self.assertEqual(manifest["records"], 11874)
        self.assertEqual(manifest["teacher_outputs"], 0)
        self.assertEqual(manifest["teacher_promoted_to_ground_truth"], 0)
        self.assertEqual(manifest["authority_nonzero"], 0)
        for candidate_id in manifest["candidates"]:
            metadata = load_json(f"data/staged_nanolm_v2/{candidate_id}.metadata.json")
            self.assertEqual(metadata["cross_split_group_overlap"], 0)
            self.assertEqual(metadata["authority"], 0)
            self.assertEqual(metadata["teacher_outputs"], 0)
            self.assertFalse(metadata["teacher_may_view_validation_or_test"])
            self.assertTrue(all(value >= 16 for value in metadata["split_counts"].values()))

    def test_corrective_tokenizer_is_train_only_reversible_and_content_addressed(self):
        tokenizer = load_json("contracts/nanolm_tokenizer.v1.json")
        self.assertEqual(tokenizer["status"], "FROZEN_FOR_CORRECTIVE_GPU_TRAINING_BOARD_PENDING")
        self.assertEqual(tokenizer["fit_split"], "train_only")
        self.assertEqual(tokenizer["vocab_size"], 2048)
        self.assertEqual(len(tokenizer["pieces"]), 2048)
        self.assertEqual(tokenizer["special_ids"], {"pad": 0, "bos": 1, "eos": 2})
        self.assertEqual(tokenizer["authority"], 0)
        for source, decoded in tokenizer["roundtrip_fixture"].items():
            self.assertEqual(source, decoded)

    def test_host_closure_v4_is_unique_authority_zero_and_not_board_counted(self):
        closure = load_json("evidence/host_closure.v4.json")
        exact = closure["exact_contract"]["records"]
        extensions = closure["sim_only_extensions"]["records"]
        records = exact + extensions
        self.assertEqual(closure["status"], "HOST_CLOSURE_PARTIAL_RELEASE_FLOOR_NOT_MET_BOARD_PENDING")
        self.assertEqual(len(exact), 78)
        self.assertEqual(closure["exact_contract"]["by_category"], {"G": 25, "P": 25, "S": 28})
        self.assertEqual(len(extensions), 7)
        self.assertEqual(len(records), 85)
        self.assertEqual(len({row["candidate_id"] for row in records}), 85)
        self.assertEqual(len({row["package"]["sha256"] for row in records}), 85)
        self.assertEqual(len({row["package"]["payload_sha256"] for row in records}), 85)
        self.assertTrue(all(row["authority"] == 0 for row in records))
        self.assertTrue(all(not row["board_accepted"] for row in records))
        self.assertTrue(all(not row["countable_model"] for row in records))
        self.assertFalse(closure["release_floor"]["met"])

    def test_host_artifact_verification_v4(self):
        receipt = load_json("evidence/host_artifact_verification.v4.json")
        self.assertEqual(receipt["status"], "PASS_HOST_ONLY_BOARD_PENDING")
        self.assertEqual(receipt["model_count"], 85)
        self.assertEqual(receipt["onnx_full_check_pass"], 85)
        self.assertEqual(receipt["golden_archives_pass"], 85)
        self.assertGreaterEqual(receipt["manifest_files_checked"], 1600)
        self.assertGreaterEqual(receipt["manifest_bytes_checked"], 1_000_000_000)
        self.assertEqual(receipt["authority_nonzero"], 0)
        self.assertEqual(receipt["board_accepted"], 0)

    def test_modelbank_v4_manifest_and_1000_swap_dry_run(self):
        build = load_json("evidence/modelbank_build.v4.json")
        dry_run = load_json("evidence/modelbank_host_dry_run.v4.json")
        self.assertEqual(build["status"], "HOST_MODELBANK_BUILT_BOARD_PENDING")
        self.assertEqual(build["model_count"], 85)
        self.assertEqual(build["exact_count"], 78)
        self.assertEqual(build["sim_only_extension_count"], 7)
        self.assertEqual(build["authority_nonzero"], 0)
        self.assertEqual(build["board_actions"], 0)
        bank = ROOT / dry_run["modelbank"]
        manifest = bank / "MANIFEST.v4.json"
        self.assertTrue(manifest.is_file())
        self.assertEqual(sha256_file(manifest), build["manifest_sha256"])
        self.assertEqual(dry_run["status"], "PASS")
        self.assertEqual(dry_run["successful_swaps"], 1000)
        self.assertEqual(dry_run["model_count"], 85)
        self.assertEqual(len(dry_run["fault_injections"]), 6)
        self.assertTrue(all(dry_run["invariants"].values()))
        self.assertEqual(dry_run["authority_nonzero"], 0)
        self.assertEqual(dry_run["new_models_board_accepted"], 0)
        self.assertEqual(dry_run["countable_models"], 0)

    def test_modelbank_v5_is_unique_evidence_indexed_and_board_pending(self):
        build = load_json("evidence/modelbank_build.v5.json")
        dry_run = load_json("evidence/modelbank_host_dry_run.v5.json")
        bank = ROOT / "releases/forge200-host-modelbank-v5-20260803"
        catalog_a = json.loads((bank / "catalog_A.json").read_text(encoding="utf-8"))
        catalog_b = json.loads((bank / "catalog_B.json").read_text(encoding="utf-8"))
        evidence_index = json.loads((bank / "EVIDENCE_INDEX.v5.json").read_text(encoding="utf-8"))
        self.assertEqual(build["model_count"], 85)
        self.assertEqual(build["exact_count"], 78)
        self.assertEqual(build["sim_only_extension_count"], 7)
        self.assertEqual(build["package_sha256_collision_count"], 0)
        self.assertEqual(catalog_a["models"], catalog_b["models"])
        package_hashes = [item["files"]["model.icmf"]["sha256"] for item in catalog_a["models"]]
        self.assertEqual(len(package_hashes), len(set(package_hashes)))
        self.assertEqual(evidence_index["authority"], 0)
        self.assertFalse(evidence_index["board_accepted"])
        for item in evidence_index["records"]:
            self.assertEqual(sha256_file(ROOT / item["path"]), item["sha256"])
        self.assertEqual(dry_run["status"], "PASS")
        self.assertEqual(dry_run["model_count"], 85)
        self.assertEqual(dry_run["successful_swaps"], 1000)
        self.assertEqual(len(dry_run["fault_injections"]), 6)
        self.assertEqual(dry_run["authority_nonzero"], 0)
        self.assertEqual(dry_run["new_models_board_accepted"], 0)
        self.assertEqual(dry_run["countable_models"], 0)

    def test_kaggle_p107_reject_and_p122_exact_admission_are_source_bound(self):
        binding = load_json("contracts/kaggle_p122_solder_fatigue_binding.v1.json")
        staging = load_json("evidence/kaggle_p122_exact_staging.v1.json")
        receipt = load_json("evidence/kaggle_p107_p122_source_contract_audit.v1.json")
        self.assertEqual(
            receipt["status"],
            "SOURCE_LICENSE_AND_HASH_PASS_P107_EXACT_REJECTED_P122_EXACT_ADMITTED",
        )
        self.assertEqual(receipt["p107"]["official_repository_entries"], 3)
        self.assertEqual(receipt["p107"]["official_repository_raster_images"], 0)
        self.assertEqual(receipt["p107"]["official_repository_mask_files"], 0)
        self.assertEqual(receipt["p107"]["kaggle_mask_filename_hits"], 0)
        self.assertFalse(receipt["p107"]["physical_pixel_scale_bound"])
        self.assertEqual(receipt["p107"]["training_actions"], 0)
        self.assertEqual(receipt["p122"]["records"], 1531)
        self.assertEqual(receipt["p122"]["events"], 1032)
        self.assertTrue(receipt["p122"]["training_authorized"])
        self.assertEqual(receipt["p122"]["cross_split_family_overlap"], 0)
        self.assertEqual(receipt["p122"]["cross_split_unit_overlap"], 0)
        self.assertEqual(staging["status"], "PASS_EXACT_SOURCE_LABEL_SPLIT_TRAINING_AUTHORIZED")
        self.assertFalse(staging["future_history_in_inputs"])
        self.assertFalse(staging["paper_group_mean_as_record_truth"])
        self.assertEqual(binding["task_contract"]["authority"], 0)
        self.assertFalse(binding["promotion_boundary"]["P107_mask_training_authorized"])
        for item in receipt["artifacts"].values():
            self.assertEqual(sha256_file(ROOT / item["path"]), item["sha256"])

    def test_kaggle_p122_frozen_test_rejection_is_preserved_and_not_counted(self):
        freeze = load_json("artifacts/local4050_p122_exact_v1/CAND-P-122/baseline_selection_frozen_before_test.json")
        test_once = load_json("artifacts/local4050_p122_exact_v1/CAND-P-122/frozen_test_evaluation.v1.json")
        promotion = load_json("artifacts/local4050_p122_exact_v1/CAND-P-122/promotion_receipt.json")
        closure = load_json("evidence/kaggle_p122_exact_closure.v1.json")
        self.assertFalse(freeze["test_labels_or_metrics_read_during_selection"])
        self.assertEqual(freeze["selection_split"], "validation_only")
        self.assertEqual(test_once["status"], "EVALUATED_ONCE_AFTER_VALIDATION_FREEZE")
        self.assertFalse(test_once["test_used_for_hyperparameter_or_seed_selection"])
        self.assertFalse(test_once["aggregate_mean_gate"])
        self.assertLess(test_once["aggregate"]["mean"], test_once["baseline_test"]["primary_composite"])
        self.assertEqual(promotion["status"], "HOST_GPU_REJECTED_CONTRACT_BASELINE")
        self.assertFalse(promotion["host_contract_pass"])
        self.assertEqual(promotion["authority"], 0)
        self.assertFalse(promotion["board_accepted"])
        self.assertFalse(promotion["countable_model"])
        self.assertEqual(closure["status"], "PARTIAL")
        self.assertEqual(closure["authority_nonzero"], 0)
        self.assertEqual(closure["board_actions"], 0)
        self.assertEqual(sha256_file(ROOT / closure["promotion_receipt"]["path"]), closure["promotion_receipt"]["sha256"])

    def test_interface_freeze_conformance_and_mutations(self):
        receipt = load_json("evidence/interface_freeze_verification.v2.json")
        self.assertEqual(receipt["status"], "PASS_HOST_CONFORMANCE_AND_MUTATION_BOARD_PENDING")
        self.assertEqual(receipt["case_count"], 23)
        self.assertEqual(receipt["cases_passed_as_expected"], 23)
        self.assertEqual(receipt["authority_nonzero"], 0)
        self.assertFalse(receipt["board_accepted"])
        expected_rejections = {item["name"] for item in receipt["cases"] if item["expected"] == "REJECT"}
        self.assertIn("sintergraph_same_run_xrd", expected_rejections)
        self.assertIn("sintergraph_future_time", expected_rejections)
        self.assertIn("commit_without_golden", expected_rejections)

    def test_firmware_adapter_compiles_for_cortex_m7_without_control_symbols(self):
        receipt = load_json("evidence/firmware_adapter_host_compile.v4.json")
        self.assertEqual(receipt["status"], "PASS_ARMCLANG_CORTEX_M7_BOARD_PENDING_NOT_IN_PRODUCTION_TARGET")
        self.assertEqual(len(receipt["objects"]), 2)
        self.assertTrue(all(item["warnings"] == 0 and item["errors"] == 0 for item in receipt["objects"]))
        self.assertTrue(all(receipt["invariants"].values()))
        self.assertEqual(receipt["authority"], 0)
        self.assertEqual(receipt["production_files_modified"], 0)
        self.assertEqual(receipt["board_actions"], 0)

    def test_unified_staging_preserves_frozen_fallback_and_board_boundary(self):
        receipt = load_json("evidence/unified_staging.v4.json")
        self.assertEqual(receipt["status"], "HOST_UNIFIED_STAGING_PARTIAL_RELEASE_FLOOR_BLOCKED_BOARD_PENDING")
        self.assertEqual(receipt["frozen_initial_baseline"]["assets"], 30)
        self.assertEqual(receipt["frozen_initial_baseline"]["logical_models"], 28)
        self.assertTrue(receipt["frozen_initial_baseline"]["fallback_preserved"])
        self.assertEqual(receipt["new_host_modelbank"]["host_qualified"], 85)
        self.assertEqual(receipt["new_host_modelbank"]["board_accepted"], 0)
        self.assertEqual(receipt["combined_projection"]["assets_if_all_new_host_models_later_pass_board"], 115)
        self.assertFalse(receipt["combined_projection"]["floor_met"])
        self.assertEqual(receipt["production_files_modified"], 0)
        self.assertEqual(receipt["sd_or_board_actions"], 0)

    def test_release_gap_audit_covers_all_244_without_count_gaming(self):
        receipt = load_json("evidence/release_gap_audit.v4.json")
        self.assertEqual(receipt["candidate_universe"], 244)
        self.assertEqual(len(receipt["records"]), 244)
        self.assertEqual(receipt["host_exact_source_bound"]["total"], 78)
        self.assertEqual(receipt["host_sim_only_extensions"]["total"], 7)
        self.assertEqual(receipt["host_total_including_sim_only"], 85)
        self.assertEqual(receipt["host_exact_source_bound"]["minimum_release_shortfall"], 42)
        self.assertEqual(receipt["minimum_release_shortfall_including_sim_only"], 35)
        self.assertTrue(all(row["authority"] == 0 for row in receipt["records"]))
        self.assertTrue(all(not row["board_accepted"] for row in receipt["records"]))
        self.assertTrue(all(not row["countable_new_model"] for row in receipt["records"]))
        self.assertFalse(receipt["additional_gpu_training_productive_without_new_exact_data"])

    def test_exact_data_intake_preserves_contracts_and_prohibits_proxy_truth(self):
        receipt = load_json("evidence/exact_data_intake.v4.json")
        self.assertEqual(receipt["status"], "INTAKE_SPEC_READY_EXACT_DATA_NOT_PRESENT")
        self.assertEqual(receipt["unresolved_or_non_exact_tasks"], 166)
        self.assertEqual(receipt["minimum_additional_exact_host_passes_required"], 42)
        self.assertEqual(receipt["target_additional_exact_host_passes_required"], 92)
        self.assertEqual(receipt["acceptance_boundary"]["authority"], 0)
        self.assertTrue(receipt["acceptance_boundary"]["training_and_frozen_test_evaluation_still_required"])
        self.assertTrue(all(len(row["required_proof"]) >= 7 for row in receipt["records"]))
        self.assertTrue(all("teacher_or_API_output_as_ground_truth" in row["prohibited_substitutions"] for row in receipt["records"]))

    def test_nist_p099_preflight_freezes_source_and_split_without_promotion(self):
        binding = load_json("contracts/nist_p099_contract_binding.v1.json")
        receipt = load_json("evidence/nist_p099_source_contract_audit.v1.json")
        split = load_json("data/splits/nist_p099_noise_group_split.v1.json")
        self.assertEqual(binding["candidate_id"], "CAND-P-099")
        self.assertEqual(binding["promotion_boundary"]["authority"], 0)
        self.assertFalse(binding["promotion_boundary"]["countable_model"])
        self.assertIn("DICE-COEFFICIENT", binding["input_binding"]["forbidden_mask_or_target_derived_fields"])
        self.assertEqual(receipt["status"], "PREFLIGHT_PASS_REQUIRED_RAW_IMAGE_DOWNLOAD_PENDING")
        self.assertEqual(receipt["source_grid"]["records"], 567)
        self.assertEqual(receipt["source_grid"]["noise_groups"], 27)
        self.assertEqual(receipt["source_grid"]["contrast_levels"], 21)
        self.assertEqual(receipt["source_grid"]["shared_input_field_mismatches"], 0)
        self.assertFalse(receipt["required_training_payloads_ready"])
        self.assertEqual(receipt["training_actions"], 0)
        self.assertEqual(receipt["test_evaluation_actions"], 0)
        self.assertFalse(receipt["host_promoted"])
        self.assertEqual(split["status"], "FROZEN_BEFORE_TRAINING_TEST_LABELS_NOT_EVALUATED")
        self.assertEqual(split["group_counts"], {"train": 19, "validation": 4, "test": 4})
        self.assertEqual(split["record_counts"], {"train": 399, "validation": 84, "test": 84})
        self.assertEqual(split["cross_split_group_overlap"], 0)
        self.assertFalse(split["test_labels_read_for_split_selection"])
        self.assertEqual(len(split["records"]), 567)
        for table in binding["official_merged_tables"]:
            self.assertEqual(sha256_file(ROOT / table["path"]), table["sha256"])

    def test_mendeley_p101_source_is_verified_without_counting_correlated_curve_points(self):
        binding = load_json("contracts/mendeley_p101_contract_binding.v1.json")
        receipt = load_json("evidence/mendeley_p101_source_contract_audit.v1.json")
        self.assertEqual(binding["candidate_id"], "CAND-P-101")
        self.assertEqual(binding["source"]["license"], "CC BY 4.0")
        self.assertEqual(binding["observed_experiment_design"]["independent_run_count"], 8)
        self.assertEqual(binding["observed_experiment_design"]["published_kinetic_parameter_vectors"], 1)
        self.assertFalse(binding["exact_rejection"]["training_allowed_for_exact_p101"])
        self.assertEqual(receipt["status"], "SOURCE_AND_LICENSE_VERIFIED_EXACT_CONTRACT_REJECTED")
        self.assertEqual(receipt["observed"]["unique_run_serials"], 8)
        self.assertEqual(receipt["observed"]["curve_rows_total"], 27004)
        self.assertFalse(receipt["observed"]["curve_points_treated_as_independent_experiments"])
        self.assertEqual(receipt["training_actions"], 0)
        self.assertFalse(receipt["host_promoted"])
        self.assertEqual(receipt["authority"], 0)
        self.assertFalse(receipt["board_accepted"])
        self.assertFalse(receipt["countable_model"])
        for item in binding["artifacts"].values():
            self.assertEqual(sha256_file(ROOT / item["path"]), item["sha256"])

    def test_mendeley_p105_afm_source_does_not_turn_input_derived_roughness_into_truth(self):
        receipt = load_json("evidence/mendeley_p105_afm_source_contract_audit.v1.json")
        self.assertEqual(
            receipt["status"],
            "SOURCE_LICENSE_AND_HASH_PASS_P105_EXACT_REJECTED_BASELINE_BOUNDARY",
        )
        self.assertEqual(receipt["observed"]["scan_files"], 24)
        self.assertEqual(receipt["observed"]["sample_families"], 12)
        self.assertFalse(receipt["observed"]["curve_or_pixel_rows_treated_as_independent_samples"])
        self.assertEqual(receipt["observed"]["independent_measured_Ra_Rq_peak_density_labels"], 0)
        self.assertFalse(receipt["observed"]["input_derived_labels_may_bypass_frozen_baseline"])
        self.assertTrue(receipt["candidate_disposition"]["status"].startswith("EXACT_REJECTED"))
        self.assertEqual(receipt["training_actions"], 0)
        self.assertEqual(receipt["host_promotions"], 0)
        self.assertEqual(receipt["authority"], 0)
        self.assertFalse(receipt["board_accepted"])
        self.assertFalse(receipt["countable_model"])

    def test_phm2016_p089_semantic_match_does_not_bypass_payload_or_license(self):
        receipt = load_json("evidence/phm2016_p089_source_preflight.v1.json")
        self.assertEqual(receipt["candidate_id"], "CAND-P-089")
        self.assertTrue(receipt["semantic_match"]["input_contract_match"])
        self.assertTrue(receipt["semantic_match"]["target_contract_match"])
        self.assertFalse(receipt["semantic_match"]["curve_rows_may_be_treated_as_independent_wafers"])
        self.assertFalse(receipt["source"]["explicit_reusable_data_license_found"])
        self.assertFalse(receipt["source"]["official_payload_materialized_and_hashed"])
        self.assertEqual(receipt["training_actions"], 0)
        self.assertEqual(receipt["host_promotions"], 0)
        self.assertEqual(receipt["authority"], 0)
        self.assertFalse(receipt["board_accepted"])
        self.assertFalse(receipt["countable_model"])

    def test_mendeley_p095_source_is_verified_without_inventing_missing_images_or_taxonomy(self):
        binding = load_json("contracts/mendeley_p095_contract_binding.v1.json")
        receipt = load_json("evidence/mendeley_p095_source_contract_audit.v1.json")
        self.assertEqual(binding["candidate_id"], "CAND-P-095")
        self.assertEqual(binding["source"]["license"], "CC BY 4.0")
        self.assertEqual(binding["observed_source"]["public_api_files"], 3)
        self.assertEqual(binding["observed_source"]["public_api_raster_image_files"], 0)
        self.assertEqual(binding["observed_source"]["matlab_session_external_tiff_paths"], 49)
        self.assertFalse(binding["observed_source"]["authoritative_taxonomy_mapping_supplied"])
        self.assertFalse(binding["exact_rejection"]["training_allowed_for_exact_p095"])
        self.assertEqual(
            receipt["status"],
            "SOURCE_LICENSE_AND_BOX_LABELS_VERIFIED_EXACT_INPUT_AND_TAXONOMY_REJECTED",
        )
        self.assertEqual(receipt["observed"]["downloaded_raster_image_count"], 0)
        self.assertFalse(receipt["observed"]["leakage_safe_split_materialized"])
        self.assertEqual(receipt["training_actions"], 0)
        self.assertFalse(receipt["host_promoted"])
        self.assertEqual(receipt["authority"], 0)
        self.assertFalse(receipt["board_accepted"])
        self.assertFalse(receipt["countable_model"])
        for item in binding["artifacts"].values():
            self.assertEqual(sha256_file(ROOT / item["path"]), item["sha256"])

    def test_zenodo_batio3_source_is_verified_without_relabeling_thermal_expansion_as_sintering(self):
        binding = load_json("contracts/zenodo_batio3_rapid_sintering_contract_binding.v1.json")
        receipt = load_json("evidence/zenodo_batio3_rapid_sintering_source_contract_audit.v1.json")
        self.assertEqual(binding["source_id"], "ZENODO-18233071")
        self.assertEqual(binding["source"]["license"], "CC BY 4.0")
        self.assertEqual(binding["expected_inventory"]["xrd_patterns"], 19)
        self.assertEqual(binding["expected_inventory"]["sem_tiff_files"], 122)
        self.assertLess(
            binding["expected_inventory"]["dilatometry_temperature_max_c"],
            binding["article_facts_visually_verified"]["sintering_temperature_range_c"][0],
        )
        self.assertEqual(
            {item["candidate_id"] for item in binding["candidate_dispositions"]},
            {"CAND-P-049", "CAND-P-050", "CAND-P-051", "CAND-P-066", "CAND-P-085"},
        )
        self.assertTrue(all(item["status"] == "EXACT_REJECTED" for item in binding["candidate_dispositions"]))
        self.assertEqual(
            receipt["status"],
            "SOURCE_AND_LICENSE_VERIFIED_FIVE_EXACT_CONTRACTS_REJECTED",
        )
        self.assertEqual(receipt["observed"]["xrd_patterns"], 19)
        self.assertEqual(receipt["observed"]["sem_tiff_files"], 122)
        self.assertEqual(receipt["training_actions"], 0)
        self.assertEqual(receipt["host_promotions"], 0)
        self.assertEqual(receipt["authority"], 0)
        self.assertFalse(receipt["board_accepted"])
        self.assertFalse(receipt["countable_model"])
        for item in binding["artifacts"].values():
            self.assertEqual(sha256_file(ROOT / item["path"]), item["sha256"])

    def test_mendeley_sintering_curves_are_not_recast_as_missing_record_labels(self):
        binding = load_json("contracts/mendeley_sintering_w4_contract_binding.v1.json")
        receipt = load_json("evidence/mendeley_sintering_w4_source_contract_audit.v1.json")
        self.assertEqual(binding["source_id"], "MENDELEY-W4N4JDCGCV-V1")
        self.assertEqual(binding["source"]["license"], "CC BY 4.0")
        self.assertEqual(binding["expected_inventory"]["constant_heating_rate_runs"], 5)
        self.assertEqual(binding["expected_inventory"]["isothermal_runs"], 4)
        self.assertEqual(binding["expected_inventory"]["material_families"], 1)
        self.assertEqual(
            {item["candidate_id"] for item in binding["candidate_dispositions"]},
            {"CAND-P-049", "CAND-P-050", "CAND-P-051", "CAND-P-066", "CAND-P-082", "CAND-P-085"},
        )
        self.assertTrue(all(item["status"] == "EXACT_REJECTED" for item in binding["candidate_dispositions"]))
        self.assertEqual(receipt["status"], "SOURCE_AND_LICENSE_VERIFIED_SIX_EXACT_CONTRACTS_REJECTED")
        self.assertFalse(receipt["observed"]["curve_points_treated_as_independent_experiments"])
        self.assertEqual(receipt["observed"]["activation_energy_record_labels"], 0)
        self.assertEqual(receipt["training_actions"], 0)
        self.assertEqual(receipt["host_promotions"], 0)
        self.assertEqual(receipt["authority"], 0)
        self.assertFalse(receipt["board_accepted"])
        self.assertFalse(receipt["countable_model"])
        for item in binding["artifacts"].values():
            self.assertEqual(sha256_file(ROOT / item["path"]), item["sha256"])

    def test_pmc_phosphor_supplements_are_only_source_inventory_until_exact_binding(self):
        receipt = load_json("evidence/pmc_phosphor_supplement_inventory.v1.json")
        self.assertEqual(
            receipt["status"],
            "SOURCE_LICENSE_AND_SUPPLEMENT_INVENTORY_PASS_CONTRACT_BINDING_PENDING",
        )
        self.assertEqual(receipt["source_family_count"], 14)
        self.assertEqual(len({item["pmcid"] for item in receipt["records"]}), 14)
        self.assertTrue(all("cc by" in item["license"].lower() for item in receipt["records"]))
        self.assertTrue(all(not item["training_label_authorized"] for item in receipt["records"]))
        self.assertEqual(receipt["task_contract_bindings"], 0)
        self.assertEqual(receipt["training_actions"], 0)
        self.assertEqual(receipt["host_promotions"], 0)
        self.assertEqual(receipt["authority"], 0)
        self.assertFalse(receipt["board_accepted"])
        self.assertFalse(receipt["countable_model"])
        for item in receipt["records"]:
            self.assertEqual(sha256_file(ROOT / item["metadata"]["path"]), item["metadata"]["sha256"])
            self.assertEqual(sha256_file(ROOT / item["jats_xml"]["path"]), item["jats_xml"]["sha256"])
            self.assertEqual(sha256_file(ROOT / item["outer_archive"]["path"]), item["outer_archive"]["sha256"])

    def test_pmc13157481_nir_supplement_is_not_overclaimed_as_exact_training_truth(self):
        receipt = load_json("evidence/pmc13157481_nir_source_contract_audit.v1.json")
        self.assertEqual(
            receipt["status"],
            "SOURCE_LICENSE_AND_HASH_PASS_FIVE_EXACT_CONTRACTS_REJECTED",
        )
        self.assertEqual(receipt["source"]["pmcid"], "PMC13157481")
        self.assertEqual(receipt["source"]["doi"], "10.1038/s41377-026-02284-8")
        self.assertEqual(receipt["structural_review"]["docx_table_shapes"], [[7, 5], [15, 7], [8, 7]])
        self.assertFalse(receipt["structural_review"]["libreoffice_visual_render_available"])
        self.assertEqual(
            {item["candidate_id"] for item in receipt["candidate_dispositions"]},
            {"CAND-P-061", "CAND-P-062", "CAND-P-079", "CAND-P-080", "CAND-P-081"},
        )
        self.assertTrue(all(item["status"].startswith("EXACT_REJECTED") for item in receipt["candidate_dispositions"]))
        self.assertEqual(receipt["task_contract_bindings_created"], 0)
        self.assertEqual(receipt["training_actions"], 0)
        self.assertEqual(receipt["host_promotions"], 0)
        self.assertEqual(receipt["authority"], 0)
        self.assertFalse(receipt["board_accepted"])
        self.assertFalse(receipt["countable_model"])
        source = receipt["source"]["supplement_docx"]
        self.assertEqual(sha256_file(ROOT / source["path"]), source["sha256"])

    def test_pmc_phosphor_table_index_is_family_split_and_not_prematurely_authorized(self):
        index = load_json("data/ledgers/pmc_phosphor_jats_table_index.v1.json")
        receipt = load_json("evidence/pmc_phosphor_jats_table_index_receipt.v1.json")
        self.assertEqual(index["status"], "INDEXED_CONTRACT_BINDING_PENDING")
        self.assertEqual(index["totals"]["families"], 186)
        self.assertEqual(index["totals"]["families_with_tables"], 119)
        self.assertEqual(index["totals"]["tables"], 360)
        self.assertEqual(len({item["pmcid"] for item in index["families"]}), 186)
        self.assertTrue(all(not item["record_truth_authorized"] for item in index["tables"]))
        self.assertEqual(receipt["status"], "PASS_INDEX_ONLY_NO_TASK_LABELS_AUTHORIZED")
        self.assertEqual(receipt["index"]["sha256"], sha256_file(ROOT / receipt["index"]["path"]))
        self.assertEqual(receipt["task_contract_bindings"], 0)
        self.assertEqual(receipt["training_actions"], 0)
        self.assertEqual(receipt["host_promotions"], 0)
        self.assertEqual(receipt["authority"], 0)
        self.assertFalse(receipt["board_accepted"])
        self.assertFalse(receipt["countable_model"])

    def test_ccby_multidomain_table_index_is_discovery_only_and_family_split(self):
        index = load_json("data/ledgers/ccby_multidomain_jats_table_index.v1.json")
        receipt = load_json("evidence/ccby_multidomain_jats_table_index_receipt.v1.json")
        self.assertEqual(index["status"], "INDEXED_CONTRACT_BINDING_PENDING")
        self.assertEqual(index["totals"]["families"], 1080)
        self.assertEqual(index["totals"]["families_with_tables"], 797)
        self.assertEqual(index["totals"]["tables"], 2701)
        self.assertFalse(index["keyword_hits_are_labels"])
        self.assertEqual(len({item["pmcid"] for item in index["families"]}), 1080)
        self.assertTrue(all(not item["record_truth_authorized"] for item in index["tables"]))
        self.assertTrue(all(item["source_scope_status"] == "MANUAL_IN_STUDY_RECORD_AUDIT_REQUIRED" for item in index["tables"]))
        self.assertEqual(receipt["status"], "PASS_INDEX_ONLY_NO_TASK_LABELS_AUTHORIZED")
        self.assertEqual(receipt["index"]["sha256"], sha256_file(ROOT / receipt["index"]["path"]))
        self.assertEqual(receipt["task_contract_bindings"], 0)
        self.assertEqual(receipt["training_actions"], 0)
        self.assertEqual(receipt["host_promotions"], 0)
        self.assertEqual(receipt["authority"], 0)
        self.assertFalse(receipt["board_accepted"])
        self.assertFalse(receipt["countable_model"])

    def test_mendeley_sic_plunger_source_is_hash_verified_before_contract_binding(self):
        receipt = load_json("data/sources/mendeley_sic_plunger_v1/download_receipt.v1.json")
        self.assertEqual(
            receipt["status"],
            "SOURCE_LICENSE_AND_PAYLOAD_HASH_PASS_CONTRACT_BINDING_PENDING",
        )
        self.assertEqual(receipt["doi"], "10.17632/nknvz6gy6k.1")
        self.assertEqual(receipt["license"], "CC BY 4.0")
        self.assertEqual(len(receipt["verified_files"]), 6)
        self.assertEqual(sum(item["bytes"] for item in receipt["verified_files"]), receipt["total_bytes"])
        self.assertEqual(receipt["task_contract_bindings"], 0)
        self.assertEqual(receipt["training_actions"], 0)
        self.assertEqual(receipt["host_promotions"], 0)
        self.assertEqual(receipt["authority"], 0)
        self.assertFalse(receipt["board_accepted"])
        self.assertFalse(receipt["countable_model"])
        for item in receipt["verified_files"]:
            self.assertEqual(sha256_file(ROOT / item["path"]), item["sha256"])

    def test_mendeley_sic_plunger_exact_contract_rejections_are_fail_closed(self):
        receipt = load_json("evidence/mendeley_sic_plunger_source_contract_audit.v1.json")
        self.assertEqual(
            receipt["status"],
            "SOURCE_LICENSE_AND_HASH_PASS_SEVEN_EXACT_CONTRACTS_REJECTED",
        )
        self.assertEqual(receipt["observed"]["independent_sample_family_count"], 3)
        self.assertFalse(receipt["observed"]["curve_points_are_independent_experiments"])
        self.assertEqual(len(receipt["candidate_dispositions"]), 7)
        self.assertTrue(
            all(item["status"].startswith("EXACT_REJECTED") for item in receipt["candidate_dispositions"])
        )
        self.assertEqual(receipt["training_actions"], 0)
        self.assertEqual(receipt["host_promotions"], 0)
        self.assertEqual(receipt["authority"], 0)
        self.assertFalse(receipt["board_accepted"])
        self.assertFalse(receipt["countable_model"])

    def test_p073_failed_residual_exploration_is_quarantined_from_selection(self):
        receipt = load_json("evidence/p073_residual_exploration_quarantine.v1.json")
        self.assertEqual(receipt["candidate_id"], "CAND-P-073")
        self.assertEqual(
            receipt["status"],
            "VALIDATION_GATE_FAILED_TEST_RESULT_QUARANTINED_NO_PROMOTION",
        )
        self.assertFalse(receipt["selection_protocol"]["validation_gate_passed"])
        self.assertTrue(
            receipt["quarantine"]["test_evaluation_mistakenly_performed_after_validation_failure"]
        )
        self.assertFalse(receipt["quarantine"]["test_result_may_be_used_for_selection"])
        self.assertFalse(receipt["quarantine"]["test_result_may_be_used_for_promotion"])
        self.assertEqual(receipt["actions"]["training_promotion_actions"], 0)
        self.assertEqual(receipt["actions"]["host_promotions"], 0)
        self.assertEqual(receipt["authority"], 0)
        self.assertFalse(receipt["board_accepted"])
        self.assertFalse(receipt["countable_model"])

    def test_sim_extension_pack_39_is_frozen_unique_and_never_relabels_experimental_truth(self):
        contract = load_json("contracts/sim_extension_pack_39.v1.json")
        staging = load_json("evidence/sim_extension_pack_39_staging.v1.json")
        closure = load_json("evidence/sim_extension_pack_39_closure.v1.json")
        self.assertEqual(contract["status"], "PRETRAIN_FROZEN")
        self.assertEqual(len(contract["tasks"]), 39)
        self.assertEqual(len({item["candidate_id"] for item in contract["tasks"]}), 39)
        self.assertEqual(staging["status"], "PASS_39_SIM_EXTENSION_DATASETS_FROZEN")
        self.assertEqual(staging["task_count"], 39)
        self.assertTrue(all(item["cross_split_family_overlap"] == 0 for item in staging["records"]))
        self.assertTrue(all(item["original_task_contract_status"] == "UNCHANGED_FAIL_CLOSED" for item in staging["records"]))
        self.assertEqual(closure["status"], "PASS")
        self.assertEqual(closure["host_extension_passes"], 39)
        self.assertEqual(closure["rejections"], 0)
        self.assertEqual(closure["original_exact_contract_promotions"], 0)
        self.assertTrue(all(item["host_extension_pass"] for item in closure["records"]))
        self.assertTrue(all(not item["host_contract_pass"] for item in closure["records"]))
        self.assertTrue(all(item["authority"] == 0 and not item["board_accepted"] for item in closure["records"]))

    def test_host_closure_v6_meets_asset_floor_only_with_explicit_sim_boundary(self):
        closure = load_json("evidence/host_closure.v6.json")
        self.assertEqual(
            closure["status"],
            "HOST_ASSET_FLOOR_MET_IF_UNIFIED_BOARD_PASSES_FULL_TARGET_OPEN_BOARD_PENDING",
        )
        self.assertEqual(closure["exact_contract"]["unique_candidates"], 78)
        self.assertEqual(closure["sim_only_extensions"]["unique_candidates"], 46)
        self.assertEqual(closure["host_qualified_total_including_extensions"], 124)
        self.assertEqual(closure["combined_assets_if_all_host_assets_later_board_pass"], 154)
        self.assertTrue(closure["release_floor"]["asset_floor_met_if_all_host_assets_board_pass"])
        self.assertFalse(closure["release_floor"]["exact_source_bound_floor_met"])
        self.assertEqual(closure["full_target"]["gap_by_category"], {"G": 5, "P": 41, "S": 0})
        self.assertEqual(closure["full_target"]["total_gap"], 46)
        self.assertEqual(closure["integrity"]["package_collisions"], 0)
        self.assertEqual(closure["integrity"]["payload_collisions"], 0)
        self.assertEqual(closure["authority_nonzero"], 0)
        self.assertEqual(closure["new_models_board_accepted"], 0)
        self.assertEqual(closure["new_models_countable_publicly"], 0)

    def test_modelbank_v6_and_full_host_checks_cover_all_124_assets(self):
        build = load_json("evidence/modelbank_build.v6.json")
        dry_run = load_json("evidence/modelbank_host_dry_run.v6.json")
        artifacts = load_json("evidence/host_artifact_verification.v6.json")
        self.assertEqual(build["model_count"], 124)
        self.assertEqual(build["exact_count"], 78)
        self.assertEqual(build["sim_only_extension_count"], 46)
        self.assertEqual(build["file_count"], 573)
        self.assertEqual(build["manifest_sha256"], sha256_file(ROOT / build["manifest_path"]))
        self.assertEqual(dry_run["status"], "PASS")
        self.assertEqual(dry_run["model_count"], 124)
        self.assertEqual(dry_run["successful_swaps"], 1000)
        self.assertEqual(len(dry_run["fault_injections"]), 6)
        self.assertTrue(all(dry_run["invariants"].values()))
        self.assertEqual(artifacts["status"], "PASS_HOST_ONLY_BOARD_PENDING")
        self.assertEqual(artifacts["model_count"], 124)
        self.assertEqual(artifacts["onnx_full_check_pass"], 124)
        self.assertEqual(artifacts["golden_archives_pass"], 124)
        self.assertEqual(artifacts["authority_nonzero"], 0)
        self.assertEqual(artifacts["board_accepted"], 0)

    def test_board41_sim_extensions_are_frozen_unique_and_fail_closed_for_exact_truth(self):
        staging = load_json("evidence/sim_extension_board41_staging.v1.json")
        closure = load_json("evidence/sim_extension_board41_closure.v1.json")
        self.assertEqual(staging["status"], "PASS_41_SIM_EXTENSION_DATASETS_FROZEN")
        self.assertEqual(staging["task_count"], 41)
        self.assertEqual(staging["original_exact_contract_promotions"], 0)
        self.assertTrue(all(item["cross_split_family_overlap"] == 0 for item in staging["records"]))
        self.assertTrue(all(item["original_task_contract_status"] == "UNCHANGED_FAIL_CLOSED" for item in staging["records"]))
        self.assertEqual(closure["status"], "PASS")
        self.assertEqual(closure["host_extension_passes"], 41)
        self.assertEqual(closure["rejections"], 0)
        self.assertEqual(closure["original_exact_contract_promotions"], 0)
        self.assertTrue(closure["training_view_payloads_are_ntfs_hardlinks"])
        self.assertTrue(all(item["authority"] == 0 and not item["board_accepted"] for item in closure["records"]))

    def test_nanolm_sim5_completes_new_generative_slots_without_expert_claims(self):
        staging = load_json("evidence/nanolm_sim_extension5_staging.v1.json")
        closure = load_json("evidence/nanolm_sim_extension5_closure.v1.json")
        self.assertEqual(staging["status"], "PASS_5_SIM_ONLY_NANOLM_DATASETS_FROZEN")
        self.assertEqual(staging["candidate_count"], 5)
        self.assertEqual(staging["original_exact_contract_promotions"], 0)
        self.assertTrue(all(item["cross_split_group_overlap"] == 0 for item in staging["records"]))
        self.assertEqual(closure["status"], "PASS")
        self.assertEqual(closure["candidate_count"], 5)
        self.assertEqual(closure["host_extension_passes"], 5)
        self.assertEqual(closure["original_exact_contract_promotions"], 0)
        self.assertEqual(closure["package_collisions"], 0)
        self.assertEqual(closure["payload_collisions"], 0)
        self.assertTrue(closure["independent_full_weight_packages"])
        self.assertTrue(all(item["promotion"]["public_claim_scope"] == "SIM_ONLY" for item in closure["records"]))
        self.assertTrue(
            all(item["promotion"]["authority"] == 0 and not item["promotion"]["board_accepted"] for item in closure["records"])
        )

    def test_host_closure_v7_meets_full_170_target_with_exact_sim_separation(self):
        closure = load_json("evidence/host_closure.v7.json")
        self.assertEqual(
            closure["status"],
            "HOST_FULL_170_TARGET_MET_IF_UNIFIED_BOARD_PASSES_EXACT_AND_SIM_BOUNDARIES_SEPARATED",
        )
        self.assertEqual(closure["exact_contract"]["unique_candidates"], 78)
        self.assertEqual(closure["sim_only_extensions"]["unique_candidates"], 92)
        self.assertEqual(closure["host_qualified_total_including_extensions"], 170)
        self.assertEqual(closure["host_by_category"], {"G": 30, "P": 112, "S": 28})
        self.assertEqual(closure["combined_assets_if_all_host_assets_later_board_pass"], 200)
        self.assertEqual(closure["combined_logical_generative_if_all_host_assets_later_board_pass"], 38)
        self.assertEqual(closure["full_target"]["total_gap"], 0)
        self.assertEqual(closure["integrity"]["package_collisions"], 0)
        self.assertEqual(closure["integrity"]["payload_collisions"], 0)
        self.assertEqual(closure["authority_nonzero"], 0)
        self.assertEqual(closure["new_models_board_accepted"], 0)
        self.assertEqual(closure["new_models_countable_publicly"], 0)

    def test_modelbank_v7_and_full_host_checks_cover_all_170_assets(self):
        build = load_json("evidence/modelbank_build.v7.json")
        dry_run = load_json("evidence/modelbank_host_dry_run.v7.json")
        artifacts = load_json("evidence/host_artifact_verification.v7.json")
        self.assertEqual(build["model_count"], 170)
        self.assertEqual(build["exact_count"], 78)
        self.assertEqual(build["sim_only_extension_count"], 92)
        self.assertEqual(build["category_counts"], {"G": 30, "P": 112, "S": 28})
        self.assertEqual(build["file_count"], 803)
        self.assertEqual(build["manifest_sha256"], sha256_file(ROOT / build["manifest_path"]))
        self.assertEqual(dry_run["status"], "PASS")
        self.assertEqual(dry_run["model_count"], 170)
        self.assertEqual(dry_run["successful_swaps"], 1000)
        self.assertEqual(len(dry_run["fault_injections"]), 6)
        self.assertTrue(all(dry_run["invariants"].values()))
        self.assertEqual(artifacts["status"], "PASS_HOST_ONLY_BOARD_PENDING")
        self.assertEqual(artifacts["model_count"], 170)
        self.assertEqual(artifacts["onnx_full_check_pass"], 170)
        self.assertEqual(artifacts["golden_archives_pass"], 170)
        self.assertEqual(artifacts["authority_nonzero"], 0)
        self.assertEqual(artifacts["board_accepted"], 0)

    def test_release_gap_v7_covers_244_and_keeps_exact_sim_claims_separate(self):
        audit = load_json("evidence/release_gap_audit.v7.json")
        self.assertEqual(
            audit["status"],
            "HOST_FULL_170_TARGET_MET_EXACT_SOURCE_FLOOR_STILL_SHORT_42_UNIFIED_BOARD_PENDING",
        )
        self.assertEqual(audit["candidate_universe"], 244)
        self.assertEqual(len(audit["records"]), 244)
        self.assertEqual(len({item["candidate_id"] for item in audit["records"]}), 244)
        self.assertEqual(audit["host_exact_source_bound"]["total"], 78)
        self.assertEqual(audit["host_exact_source_bound"]["minimum_release_shortfall"], 42)
        self.assertEqual(audit["host_sim_only_extensions"]["total"], 92)
        self.assertTrue(audit["host_sim_only_extensions"]["not_source_gate_substitutes"])
        self.assertEqual(audit["host_total_including_sim_only"], 170)
        self.assertEqual(audit["host_total_by_category"], {"G": 30, "P": 112, "S": 28})
        self.assertEqual(audit["host_target_gap"], 0)
        self.assertEqual(audit["new_board_accepted"], 0)
        self.assertEqual(audit["new_countable_publicly"], 0)
        self.assertTrue(all(item["authority"] == 0 and not item["board_accepted"] for item in audit["records"]))

    def test_unified_staging_v7_references_initial_30_and_new_170_without_production_copy(self):
        staging = load_json("evidence/unified_staging.v7.json")
        self.assertEqual(staging["status"], "HOST_FULL_200_ASSET_STAGING_PASS_UNIFIED_BOARD_PENDING")
        self.assertEqual(staging["scope"], "HASH_ONLY_REFERENCE_NO_PRODUCTION_COPY_NO_BOARD_ACTION")
        self.assertEqual(staging["frozen_initial_baseline"]["assets"], 30)
        self.assertEqual(staging["frozen_initial_baseline"]["logical_models"], 28)
        self.assertTrue(staging["frozen_initial_baseline"]["fallback_preserved"])
        self.assertEqual(staging["new_host_modelbank"]["host_qualified"], 170)
        self.assertEqual(staging["new_host_modelbank"]["by_category"], {"G": 30, "P": 112, "S": 28})
        self.assertEqual(staging["new_host_modelbank"]["exact_source_bound"], 78)
        self.assertEqual(staging["new_host_modelbank"]["sim_only_extensions"], 92)
        self.assertEqual(staging["combined_projection"]["assets_if_all_new_host_models_later_pass_board"], 200)
        self.assertEqual(staging["combined_projection"]["logical_generative_models_if_all_new_host_models_later_pass_board"], 38)
        self.assertEqual(staging["production_files_modified"], 0)
        self.assertEqual(staging["sd_or_board_actions"], 0)
        for item in staging["new_host_modelbank"]["receipts"].values():
            self.assertEqual(sha256_file(ROOT / item["path"]), item["sha256"])

    def test_mcu_v8_runtime_reproduces_all_170_binary_goldens(self):
        export = load_json("evidence/mcu_runtime_export.v8.json")
        runtime = load_json("evidence/mcu_runtime_c_verification.v8.json")
        compile_receipt = load_json("evidence/firmware_runtime_host_compile.v8.json")
        self.assertEqual(export["model_count"], 170)
        self.assertEqual(export["by_category"], {"G": 30, "P": 112, "S": 28})
        self.assertEqual(export["exact_count"], 78)
        self.assertEqual(export["sim_only_count"], 92)
        self.assertEqual(export["package_collisions"], 0)
        self.assertEqual(export["payload_collisions"], 0)
        self.assertEqual(runtime["status"], "PASS_170_PORTABLE_C_GOLDENS_CORTEX_M7_EXECUTION_PENDING")
        self.assertEqual(runtime["model_count"], 170)
        self.assertEqual(runtime["by_engine"], {"1": 139, "2": 1, "5": 30})
        self.assertEqual(runtime["authority_nonzero"], 0)
        self.assertEqual(compile_receipt["models_checked"], 170)
        self.assertEqual(len(compile_receipt["objects"]), 3)
        self.assertTrue(all(item["warnings"] == 0 and item["errors"] == 0 for item in compile_receipt["objects"]))

    def test_sd_v8r2_catalog_and_c_refusal_fixtures_are_complete(self):
        staging = load_json("evidence/sd_staging_verification.v8r2.json")
        faults = load_json("evidence/sd_fault_c_verification.v8.json")
        self.assertEqual(staging["status"], "SD_STAGING_170_HOST_VERIFIED_UNIFIED_BOARD_PENDING")
        self.assertEqual(staging["model_count"], 170)
        self.assertEqual(staging["by_category"], {"G": 30, "P": 112, "S": 28})
        self.assertEqual(staging["exact_count"], 78)
        self.assertEqual(staging["sim_only_count"], 92)
        self.assertEqual(staging["fault_fixture_count"], 7)
        self.assertEqual(staging["authority_nonzero"], 0)
        self.assertEqual(faults["status"], "PASS_SEVEN_C_LOADER_REFUSAL_CASES_BOARD_PENDING")
        self.assertEqual(faults["passed"], 7)
        self.assertEqual(faults["failed"], 0)
        self.assertEqual(faults["authority_nonzero_accepted"], 0)
        self.assertTrue(all(item["pass"] for item in faults["cases"]))

    def test_gd32_r21_integration_is_locally_linked_but_hardware_pending(self):
        receipt = load_json("evidence/gd32_production_integration.v8r8.json")
        self.assertEqual(
            receipt["status"],
            "LOCAL_KEIL_AND_SD_STAGING_PASS_UNIFIED_PHYSICAL_BOARD_PENDING",
        )
        self.assertEqual(receipt["build_config"]["LAB_HARDWARE_BRINGUP"], "0")
        self.assertEqual(receipt["build_config"]["LAB_FORGE200_BOARD_ACCEPTANCE"], "1")
        self.assertEqual(receipt["build_config"]["LAB_FORGE200_ACTION_AUTHORITY"], "0")
        self.assertEqual(receipt["build_config"]["configTOTAL_HEAP_SIZE_KiB"], 96)
        self.assertEqual(receipt["keil"]["target"], "R2.1")
        self.assertEqual(receipt["keil"]["full_rebuild_errors"], 0)
        self.assertEqual(receipt["keil"]["new_or_modified_source_warnings"], 0)
        self.assertLessEqual(
            receipt["keil"]["program_size"]["rom_bytes"],
            receipt["keil"]["program_size"]["rom_88_percent_limit"],
        )
        self.assertEqual(receipt["sd_staging"]["models"], 170)
        self.assertEqual(receipt["authority_nonzero"], 0)
        self.assertEqual(receipt["board_actions"], 0)
        self.assertFalse(receipt["ready_for_gd32_burn_now"])
        self.assertGreater(len(receipt["remaining_physical_gates"]), 0)

    def test_board_catalog_release_root_is_bound_to_both_firmware_copies(self):
        production = ROOT.parents[1] / "CIMC"
        catalog = (
            ROOT
            / "releases/forge200-sd-staging-v9r2-20260804/F200/CATALOGA.BIN"
        ).read_bytes()
        self.assertEqual(catalog[:4], b"F2CT")
        catalog_root = catalog[64:96]
        self.assertEqual(len(catalog_root), 32)
        sources = (
            production
            / "firmware/keil_proj/HardWare/Lab_Sentinel/forge200_board_port.c",
            ROOT
            / "firmware_integration/modelbank_v8_gd32/forge200_board_port.c",
        )
        for source in sources:
            text = source.read_text(encoding="utf-8")
            block = text.split(
                "static const uint8_t s_release_content_root[32] = {", 1
            )[1].split("};", 1)[0]
            compiled_root = bytes(
                int(value, 16)
                for value in re.findall(r"0x([0-9a-fA-F]+)U", block)
            )
            self.assertEqual(compiled_root, catalog_root, source)

    def test_sd_crc_retry_reaches_verified_conservative_clock_and_recovers_nominal(self):
        production = ROOT.parents[1] / "CIMC"
        sd_source = (
            production / "firmware/keil_proj/HardWare/Sensors/sd_spi.c"
        ).read_text(encoding="utf-8")
        sd_header = (
            production / "firmware/keil_proj/HardWare/Sensors/sd_spi.h"
        ).read_text(encoding="utf-8")
        max_source = (
            production / "firmware/keil_proj/HardWare/Sensors/max31856.c"
        ).read_text(encoding="utf-8")
        board_sources = (
            production
            / "firmware/keil_proj/HardWare/Lab_Sentinel/forge200_board_port.c",
            ROOT
            / "firmware_integration/modelbank_v8_gd32/forge200_board_port.c",
        )

        self.assertIn("for (attempt = 0U; attempt < 10U; ++attempt)", sd_source)
        self.assertIn("if (s_sd_delay_cycles < 512U)", sd_source)
        self.assertIn("else if (s_sd_delay_cycles < 700U)", sd_source)
        self.assertIn("s_sd_delay_cycles = 700U;", sd_source)
        self.assertIn(
            "s_sd_delay_cycles = s_sd_nominal_delay_cycles;", sd_source
        )
        self.assertIn("sd_spi_peak_retry_delay_cycles", sd_header)
        self.assertIn("#define SD_GPIO_SPEED GPIO_OSPEED_12MHZ", sd_source)
        self.assertIn(
            "#define MAX56_GPIO_SPEED GPIO_OSPEED_12MHZ", max_source
        )
        self.assertNotIn("GPIO_OSPEED_60MHZ", sd_source)
        self.assertNotIn("GPIO_OSPEED_60MHZ", max_source)
        board_texts = [source.read_text(encoding="utf-8") for source in board_sources]
        self.assertEqual(board_texts[0], board_texts[1])
        self.assertTrue(all("sd_peak_delay=%lu" in text for text in board_texts))

    def test_board_uart_parser_is_fail_closed_and_keeps_exact_sim_boundaries(self):
        parser_text = (ROOT / "pipeline/parse_board_receipt_v8.py").read_text(
            encoding="utf-8"
        )
        for gate in (
            "INITIAL_30_REGISTRY_PASS_MISSING",
            "MODEL_UNIQUE_170_GATE",
            "SWAP1000_GATE",
            "SOAK_24_HOUR_GATE",
            "SOAK_FAULT_EVERY_2H_GATE",
            "FINAL_PASS_GATE",
        ):
            self.assertIn(gate, parser_text)
        self.assertIn('"public_exact_source_bound_new": 78', parser_text)
        self.assertIn('"public_sim_only_new": 92', parser_text)
        self.assertIn('"authority_nonzero": 0', parser_text)


if __name__ == "__main__":
    unittest.main()
