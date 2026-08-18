from __future__ import annotations

import ast
import json
import re
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SecurityContractTests(unittest.TestCase):
    def test_privacy_path_pattern_is_linear_and_matches_both_separators(self) -> None:
        source = (ROOT / "tools" / "verify_release.py").read_text(encoding="utf-8")
        self.assertIn(r"[\\\\/]+(?:Users|WorkData|xrd_backup)[\\\\/]+", source)
        pattern = re.compile(r"(?i)[A-Z]:[\\/]+(?:Users|WorkData|xrd_backup)[\\/]+")
        windows_sample = "D:" + "\\" * 2 + "Work" + "Data" + "\\private"
        slash_sample = "C:/" + "Us" + "ers/private"
        self.assertIsNotNone(pattern.search(windows_sample))
        self.assertIsNotNone(pattern.search(slash_sample))

    def test_gpu_workers_inherit_current_interpreter(self) -> None:
        for relative in (
            "ai/pipeline/gpu_queue_worker.py",
            "ai/training/forge200_pipeline/gpu_queue_worker.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("sys.executable,", source)
            self.assertNotIn('add_argument("--python"', source)
            self.assertNotIn("args.python,", source)

    def test_public_nanolm_loaders_are_weights_only(self) -> None:
        for relative in (
            "ai/training/nanolm/train_flagship.py",
            "ai/training/nanolm/export_flagship_to_c.py",
            "ai/training/nanolm/train_nanolm.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("weights_only=True", source)
        flagship = (ROOT / "ai/training/nanolm/train_flagship.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("type=parse_tag", flagship)

    def test_all_public_torch_load_calls_use_weights_only(self) -> None:
        unsafe: list[str] = []
        for path in (ROOT / "ai").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "torch"
                    and node.func.attr == "load"
                ):
                    continue
                keyword = next(
                    (item.value for item in node.keywords if item.arg == "weights_only"),
                    None,
                )
                if not (
                    isinstance(keyword, ast.Constant)
                    and keyword.value is True
                ):
                    unsafe.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual(unsafe, [])

    def test_public_numpy_loaders_never_enable_pickle(self) -> None:
        unsafe: list[str] = []
        for path in (ROOT / "ai").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in {"np", "numpy"}
                    and node.func.attr == "load"
                ):
                    continue
                keyword = next(
                    (item.value for item in node.keywords if item.arg == "allow_pickle"),
                    None,
                )
                if isinstance(keyword, ast.Constant) and keyword.value is True:
                    unsafe.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual(unsafe, [])

    def test_public_c_runtimes_keep_explicit_range_guards(self) -> None:
        runtime = (
            ROOT / "ai/firmware_integration/modelbank_v8/forge200_runtime_v8.c"
        ).read_text(encoding="utf-8")
        rag = (ROOT / "ai/firmware_integration/rag_v9/forge200_rag_v9.c").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("24ULL *", runtime)
        self.assertIn("plane64 > UINT32_MAX / 24U", runtime)
        self.assertIn("channels > UINT32_MAX / plane", runtime)
        self.assertIn("!count_fits_size_t((uintmax_t)maximum * 2U, sizeof(float))", runtime)
        self.assertIn("output_elems != model->output_elems", runtime)
        self.assertIn("!count_fits_size_t(output_elems, sizeof(float))", runtime)
        self.assertIn("index >= F2RAG_WORKLOAD_PER_DOMAIN", rag)

    def test_unlicensed_optional_lru_is_not_redistributed(self) -> None:
        lru_root = ROOT / "firmware/lvgl_ui/lvgl-8.3.11/src/misc"
        self.assertFalse((lru_root / "lv_lru.c").exists())
        self.assertFalse((lru_root / "lv_lru.h").exists())
        project = (
            ROOT / "firmware/keil_proj/project/CIMC_GD32_Template.uvprojx"
        ).read_text(encoding="utf-8")
        self.assertNotIn("lv_lru.c", project)
        lvgl = ROOT / "firmware/lvgl_ui/lvgl-8.3.11/src"
        self.assertEqual(list((lvgl / "draw/sdl").rglob("*")), [])
        self.assertEqual(list((lvgl / "extra/libs/tiny_ttf").rglob("*")), [])
        font_root = lvgl / "font"
        self.assertEqual(list(font_root.glob("lv_font_dejavu*.c")), [])
        self.assertEqual(list(font_root.glob("lv_font_unscii*.c")), [])
        self.assertEqual(
            {path.name for path in font_root.glob("lv_font_montserrat_*.c")},
            {
                "lv_font_montserrat_14.c",
                "lv_font_montserrat_18.c",
                "lv_font_montserrat_20.c",
                "lv_font_montserrat_28.c",
            },
        )
        fallback = (ROOT / "firmware/ai_models_c/lab_font_cn.h").read_text(
            encoding="utf-8"
        )
        self.assertIn("lab_font_cn16 lv_font_montserrat_14", fallback)
        self.assertNotIn("lab_font_cn16 lv_font_montserrat_16", fallback)
        draw_make = (lvgl / "draw/lv_draw.mk").read_text(encoding="utf-8")
        libs_header = (lvgl / "extra/libs/lv_libs.h").read_text(encoding="utf-8")
        self.assertNotIn("draw/sdl", draw_make)
        self.assertNotIn("tiny_ttf", libs_header)

    def test_privacy_scanner_does_not_publish_private_denylist_values(self) -> None:
        source = (ROOT / "tools" / "verify_release.py").read_text(encoding="utf-8")
        self.assertNotIn('"private_identity"', source)
        self.assertNotIn('"tools/verify_release.py"', source)
        self.assertIn('"registration_number"', source)
        self.assertIn('"repeated_short_secret"', source)
        self.assertIn('"local_username_shape"', source)
        self.assertIn("LAB_SENTINEL_PRIVATE_DENYLIST", source)

    def test_private_strong_name_key_is_not_redistributed(self) -> None:
        compiler = ROOT / "firmware/keil_proj/lwip/contrib/apps/LwipMibCompiler"
        self.assertFalse(compiler.exists())
        verifier = (ROOT / "tools/verify_release.py").read_text(encoding="utf-8")
        self.assertIn('".snk"', verifier)

    def test_generated_font_subset_has_separate_sbom_license(self) -> None:
        builder = (ROOT / "tools/build_sbom.py").read_text(encoding="utf-8")
        self.assertIn('"SPDXRef-Package-LVGL-Fonts"', builder)
        self.assertIn('"OFL-1.1 AND CC-BY-4.0"', builder)
        self.assertIn('rule.license_id.split(" AND ")', builder)
        for package_id in (
            "SPDXRef-Package-Arm2D",
            "SPDXRef-Package-LodePNG",
            "SPDXRef-Package-TJpgDec",
            "SPDXRef-Package-TLSF",
            "SPDXRef-Package-mpaland-printf",
            "SPDXRef-Package-NXP-LVGL-Adapters",
        ):
            self.assertIn(package_id, builder)
        self.assertIn('"LicenseRef-TJpgDec"', builder)
        self.assertIn('"hasExtractedLicensingInfos"', builder)
        self.assertIn('"SPDXRef-Package-License-Metadata"', builder)
        self.assertIn('"firmware/keil_proj/CMSIS/gd32h7xx_libopt.h"', builder)
        self.assertIn('"firmware/keil_proj/User/gd32h7xx_it.c"', builder)
        self.assertIn('"hardware/design/README.md"', builder)
        self.assertIn('"firmware/keil_proj/lwip/PUBLIC_SUBSET.md"', builder)

    def test_hardware_fabrication_archives_exclude_exporter_instructions(self) -> None:
        archives = sorted((ROOT / "hardware/design").rglob("*.zip"))
        self.assertEqual(len(archives), 3)
        for archive_path in archives:
            with self.subTest(archive=archive_path.relative_to(ROOT)):
                with zipfile.ZipFile(archive_path) as archive:
                    self.assertIsNone(archive.testzip())
                    self.assertFalse(
                        any(
                            Path(info.filename).suffix.lower() == ".txt"
                            for info in archive.infolist()
                            if not info.is_dir()
                        )
                    )

    def test_public_taxonomy_does_not_redistribute_paper_snippets(self) -> None:
        taxonomy = json.loads(
            (ROOT / "ai/training/ai5_rootcause/taxonomy.json").read_text(
                encoding="utf-8"
            )
        )

        def keys(value: object) -> set[str]:
            if isinstance(value, dict):
                return set(value) | set().union(*(keys(item) for item in value.values()))
            if isinstance(value, list):
                return set().union(*(keys(item) for item in value))
            return set()

        self.assertNotIn("snippet", keys(taxonomy))
        builder = (
            ROOT / "ai/training/ai5_rootcause/build_taxonomy.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('"snippet":', builder)
        self.assertNotIn("def snippet(", builder)

    def test_disabled_optional_vendor_trees_are_not_redistributed(self) -> None:
        lvgl = ROOT / "firmware/lvgl_ui/lvgl-8.3.11/src/extra/libs"
        self.assertFalse((lvgl / "gif").exists())
        self.assertFalse((lvgl / "qrcode").exists())
        lwip = ROOT / "firmware/keil_proj/lwip"
        self.assertFalse((lwip / "src/netif/ppp").exists())
        self.assertFalse((lwip / "src/apps/http/makefsdata").exists())
        self.assertFalse((lwip / "test").exists())
        self.assertFalse((lwip / "codespell_changed_files.sh").exists())
        self.assertFalse((lwip / "codespell_check.sh").exists())
        self.assertTrue((lwip / "src/include/netif/ppp/ppp_opts.h").is_file())
        libs_header = (lvgl / "lv_libs.h").read_text(encoding="utf-8")
        self.assertNotIn('"gif/lv_gif.h"', libs_header)
        self.assertNotIn('"qrcode/lv_qrcode.h"', libs_header)

    def test_gitleaks_allowlist_is_line_scoped(self) -> None:
        config = (ROOT / ".gitleaks.toml").read_text(encoding="utf-8")
        self.assertNotIn("paths =", config)
        self.assertIn('regexTarget = "line"', config)

    def test_lvgl_radius_mask_checks_index_before_reading_circle_row(self) -> None:
        source = (
            ROOT / "firmware/lvgl_ui/lvgl-8.3.11/src/draw/lv_draw_mask.c"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "i < (int32_t)cir_size && cir_y[i] == y",
            source,
        )
        self.assertNotIn(
            "cir_y[i] == y && i < (int32_t)cir_size",
            source,
        )

    def test_archive_scanner_rejects_keys_and_nested_archives(self) -> None:
        tools_dir = str(ROOT / "tools")
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        import verify_release  # pylint: disable=import-outside-toplevel

        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "fixture.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("secret.snk", b"not-a-real-key")
                archive.writestr(
                    "opaque.bin",
                    b"\x07\x02\x00\x00\x00\xa4\x00\x00RSA2" + b"x" * 16,
                )
                archive.writestr("nested.zip", b"PK\x05\x06" + b"\x00" * 18)
            findings = verify_release._scan_zip(archive_path, "fixture.zip")
        joined = "\n".join(findings)
        self.assertIn("forbidden archive member", joined)
        self.assertIn("private CryptoAPI key blob", joined)
        self.assertIn("nested archive member", joined)

    def test_public_inventory_ignores_build_products(self) -> None:
        tools_dir = str(ROOT / "tools")
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        from _common import iter_public_files  # pylint: disable=import-outside-toplevel

        relative = [path.relative_to(ROOT).as_posix() for path in iter_public_files()]
        self.assertFalse(any("/__pycache__/" in f"/{path}/" for path in relative))
        self.assertFalse(any("/Objects/" in f"/{path}/" for path in relative))
        self.assertFalse(any("/Listings/" in f"/{path}/" for path in relative))


if __name__ == "__main__":
    unittest.main()
