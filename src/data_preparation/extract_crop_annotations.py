#!/usr/bin/env python3
"""정본 Crop과 원본 Annotation JSON을 검증하고 약품별로 원본 그대로 복사한다.

입력: Crop 루트, crop_metadata.csv, 원본 Annotation JSON 루트.
출력: category_id_약품명 폴더의 원본 JSON 복사본과 선택적 검증 보고서.
처리: crop_path 정본 확인 → Crop/JSON 1:1·중복·누락·모호성·분류 불일치 검증
      → --execute일 때만 copy2 후 SHA-256 동일성 검증.

예시:
  python extract_crop_annotations.py --crop-dir CROPS \
    --crop-metadata CROPS/crop_metadata.csv --annotation-dir ANNOTATIONS \
    --output-dir OUTPUT --preflight
  python extract_crop_annotations.py ... --execute

JSON은 검증을 위해 파싱할 뿐 복사 시 원본 바이트를 재직렬화하지 않는다.
idx/dl_idx는 클래스 식별이나 매칭에 사용하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def norm(value: object) -> str:
    return unicodedata.normalize("NFC", str(value if value is not None else "").strip())


def safe(value: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", norm(value))[:100]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crop-dir", required=True, type=Path)
    ap.add_argument("--crop-metadata", type=Path)
    ap.add_argument("--annotation-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--report", type=Path)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="검증만 수행하고 JSON을 복사하지 않음(기본값)")
    mode.add_argument("--execute", action="store_true", help="preflight 통과 후 원본 JSON을 그대로 복사")
    args = ap.parse_args()
    metadata = args.crop_metadata or args.crop_dir / "crop_metadata.csv"

    physical = [p.resolve() for p in args.crop_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".png"]
    physical_set = {norm(p) for p in physical}
    with metadata.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    key_counts = Counter((norm(r.get("category_id")), norm(r.get("crop_file_name") or r.get("file_name"))) for r in rows)
    duplicate_metadata_keys = {k: n for k, n in key_counts.items() if k[1] and n > 1}
    canonical = []
    canonical_seen = set()
    canonical_missing = []
    for r in rows:
        category_id = norm(r.get("category_id"))
        file_name = norm(r.get("crop_file_name") or r.get("file_name"))
        crop_path = Path(norm(r.get("crop_path"))).expanduser()
        key = (category_id, file_name)
        if key in canonical_seen:
            continue
        canonical_seen.add(key)
        if not crop_path.is_absolute():
            crop_path = args.crop_dir / crop_path
        crop_path = crop_path.resolve()
        if norm(crop_path) not in physical_set:
            canonical_missing.append((category_id, file_name, str(crop_path)))
            continue
        canonical.append((r, crop_path, category_id, file_name))

    json_files = [p.resolve() for p in args.annotation_dir.rglob("*.json") if p.is_file()]
    by_name = defaultdict(list)
    for p in json_files:
        by_name[norm(p.name)].append(p)

    matched = []
    missing_json = []
    ambiguous = []
    category_mismatch = []
    parse_fail = []
    class_names = defaultdict(Counter)
    used_json = set()
    for r, crop_path, category_id, file_name in canonical:
        json_name = norm(Path(file_name).with_suffix(".json").name)
        hits = by_name.get(json_name, [])
        row_name = norm(r.get("category_name") or r.get("drug_name") or r.get("item_name"))
        if len(hits) == 0:
            missing_json.append((category_id, file_name, str(crop_path)))
            continue
        if len(hits) > 1:
            ambiguous.append((category_id, file_name, [str(p) for p in hits]))
            continue
        jp = hits[0]
        try:
            data = json.loads(jp.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            parse_fail.append((category_id, file_name, str(jp), str(exc)))
            continue
        images = data.get("images") if isinstance(data, dict) else None
        image = None
        if isinstance(images, list):
            same = [x for x in images if isinstance(x, dict) and norm(Path(norm(x.get("file_name"))).name) == norm(file_name)]
            image = same[0] if same else (images[0] if len(images) == 1 and isinstance(images[0], dict) else None)
        json_name_value = ""
        if image:
            for k in ("item_name", "drug_name", "dl_name", "category_name"):
                if norm(image.get(k)): json_name_value = norm(image.get(k)); break
        if row_name and json_name_value and row_name != json_name_value:
            category_mismatch.append((category_id, file_name, row_name, json_name_value, str(jp)))
            continue
        # JSON의 annotations[].category_id는 객체 순번, item_seq는 품목일련번호이며
        # dl_idx/idx도 TS class ID 체계가 아니다. 클래스는 검증된 Crop metadata만 사용한다.
        drug_name = row_name or json_name_value
        if drug_name:
            class_names[category_id][drug_name] += 1
        matched.append((r, crop_path, jp, category_id, file_name, drug_name))
        used_json.add(norm(jp))

    name_conflicts = {cid: dict(names) for cid, names in class_names.items() if len(names) > 1}
    folders = {f"{safe(cid)}_{safe(max(names, key=names.get))}" for cid, names in class_names.items() if names and cid not in name_conflicts}
    physical_duplicates_ignored = max(0, len(physical) - len(canonical))
    orphan_json = [p for p in json_files if norm(p) not in used_json]
    by_class = defaultdict(lambda: {"crop": 0, "json": 0, "missing": 0})
    for _, _, cid, _ in canonical: by_class[cid]["crop"] += 1
    for _, _, _, cid, _, _ in matched: by_class[cid]["json"] += 1
    for cid, _, _ in missing_json: by_class[cid]["missing"] += 1

    result = {
        "crop_dir": str(args.crop_dir.resolve()),
        "metadata": str(metadata.resolve()),
        "annotation_dir": str(args.annotation_dir.resolve()),
        "output_dir_planned": str(args.output_dir.resolve()),
        "physical_png": len(physical),
        "metadata_rows": len(rows),
        "canonical_target_crops": len(canonical),
        "physical_duplicate_crops_ignored": physical_duplicates_ignored,
        "canonical_crop_path_missing": len(canonical_missing),
        "json_match_success": len(matched),
        "json_match_failure": len(missing_json) + len(ambiguous) + len(parse_fail) + len(category_mismatch),
        "missing_json": len(missing_json),
        "duplicate_json_basename": sum(1 for v in by_name.values() if len(v) > 1),
        "ambiguous_matches": len(ambiguous),
        "json_parse_fail": len(parse_fail),
        "category_mismatch": len(category_mismatch),
        "duplicate_metadata_keys": len(duplicate_metadata_keys),
        "classes": len(by_class),
        "planned_folders": len(folders),
        "drug_name_conflicts": len(name_conflicts),
        "all_annotation_json": len(json_files),
        "json_without_target_crop": len(orphan_json),
        "class_summary": [
            {"category_id": cid, "crop_count": x["crop"], "matched_json_count": x["json"], "missing_json_count": x["missing"]}
            for cid, x in sorted(by_class.items(), key=lambda z: int(z[0]) if z[0].isdigit() else 10**18)
        ],
        "warnings": {
            "canonical_crop_path_missing": canonical_missing[:20],
            "missing_json": missing_json[:20],
            "ambiguous": ambiguous[:20],
            "category_mismatch": category_mismatch[:20],
            "parse_fail": parse_fail[:20],
            "drug_name_conflicts": name_conflicts,
            "orphan_json_samples": [str(p) for p in orphan_json[:20]],
        },
    }
    if args.execute:
        blockers = (
            result["canonical_crop_path_missing"] + result["json_match_failure"]
            + result["duplicate_json_basename"] + result["ambiguous_matches"]
            + result["duplicate_metadata_keys"] + result["drug_name_conflicts"]
        )
        if blockers:
            raise SystemExit(f"복사 중단: preflight blocker {blockers}건")
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise SystemExit(f"복사 중단: 출력 폴더가 비어 있지 않습니다: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for _, _, source_json, category_id, _, drug_name in matched:
            folder = args.output_dir / f"{safe(category_id)}_{safe(drug_name)}"
            folder.mkdir(parents=True, exist_ok=True)
            destination = folder / source_json.name
            if destination.exists():
                raise SystemExit(f"복사 중단: 중복 출력 JSON: {destination}")
            shutil.copy2(source_json, destination)
            src_hash = hashlib.sha256(source_json.read_bytes()).digest()
            dst_hash = hashlib.sha256(destination.read_bytes()).digest()
            if src_hash != dst_hash:
                raise SystemExit(f"복사 검증 실패: {destination}")
            copied += 1
        result["execute"] = {
            "copied_json": copied,
            "verified_identical": copied,
            "output_dir": str(args.output_dir.resolve()),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
