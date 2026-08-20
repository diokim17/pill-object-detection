#!/usr/bin/env node
/**
 * 기존 Train + 추가 TS에서 클래스별 목표 장수까지 균형 있게 이미지를 선별한다.
 *
 * 입력: CSV, JSON(array 또는 {rows:[...]}), XLSX
 * 출력: class_summary / metadata_distribution / selected_images 시트의 XLSX
 *
 * 예시:
 *   node select_balanced_training_images.mjs \
 *     --train train_metadata.csv --ts ts_metadata.csv \
 *     --train-crop-dir cropped_train --ts-crop-dir cropped_ts \
 *     --output outputs/training_selection.xlsx --preflight
 *   node select_balanced_training_images.mjs \
 *     --train train_metadata.csv --ts ts_metadata.csv \
 *     --train-crop-dir cropped_train --ts-crop-dir cropped_ts \
 *     --output outputs/training_selection.xlsx
 */

import fs from "node:fs/promises";
import { constants as fsConstants } from "node:fs";
import path from "node:path";
import process from "node:process";
import { createHash } from "node:crypto";
let FileBlob;
let SpreadsheetFile;
let Workbook;
let sharp;

async function loadRuntimeDependencies() {
  try {
    ({ FileBlob, SpreadsheetFile, Workbook } = await import("@oai/artifact-tool"));
    ({ default: sharp } = await import("sharp"));
  } catch (error) {
    throw new Error(
      "실행 의존성이 필요합니다: npm install @oai/artifact-tool sharp",
      { cause: error },
    );
  }
}

const TARGET_DEFAULT = 100;
const RANDOM_SEED = 20260813;
const META_FIELDS = ["back_color", "light_color", "drug_dir", "camera_la", "camera_lo", "size"];
let activeMetaFields = [...META_FIELDS];
const REQUIRED_OUTPUT = [
  "category_id", "drug_name", "file_name", "back_color", "light_color",
  "drug_dir", "camera_la", "camera_lo", "size", "source", "selection_round",
  "crop_path", "selected_path",
];
const ALIASES = {
  category_id: ["category_id", "class_id", "label_id"],
  drug_name: ["drug_name", "category_name", "class_name", "dl_name"],
  file_name: ["file_name", "filename", "image_name", "crop_file_name", "source_file_name"],
  back_color: ["back_color", "background_color"],
  light_color: ["light_color", "lighting_color"],
  drug_dir: ["drug_dir", "direction"],
  camera_la: ["camera_la", "camera_latitude"],
  camera_lo: ["camera_lo", "camera_longitude"],
  size: ["size", "image_size", "pill_size"],
  source: ["source"],
  selection_round: ["selection_round"],
  crop_path: ["crop_path", "source_crop_path"],
  selected_path: ["selected_path", "train_path", "output_path"],
};

function usage(message = "") {
  if (message) console.error(`오류: ${message}\n`);
  console.error(`사용법:
  node select_balanced_training_images.mjs --train FILE --ts FILE --output FILE [옵션]

필수:
  --train FILE              기존 Train 이미지 메타데이터 (행 단위)
  --ts FILE                 추가 TS 이미지 메타데이터 (여러 번 지정 가능)
  --output FILE             결과 Excel 경로
  --train-crop-dir DIR      기존 Train Crop 루트 (여러 번 지정 가능)
  --ts-crop-dir DIR         추가 TS Crop 루트 (여러 번 지정 가능)

옵션:
  --target N                클래스별 목표 장수 (기본 100)
  --ts-crop-metadata FILE   file_name→절대 crop_path 정본 metadata
                            (미지정 시 각 ts-crop-dir/crop_metadata.csv 자동 사용)
  --ts-annotation-dir DIR   back_color/light/angle/size 원본 Annotation JSON 폴더
  --train-sheet NAME        Train XLSX 시트명 (기본 selected_images 또는 첫 시트)
  --ts-sheet NAME           TS XLSX 시트명 (기본 selected_images 또는 첫 시트)
  --selected-dir DIR        실제 Train 이미지 출력 루트
                            (기본: Excel 폴더/output/train_images)
  --flat-output             클래스 폴더 없이 selected-dir 바로 아래에 복사
  --exclude-category-id ID  TS 추가 대상에서 제외할 category_id (여러 번 지정 가능)
  --preflight, --dry-run    파일을 수정하지 않고 예상 결과만 출력
  --help                    도움말

기존 --output Excel이 있으면 selected_images를 읽어 그대로 유지하고 신규 행만 추가합니다.`);
  process.exit(message ? 2 : 0);
}

function parseArgs(argv) {
  const out = { ts: [], trainCropDirs: [], tsCropDirs: [], excludeCategoryIds: [], target: TARGET_DEFAULT, dryRun: false, flatOutput: false };
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    const take = () => {
      if (++i >= argv.length) usage(`${key} 값이 없습니다.`);
      return argv[i];
    };
    if (key === "--train") out.train = take();
    else if (key === "--ts") out.ts.push(take());
    else if (key === "--output") out.output = take();
    else if (key === "--train-crop-dir") out.trainCropDirs.push(take());
    else if (key === "--ts-crop-dir") out.tsCropDirs.push(take());
    else if (key === "--target") out.target = Number(take());
    else if (key === "--ts-crop-metadata") out.tsCropMetadata = take();
    else if (key === "--ts-annotation-dir") out.tsAnnotationDir = take();
    else if (key === "--train-sheet") out.trainSheet = take();
    else if (key === "--ts-sheet") out.tsSheet = take();
    else if (key === "--selected-dir") out.selectedDir = take();
    else if (key === "--flat-output") out.flatOutput = true;
    else if (key === "--exclude-category-id") out.excludeCategoryIds.push(take());
    else if (key === "--preflight" || key === "--dry-run") out.dryRun = true;
    else if (key === "--help" || key === "-h") usage();
    else usage(`알 수 없는 옵션: ${key}`);
  }
  if (!out.train || !out.ts.length || !out.output) usage("--train, --ts, --output이 필요합니다.");
  if (!out.trainCropDirs.length || !out.tsCropDirs.length)
    usage("실제 Crop 구성을 위해 --train-crop-dir과 --ts-crop-dir이 각각 필요합니다.");
  if (!Number.isInteger(out.target) || out.target <= 0) usage("--target은 양의 정수여야 합니다.");
  return out;
}

const text = (v) => (v === null || v === undefined ? "" : String(v).trim());
const keyText = (v) => text(v).normalize("NFKC").toLocaleLowerCase("ko-KR").replace(/\s+/g, "");
const classKey = (v) => text(v);
const comboKey = (r) => activeMetaFields.map((f) => keyText(r[f]) || "(없음)").join("\u241f");

function parseCsv(raw) {
  const rows = [];
  let row = [], cell = "", quoted = false;
  const source = raw.replace(/^\uFEFF/, "");
  for (let i = 0; i < source.length; i++) {
    const ch = source[i];
    if (quoted) {
      if (ch === '"' && source[i + 1] === '"') { cell += '"'; i++; }
      else if (ch === '"') quoted = false;
      else cell += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(cell); cell = ""; }
    else if (ch === "\n") { row.push(cell.replace(/\r$/, "")); rows.push(row); row = []; cell = ""; }
    else cell += ch;
  }
  if (cell.length || row.length) { row.push(cell.replace(/\r$/, "")); rows.push(row); }
  const nonempty = rows.filter((r) => r.some((v) => text(v)));
  if (!nonempty.length) return [];
  const headers = nonempty[0].map(text);
  return nonempty.slice(1).map((values) => Object.fromEntries(headers.map((h, i) => [h, values[i] ?? ""])));
}

async function rowsFromWorkbook(file, preferredSheet) {
  const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(file));
  const sheetNames = wb.worksheets.items.map((s) => s.name);
  const name = preferredSheet && sheetNames.includes(preferredSheet)
    ? preferredSheet : sheetNames.includes("selected_images") ? "selected_images" : sheetNames[0];
  if (!name) return [];
  const values = wb.worksheets.getItem(name).getUsedRange(true)?.values ?? [];
  if (!values.length) return [];
  const headers = values[0].map(text);
  return values.slice(1).filter((r) => r.some((v) => text(v))).map((r) =>
    Object.fromEntries(headers.map((h, i) => [h, r[i] ?? ""])),
  );
}

async function readRows(file, sheet) {
  const ext = path.extname(file).toLowerCase();
  if (ext === ".xlsx" || ext === ".xls" || ext === ".xlsm") return rowsFromWorkbook(file, sheet);
  const raw = await fs.readFile(file, "utf8");
  if (ext === ".json") {
    const parsed = JSON.parse(raw);
    const rows = Array.isArray(parsed) ? parsed : parsed.rows;
    if (!Array.isArray(rows)) throw new Error(`${file}: JSON은 배열 또는 {rows:[...]} 형식이어야 합니다.`);
    return rows;
  }
  if (ext === ".csv") return parseCsv(raw);
  throw new Error(`${file}: 지원 형식은 CSV, JSON, XLSX입니다.`);
}

function normalizeRows(rows, defaultSource, origin) {
  return rows.map((raw, index) => {
    const lowered = new Map(Object.keys(raw).map((k) => [keyText(k), k]));
    const row = { ...raw };
    for (const [canonical, aliases] of Object.entries(ALIASES)) {
      const found = aliases.map(keyText).find((a) => lowered.has(a));
      row[canonical] = found ? text(raw[lowered.get(found)]) : "";
    }
    row.category_id = classKey(row.category_id);
    row.source = row.source || defaultSource;
    row.selection_round = row.selection_round === "" ? "" : Number(row.selection_round);
    row.__origin = origin;
    row.__index = index;
    return row;
  });
}

function validateInput(rows, label) {
  const errors = [];
  rows.forEach((r, i) => {
    if (!r.category_id) errors.push(`${label} ${i + 2}행: category_id 없음`);
    if (!r.file_name) errors.push(`${label} ${i + 2}행: file_name 없음`);
  });
  if (errors.length) throw new Error(errors.slice(0, 20).join("\n") + (errors.length > 20 ? `\n... 외 ${errors.length - 20}건` : ""));
}

const IMAGE_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"]);
const pathKey = (v) => text(v).normalize("NFKC").replaceAll("\\", "/").replace(/^\.\//, "").toLocaleLowerCase("ko-KR");

async function isNonemptyFile(file) {
  try { return (await fs.stat(file)).isFile() && (await fs.stat(file)).size > 0; }
  catch { return false; }
}

async function walkImages(root) {
  const result = [];
  async function walk(dir) {
    for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) await walk(full);
      else if (entry.isFile() && IMAGE_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) result.push(path.resolve(full));
    }
  }
  await walk(root);
  return result;
}

async function buildCropIndex(roots, label) {
  const files = [], byRelative = new Map(), byBasename = new Map(), byAbsoluteKey = new Map();
  for (const rawRoot of roots) {
    const root = path.resolve(rawRoot);
    let stat;
    try { stat = await fs.stat(root); } catch { throw new Error(`${label} Crop 폴더 없음: ${root}`); }
    if (!stat.isDirectory()) throw new Error(`${label} Crop 경로가 폴더가 아닙니다: ${root}`);
    for (const file of await walkImages(root)) {
      files.push(file);
      byAbsoluteKey.set(pathKey(file), file);
      const rel = pathKey(path.relative(root, file));
      const base = pathKey(path.basename(file));
      if (!byRelative.has(rel)) byRelative.set(rel, []);
      if (!byBasename.has(base)) byBasename.set(base, []);
      byRelative.get(rel).push(file); byBasename.get(base).push(file);
    }
  }
  console.log(`${label} Crop 인덱스: ${files.length}개 파일 / ${roots.length}개 루트`);
  const duplicatePhysicalFiles = [...byBasename.values()].reduce((n, paths) => n + Math.max(0, paths.length - 1), 0);
  return { label, files, fileSet: new Set(files), byAbsoluteKey, byRelative, byBasename, duplicatePhysicalFiles };
}

const indexedPath = (index, candidate) => index.byAbsoluteKey.get(pathKey(path.resolve(candidate))) || "";

async function resolveTsCropMetadata(args) {
  if (args.tsCropMetadata) return path.resolve(args.tsCropMetadata);
  const found = [];
  for (const root of args.tsCropDirs) {
    const candidate = path.resolve(root, "crop_metadata.csv");
    if (await isNonemptyFile(candidate)) found.push(candidate);
  }
  if (found.length === 1) return found[0];
  if (!found.length) return "";
  throw new Error(`TS crop_metadata.csv가 여러 개입니다. --ts-crop-metadata로 하나를 지정하세요: ${found.join(", ")}`);
}

function applyCanonicalCropPaths(rows, cropMetadataRows, index) {
  const canonical = new Map(), canonicalCategory = new Map(), canonicalRow = new Map(), duplicateMetadata = new Set();
  for (const row of cropMetadataRows) {
    const key = keyText(row.file_name);
    if (!key || !row.crop_path) continue;
    if (canonical.has(key)) duplicateMetadata.add(key);
    else { canonical.set(key, path.resolve(row.crop_path)); canonicalCategory.set(key, row.category_id); canonicalRow.set(key, row); }
  }
  let applied = 0, canonicalMissing = 0;
  const enriched = rows.map((row) => {
    const canonicalPath = canonical.get(keyText(row.file_name));
    if (!canonicalPath) return row;
    const actualCanonicalPath = indexedPath(index, canonicalPath);
    if (!actualCanonicalPath) { canonicalMissing++; return row; }
    applied++;
    const supplement = canonicalRow.get(keyText(row.file_name)) || {};
    const enriched = { ...row, crop_path: actualCanonicalPath, crop_path_source: "ts crop_metadata.csv" };
    for (const field of META_FIELDS) if (!text(enriched[field]) && text(supplement[field])) enriched[field] = supplement[field];
    return enriched;
  });
  let duplicatePhysicalIgnored = 0;
  const duplicateByClass = new Map();
  for (const [key, canonicalPath] of canonical) {
    const actualCanonicalPath = indexedPath(index, canonicalPath);
    if (!actualCanonicalPath) continue;
    const extras = Math.max(0, (index.byBasename.get(pathKey(path.basename(actualCanonicalPath))) || []).length - 1);
    if (!extras) continue;
    duplicatePhysicalIgnored += extras;
    const id = canonicalCategory.get(key) || "(unknown)";
    duplicateByClass.set(id, (duplicateByClass.get(id) || 0) + extras);
  }
  return { rows: enriched, stats: { metadataRows: cropMetadataRows.length, canonicalEntries: canonical.size,
    applied, canonicalMissing, duplicateMetadataKeys: duplicateMetadata.size,
    duplicatePhysicalIgnored: index.duplicatePhysicalFiles, duplicateCanonicalLinked: duplicatePhysicalIgnored, duplicateByClass } };
}

async function matchCrop(row, index) {
  if (row.crop_path) {
    const explicit = path.resolve(row.crop_path);
    // crop_path가 원본 full image를 가리키는 사고를 막기 위해 지정 Crop 루트 안의
    // 인덱싱된 이미지일 때만 직접 경로로 인정한다.
    const actualExplicit = indexedPath(index, explicit);
    if (actualExplicit && await isNonemptyFile(actualExplicit)) {
      const duplicateIgnored = Math.max(0, (index.byBasename.get(pathKey(path.basename(actualExplicit))) || []).length - 1);
      return { path: actualExplicit, method: "metadata crop_path", duplicateIgnored };
    }
  }
  const requested = pathKey(row.file_name);
  let candidates = index.byRelative.get(requested) || [];
  if (candidates.length === 1) return { path: candidates[0], method: "relative path" };
  const base = pathKey(path.basename(requested));
  candidates = index.byBasename.get(base) || [];
  if (candidates.length > 1) {
    const id = keyText(row.category_id);
    const byClassFolder = candidates.filter((file) => path.dirname(file).split(path.sep).some((part) => {
      const p = keyText(part); return p === id || p.startsWith(`${id}_`);
    }));
    if (byClassFolder.length === 1) return { path: byClassFolder[0], method: "basename + class folder" };
    if (byClassFolder.length) candidates = byClassFolder;
  }
  if (candidates.length === 1) return { path: candidates[0], method: "basename" };
  if (!candidates.length) return { warning: `${index.label} metadata Crop 없음: class=${row.category_id}, file_name=${row.file_name}` };
  return { warning: `${index.label} Crop 매칭 모호(${candidates.length}개): class=${row.category_id}, file_name=${row.file_name}` };
}

async function attachCropMatches(rows, index, warnings, usedPaths, allowSelectedPath = false, stats = null) {
  const matched = [];
  for (const row of rows) {
    if (stats) {
      stats.attempted++;
      if (!stats.byClass.has(row.category_id)) stats.byClass.set(row.category_id, { attempted: 0, matched: 0, failed: 0, duplicateIgnored: 0 });
      stats.byClass.get(row.category_id).attempted++;
    }
    if (allowSelectedPath && row.selected_path && await isNonemptyFile(path.resolve(row.selected_path))) {
      const selected = path.resolve(row.selected_path);
      matched.push({ ...row, __cropSource: selected, crop_path: row.crop_path || selected, selected_path: selected, match_status: "기존 selected_path 재사용" });
      usedPaths.add(selected);
      if (row.crop_path) {
        const originalCrop = path.resolve(row.crop_path);
        const actualOriginalCrop = indexedPath(index, originalCrop);
        if (actualOriginalCrop) usedPaths.add(actualOriginalCrop);
      }
      if (stats) { stats.matched++; stats.byClass.get(row.category_id).matched++; }
      continue;
    }
    const result = await matchCrop(row, index);
    if (result.path) {
      matched.push({ ...row, __cropSource: result.path, crop_path: result.path, match_status: result.method });
      usedPaths.add(result.path);
      if (stats) {
        stats.matched++; stats.duplicateIgnored += result.duplicateIgnored || 0;
        const c = stats.byClass.get(row.category_id); c.matched++; c.duplicateIgnored += result.duplicateIgnored || 0;
      }
    } else {
      warnings.push({ category_id: row.category_id, message: result.warning });
      if (stats) { stats.failed++; stats.byClass.get(row.category_id).failed++; }
    }
  }
  return matched;
}

function safeFolderName(value) {
  return (text(value) || "unknown").normalize("NFKC").replace(/[\\/:*?"<>|]/g, "_").slice(0, 100);
}

async function fileDigest(file) {
  return createHash("sha256").update(await fs.readFile(file)).digest("hex");
}

async function copySelectedImages(rows, selectedDir, flatOutput) {
  let copied = 0, reused = 0;
  for (const row of rows) {
    let rowReused = false;
    const source = path.resolve(row.__cropSource || row.selected_path || row.crop_path);
    const folder = flatOutput ? selectedDir : path.join(selectedDir, `${safeFolderName(row.category_id)}_${safeFolderName(row.drug_name)}`);
    const destination = path.resolve(folder, path.basename(row.file_name));
    await fs.mkdir(folder, { recursive: true });
    if (source === destination) {
      if (!await isNonemptyFile(destination)) throw new Error(`선택 이미지가 비어 있습니다: ${destination}`);
      reused++; rowReused = true;
    } else if (await isNonemptyFile(destination)) {
      const [srcStat, dstStat] = await Promise.all([fs.stat(source), fs.stat(destination)]);
      if (srcStat.size !== dstStat.size || await fileDigest(source) !== await fileDigest(destination))
        throw new Error(`기존 Train 출력과 원본 Crop 내용이 다릅니다(덮어쓰지 않음): ${destination}`);
      reused++; rowReused = true;
    } else {
      try { await fs.copyFile(source, destination, fsConstants.COPYFILE_EXCL); }
      catch (error) {
        if (error?.code !== "EEXIST") throw error;
        if (await fileDigest(source) !== await fileDigest(destination)) throw new Error(`복사 경합 후 파일 불일치: ${destination}`);
      }
      if (!await isNonemptyFile(destination)) throw new Error(`Crop 복사 검증 실패: ${destination}`);
      copied++;
    }
    row.selected_path = destination;
    row.copy_status = rowReused ? "기존 파일 재사용" : "신규 복사";
  }
  return { copied, reused };
}

function selectedOutputPath(row, selectedDir, flatOutput) {
  const folder = flatOutput ? selectedDir : path.join(selectedDir, `${safeFolderName(row.category_id)}_${safeFolderName(row.drug_name)}`);
  return path.resolve(folder, path.basename(row.file_name));
}

async function reconcilePriorWithOutput(priorRows, selectedDir, flatOutput) {
  const valid = [], missing = [], byClass = new Map();
  for (const row of priorRows) {
    const expected = selectedOutputPath(row, selectedDir, flatOutput);
    if (!byClass.has(row.category_id)) byClass.set(row.category_id, { excel: 0, actual: 0, missing: 0 });
    const stat = byClass.get(row.category_id);
    stat.excel++;
    if (await isNonemptyFile(expected)) {
      stat.actual++;
      valid.push({ ...row, selected_path: expected });
    } else {
      stat.missing++;
      missing.push({ ...row, expected_selected_path: expected });
    }
  }
  return { valid, missing, byClass };
}

function dedupeStable(rows, seen = new Set()) {
  const output = [];
  for (const row of rows) {
    const key = keyText(row.file_name);
    if (!key || seen.has(key)) continue;
    seen.add(key); output.push(row);
  }
  return output;
}

function dedupeTrainCrops(rows) {
  const output = [], seen = new Set();
  for (const row of rows) {
    const key = `${keyText(row.category_id)}\u241f${pathKey(row.crop_path || row.file_name)}`;
    if (seen.has(key)) continue;
    seen.add(key); output.push(row);
  }
  return output;
}

const trainImageKey = (row) => `${keyText(row.category_id)}\u241f${keyText(row.file_name)}`;

function duplicateCount(values) {
  const seen = new Set();
  let duplicates = 0;
  for (const value of values) {
    if (seen.has(value)) duplicates++;
    else seen.add(value);
  }
  return duplicates;
}

async function inspectCropContent(row) {
  try {
    const stats = await sharp(row.__cropSource || row.crop_path)
      .resize(64, 64, { fit: "fill" }).greyscale().stats();
    const channel = stats.channels[0];
    const mean = channel.mean, stdev = channel.stdev, entropy = stats.entropy;
    const blackFrame = mean <= 12 && stdev <= 12;
    const backgroundOnly = stdev <= 3.5 || entropy <= 0.8;
    return { valid: !(blackFrame || backgroundOnly), reason: blackFrame ? "검정 화면" : backgroundOnly ? "배경만/극저대비" : "", mean, stdev, entropy };
  } catch (error) {
    return { valid: false, reason: `픽셀 검사 실패: ${error.message}` };
  }
}

async function filterInvalidCropContent(rows, concurrency = 16) {
  const valid = [], rejected = [], results = new Array(rows.length);
  let cursor = 0;
  await Promise.all(Array.from({ length: Math.min(concurrency, rows.length) }, async () => {
    while (cursor < rows.length) {
      const i = cursor++;
      results[i] = await inspectCropContent(rows[i]);
    }
  }));
  rows.forEach((row, i) => (results[i]?.valid ? valid : rejected).push({ ...row, content_check: results[i] }));
  return { valid, rejected };
}

async function enrichFromAnnotations(rows, annotationDir) {
  if (!annotationDir) throw new Error("전체 TS 후보 평가에는 --ts-annotation-dir이 필요합니다.");
  const root = path.resolve(annotationDir);
  const entries = await fs.readdir(root, { withFileTypes: true });
  const byStem = new Map(entries.filter((e) => e.isFile() && e.name.toLowerCase().endsWith(".json"))
    .map((e) => [keyText(path.basename(e.name, path.extname(e.name))), path.join(root, e.name)]));
  let matched = 0, missing = 0, parseFailed = 0;
  const result = [];
  for (const row of rows) {
    const stem = keyText(path.basename(row.file_name, path.extname(row.file_name)));
    const jsonPath = byStem.get(stem);
    if (!jsonPath) { missing++; result.push(row); continue; }
    try {
      const data = JSON.parse(await fs.readFile(jsonPath, "utf8"));
      const images = Array.isArray(data.images) ? data.images : [];
      const image = images.find((x) => keyText(path.basename(text(x.file_name))) === keyText(path.basename(row.file_name))) || images[0];
      if (!image) { missing++; result.push(row); continue; }
      const enriched = { ...row, annotation_path: jsonPath };
      for (const field of ["back_color", "light_color", "drug_dir", "camera_la", "camera_lo", "size"])
        if (!text(enriched[field]) && text(image[field])) enriched[field] = text(image[field]);
      enriched.annotation_drug_name = text(image.dl_name);
      enriched.annotation_dl_idx = text(image.dl_idx);
      matched++; result.push(enriched);
    } catch { parseFailed++; result.push(row); }
  }
  return { rows: result, stats: { matched, missing, parseFailed } };
}

function validateDrugNames(trainRows, tsRows) {
  const byClass = new Map();
  for (const row of [...trainRows, ...tsRows]) {
    if (!row.category_id || !row.drug_name) continue;
    if (!byClass.has(row.category_id)) byClass.set(row.category_id, new Set());
    byClass.get(row.category_id).add(row.drug_name);
  }
  const conflicts = new Map([...byClass].filter(([, names]) => names.size > 1));
  return { byClass, conflicts };
}

function countsFor(rows) {
  const fieldCounts = Object.fromEntries(activeMetaFields.map((f) => [f, new Map()]));
  const combos = new Map();
  for (const row of rows) {
    for (const field of activeMetaFields) {
      const value = text(row[field]) || "(없음)";
      fieldCounts[field].set(value, (fieldCounts[field].get(value) || 0) + 1);
    }
    const ck = comboKey(row); combos.set(ck, (combos.get(ck) || 0) + 1);
  }
  return { fieldCounts, combos };
}

function seededTieBreak(row) {
  const input = `${RANDOM_SEED}\u241f${keyText(row.category_id)}\u241f${keyText(row.file_name)}\u241f${pathKey(row.crop_path)}`;
  return createHash("sha256").update(input).digest().readUInt32BE(0);
}

function pickBalanced(pool, base, need) {
  const selected = [], remaining = [...pool];
  while (remaining.length && selected.length < need) {
    const current = base.concat(selected);
    const { fieldCounts, combos } = countsFor(current);
    let bestIndex = 0, bestVector = null;
    for (let i = 0; i < remaining.length; i++) {
      const r = remaining[i];
      const missing = activeMetaFields.reduce((n, f) => n + (!text(r[f]) ? 1 : 0), 0);
      const individual = activeMetaFields.map((f) => fieldCounts[f].get(text(r[f]) || "(없음)") || 0);
      const vector = [
        combos.get(comboKey(r)) || 0,                     // 동일 조합 반복 최소화
        Math.max(...individual),                          // 한 값으로의 최대 쏠림 최소화
        individual.reduce((a, b) => a + b, 0),            // 전체 희소 값 우선
        missing,                                          // 결측 메타데이터 후순위
        seededTieBreak(r),                                 // 동일 우선순위: 고정 seed 랜덤 tie-break
      ];
      const less = !bestVector || vector.some((v, j) => v < bestVector[j] && vector.slice(0, j).every((x, k) => x === bestVector[k]));
      if (less) { bestIndex = i; bestVector = vector; }
    }
    selected.push(remaining.splice(bestIndex, 1)[0]);
  }
  return selected;
}

function preflightDistribution(rows) {
  if (!rows.length) return "(이미지 없음)";
  const total = rows.length;
  const { fieldCounts } = countsFor(rows);
  return activeMetaFields.map((field) => {
    const values = [...fieldCounts[field].entries()]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], "ko"))
      .map(([value, count]) => `${value}:${count}(${(count / total * 100).toFixed(1)}%)`)
      .join(" | ");
    return `${field}=[${values}]`;
  }).join("\n    ");
}

function distributionText(rows, field) {
  if (!activeMetaFields.includes(field)) return "(다양성 점수 제외)";
  const counts = countsFor(rows).fieldCounts[field];
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], "ko"))
    .map(([v, n]) => `${v}:${n}`).join(" | ");
}

function warnings(rows) {
  const result = [];
  if (!rows.length) return "NO_IMAGES";
  for (const field of activeMetaFields) {
    const counts = countsFor(rows).fieldCounts[field];
    const missing = counts.get("(없음)") || 0;
    if (missing) result.push(`${field} 결측 ${missing}`);
    const valid = [...counts.entries()].filter(([v]) => v !== "(없음)");
    if (!valid.length) result.push(`${field} 조건 없음`);
    else {
      const [topValue, topCount] = valid.sort((a, b) => b[1] - a[1])[0];
      if (topCount / rows.length >= 0.6) result.push(`${field} '${topValue}' ${Math.round(topCount / rows.length * 100)}% 집중`);
    }
  }
  return result.length ? result.join("; ") : "";
}

function buildReports(allRows, existingTrainCounts, target, matchingWarnings = new Map()) {
  const grouped = new Map();
  for (const r of allRows) {
    if (!grouped.has(r.category_id)) grouped.set(r.category_id, []);
    grouped.get(r.category_id).push(r);
  }
  const ids = [...new Set([...grouped.keys(), ...existingTrainCounts.keys()])].sort((a, b) =>
    Number.isFinite(Number(a)) && Number.isFinite(Number(b)) ? Number(a) - Number(b) : a.localeCompare(b, "ko"));
  const summary = [], distribution = [];
  for (const id of ids) {
    const rows = grouped.get(id) || [];
    const drugName = rows.find((r) => r.drug_name)?.drug_name || "";
    const trainCount = existingTrainCounts.get(id) || 0;
    const finalCount = rows.length;
    // class_summary의 "신규"는 이번 실행분이 아니라 기존 Train 외 누적 선별분이다.
    // 따라서 incremental 실행 후에도 existing + new = final 관계가 유지된다.
    const newCount = Math.max(0, finalCount - trainCount);
    const la75 = rows.filter((r) => Number(r.camera_la) === 75).length;
    summary.push({
      category_id: id, drug_name: drugName, existing_image_count: trainCount,
      new_selected_image_count: newCount, final_image_count: finalCount,
      target_count: target, achievement_rate: finalCount / target,
      drug_dir_distribution: distributionText(rows, "drug_dir"),
      camera_la_distribution: distributionText(rows, "camera_la"),
      camera_lo_distribution: distributionText(rows, "camera_lo"),
      light_color_distribution: distributionText(rows, "light_color"),
      size_distribution: distributionText(rows, "size"),
      camera_la_75_ratio: finalCount ? la75 / finalCount : 0,
      unique_metadata_combination_count: new Set(rows.map(comboKey)).size,
      warning: [warnings(rows), ...(matchingWarnings.get(id) || [])].filter(Boolean).join("; "),
    });
    for (const field of activeMetaFields) {
      const counts = countsFor(rows).fieldCounts[field];
      for (const [value, count] of [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], "ko"))) {
        distribution.push({ category_id: id, drug_name: drugName, metadata_field: field,
          metadata_value: value, count, percentage: finalCount ? count / finalCount : 0 });
      }
    }
  }
  return { summary, distribution };
}

function matrix(rows, headers) {
  return [headers, ...rows.map((r) => headers.map((h) => r[h] ?? ""))];
}

function styleSheet(sheet, rows, cols, percentColumns = []) {
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  const used = sheet.getRangeByIndexes(0, 0, Math.max(rows, 1), cols);
  used.format.font = { name: "Aptos", size: 10, color: "#1F2937" };
  const header = sheet.getRangeByIndexes(0, 0, 1, cols);
  header.format = {
    fill: "#1F4E78", font: { bold: true, color: "#FFFFFF", size: 10 },
    rowHeight: 28, verticalAlignment: "center", wrapText: true,
    borders: { preset: "outside", style: "thin", color: "#17365D" },
  };
  used.format.autofitColumns();
  used.format.autofitRows();
  for (let c = 0; c < cols; c++) {
    const col = sheet.getRangeByIndexes(0, c, Math.max(rows, 1), 1);
    if ((col.format.columnWidth ?? 0) > 45) col.format.columnWidth = 45;
  }
  header.format.rowHeight = 42;
  for (const c of percentColumns) sheet.getRangeByIndexes(1, c, Math.max(rows - 1, 1), 1).format.numberFormat = "0.0%";
  if (rows > 1) sheet.getRangeByIndexes(1, 0, rows - 1, cols).format.borders = {
    insideHorizontal: { style: "thin", color: "#E5E7EB" },
  };
}

async function writeWorkbook(tempPath, report, selectedRows, extraHeaders) {
  const wb = Workbook.create();
  const s1 = wb.worksheets.add("class_summary");
  const s2 = wb.worksheets.add("metadata_distribution");
  const s3 = wb.worksheets.add("selected_images");
  const summaryHeaders = Object.keys(report.summary[0] || {
    category_id: "", drug_name: "", existing_image_count: "", new_selected_image_count: "",
    final_image_count: "", target_count: "", achievement_rate: "", drug_dir_distribution: "",
    camera_la_distribution: "", camera_lo_distribution: "", light_color_distribution: "",
    size_distribution: "", camera_la_75_ratio: "", unique_metadata_combination_count: "", warning: "",
  });
  const distHeaders = ["category_id", "drug_name", "metadata_field", "metadata_value", "count", "percentage"];
  const selectedHeaders = [...REQUIRED_OUTPUT, ...extraHeaders.filter((h) => !REQUIRED_OUTPUT.includes(h) && !h.startsWith("__"))];
  const blocks = [
    [s1, matrix(report.summary, summaryHeaders), [6, 12]],
    [s2, matrix(report.distribution, distHeaders), [5]],
    [s3, matrix(selectedRows, selectedHeaders), []],
  ];
  for (const [sheet, values, pct] of blocks) {
    sheet.getRangeByIndexes(0, 0, values.length, values[0].length).values = values;
    styleSheet(sheet, values.length, values[0].length, pct);
  }
  for (const headerName of ["crop_path", "selected_path"]) {
    const column = selectedHeaders.indexOf(headerName);
    if (column >= 0) {
      const range = s3.getRangeByIndexes(1, column, Math.max(selectedRows.length, 1), 1);
      range.format.wrapText = true;
      range.format.columnWidth = 45;
    }
  }
  if (selectedRows.length) s3.getRangeByIndexes(1, 0, selectedRows.length, selectedHeaders.length).format.autofitRows();
  if (report.summary.length) {
    s1.getRange(`O2:O${report.summary.length + 1}`).conditionalFormats.add("containsText", {
      text: "집중", format: { fill: "#FCE8E6", font: { color: "#B91C1C" } },
    });
    s1.getRange(`G2:G${report.summary.length + 1}`).conditionalFormats.add("cellIs", {
      operator: "lessThan", formula: 1, format: { fill: "#FFF2CC", font: { color: "#9A6700" } },
    });
  }
  const blob = await SpreadsheetFile.exportXlsx(wb);
  await blob.save(tempPath);
  return wb;
}

async function validateWorkbook(file, expectedSelectedRows) {
  const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(file));
  const names = wb.worksheets.items.map((s) => s.name);
  for (const name of ["class_summary", "metadata_distribution", "selected_images"])
    if (!names.includes(name)) throw new Error(`임시 Excel 검증 실패: ${name} 시트 없음`);
  const values = wb.worksheets.getItem("selected_images").getUsedRange(true)?.values ?? [];
  const headers = (values[0] || []).map(text);
  for (const h of REQUIRED_OUTPUT) if (!headers.includes(h)) throw new Error(`임시 Excel 검증 실패: ${h} 컬럼 없음`);
  if (values.length - 1 !== expectedSelectedRows) throw new Error(`임시 Excel 행 수 불일치: 기대 ${expectedSelectedRows}, 실제 ${values.length - 1}`);
  const fileCol = headers.indexOf("file_name");
  const categoryCol = headers.indexOf("category_id");
  const sourceCol = headers.indexOf("source");
  // 기존 Train은 서로 다른 category_id에 같은 촬영 원본 file_name이 존재할 수 있다.
  // 이를 유지하되 추가 TS에서는 file_name 전역 중복을 허용하지 않는다.
  const tsFiles = values.slice(1).filter((r) => !keyText(r[sourceCol]).includes("train"))
    .map((r) => keyText(r[fileCol]));
  if (new Set(tsFiles).size !== tsFiles.length) throw new Error("임시 Excel 검증 실패: 추가 TS file_name 중복");
  const trainKeys = values.slice(1).filter((r) => keyText(r[sourceCol]).includes("train"))
    .map((r) => `${keyText(r[categoryCol])}\u241f${keyText(r[fileCol])}`);
  if (new Set(trainKeys).size !== trainKeys.length) throw new Error("임시 Excel 검증 실패: Train category_id+file_name 중복");
  const selectedPathCol = headers.indexOf("selected_path");
  for (const row of values.slice(1)) {
    const selectedPath = text(row[selectedPathCol]);
    if (!selectedPath || !await isNonemptyFile(selectedPath))
      throw new Error(`임시 Excel 검증 실패: 실제 복사 Crop 없음: ${selectedPath || "(빈 경로)"}`);
  }
  return wb;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  await loadRuntimeDependencies();
  const excludedCategoryIds = new Set(args.excludeCategoryIds.map(classKey));
  const output = path.resolve(args.output);
  const selectedDir = path.resolve(args.selectedDir || path.join(path.dirname(output), "output", "train_images"));
  const trainRaw = await readRows(args.train, args.trainSheet);
  const tsRaw = (await Promise.all(args.ts.map((f) => readRows(f, args.tsSheet)))).flat();
  const train = normalizeRows(trainRaw, "Train", "train");
  let ts = normalizeRows(tsRaw, "TS", "ts");
  validateInput(train, "Train"); validateInput(ts, "TS");

  const [trainCropIndex, tsCropIndex] = await Promise.all([
    buildCropIndex(args.trainCropDirs, "Train"), buildCropIndex(args.tsCropDirs, "TS"),
  ]);
  const tsCropMetadataPath = await resolveTsCropMetadata(args);
  let canonicalStats = { metadataRows: 0, canonicalEntries: 0, applied: 0, canonicalMissing: 0,
    duplicateMetadataKeys: 0, duplicatePhysicalIgnored: 0, duplicateCanonicalLinked: 0, duplicateByClass: new Map() };
  if (tsCropMetadataPath) {
    const canonicalRaw = await readRows(tsCropMetadataPath);
    const canonicalRows = normalizeRows(canonicalRaw, "TS", "ts_crop_metadata");
    const canonicalResult = applyCanonicalCropPaths(ts, canonicalRows, tsCropIndex);
    ts = canonicalResult.rows; canonicalStats = canonicalResult.stats;
    console.log(`TS crop_path 정본 metadata: ${tsCropMetadataPath}`);
    console.log(`정본 경로 결합: ${canonicalStats.applied}/${ts.length}행, 물리 중복 자동 제외 ${canonicalStats.duplicatePhysicalIgnored}개 (정본 metadata 연결 ${canonicalStats.duplicateCanonicalLinked}개)`);
  } else {
    console.warn("WARNING: TS crop_metadata.csv를 찾지 못해 상대경로/basename fallback을 사용합니다.");
  }

  const annotationResult = await enrichFromAnnotations(ts, args.tsAnnotationDir);
  ts = annotationResult.rows;
  console.log(`TS Annotation 촬영조건 결합: 성공 ${annotationResult.stats.matched}/${ts.length}, 누락 ${annotationResult.stats.missing}, 파싱 실패 ${annotationResult.stats.parseFailed}`);

  const sizePresent = ts.filter((r) => text(r.size)).length;
  if (sizePresent !== ts.length) {
    activeMetaFields = META_FIELDS.filter((f) => f !== "size");
    console.warn(`WARNING: TS metadata size 사용 가능 ${sizePresent}/${ts.length}행 — size를 추정하지 않고 다양성 점수에서 제외합니다.`);
  } else {
    activeMetaFields = [...META_FIELDS];
    console.log(`다양성 점수 필드: ${activeMetaFields.join(", ")} (size ${sizePresent}/${ts.length}행 사용 가능)`);
  }
  console.log(`실제 다양성 점수 필드: ${activeMetaFields.join(", ")}`);

  let prior = [];
  try { await fs.access(args.output); prior = normalizeRows(await rowsFromWorkbook(args.output, "selected_images"), "TS", "prior"); }
  catch (e) { if (e?.code !== "ENOENT") throw e; }
  validateInput(prior, "기존 Excel");
  const priorReconciliation = await reconcilePriorWithOutput(prior, selectedDir, args.flatOutput);
  const priorValid = priorReconciliation.valid;
  console.log(`기존 Excel-실파일 대조: Excel ${prior.length}행 | 실제 유효 ${priorValid.length}장 | 삭제/누락 ${priorReconciliation.missing.length}장`);

  const matchWarnings = [], usedCropPaths = new Set();
  const trainMatchStats = { attempted: 0, matched: 0, failed: 0, duplicateIgnored: 0, byClass: new Map() };
  const tsMatchStats = { attempted: 0, matched: 0, failed: 0, duplicateIgnored: 0, byClass: new Map() };
  const trainUniqueRaw = dedupeTrainCrops(train);
  const trainUnique = await attachCropMatches(trainUniqueRaw, trainCropIndex, matchWarnings, usedCropPaths, false, trainMatchStats);
  const priorTrainRaw0 = priorValid.filter((r) => keyText(r.source).includes("train"));
  const priorTsRaw0 = priorValid.filter((r) => !keyText(r.source).includes("train"));
  const priorUniqueRaw = [...dedupeTrainCrops(priorTrainRaw0), ...dedupeStable(priorTsRaw0)];
  const priorTrainRaw = priorUniqueRaw.filter((r) => keyText(r.source).includes("train"));
  const priorTsRaw = priorUniqueRaw.filter((r) => !keyText(r.source).includes("train"));
  const priorUnique = [
    ...await attachCropMatches(priorTrainRaw, trainCropIndex, matchWarnings, usedCropPaths, true),
    ...await attachCropMatches(priorTsRaw, tsCropIndex, matchWarnings, usedCropPaths, true),
  ];
  // 기존 Excel이 있으면 그 행과 순서를 정본으로 유지한다. 원본 Train에서 Excel에
  // 아직 없는 행만 보완하여 incremental 재실행 시 Train을 이중 계산하지 않는다.
  const priorTrainKeys = new Set(prior.filter((r) => keyText(r.source).includes("train")).map(trainImageKey));
  // Incremental 실행에서는 기존 Excel을 선택 결과의 정본으로 삼는다. Excel에 없는
  // Train 행을 자동 재삽입하면 사용자가 제거한 이미지를 되살릴 수 있으므로 추가하지 않는다.
  const missingTrainRows = (prior.length ? [] : trainUnique)
    .filter((r) => !excludedCategoryIds.has(classKey(r.category_id)));
  const base = prior.length ? priorUnique.concat(missingTrainRows) : trainUnique;
  // 출력 폴더에 이미 존재하지만 과거 Excel에 누락된 Train 행도 실제 경로를 기록한다.
  // 파일이 없는 경우에는 최종 검증에서 조용히 통과하지 않고 실패하도록 빈 경로를 유지한다.
  for (const row of base) {
    if (!text(row.selected_path)) {
      const expected = selectedOutputPath(row, selectedDir, args.flatOutput);
      if (await isNonemptyFile(expected)) row.selected_path = expected;
    }
  }
  const baseByClass = new Map();
  for (const r of base) { if (!baseByClass.has(r.category_id)) baseByClass.set(r.category_id, []); baseByClass.get(r.category_id).push(r); }
  const trainCounts = new Map();
  for (const r of trainUnique) trainCounts.set(r.category_id, (trainCounts.get(r.category_id) || 0) + 1);
  const priorMaxRound = Math.max(0, ...priorUnique.map((r) => Number(r.selection_round) || 0));
  const nextRound = priorMaxRound + 1;
  const tsMatched = await attachCropMatches(ts, tsCropIndex, matchWarnings, usedCropPaths, false, tsMatchStats);
  // 삭제된 부적합 이미지도 다시 뽑히지 않도록 기존 Excel의 모든 file_name은 후보에서 제외한다.
  const selectedPriorNames = new Set(prior.map((r) => keyText(r.file_name)));
  const priorOverlapCount = tsMatched.filter((r) => selectedPriorNames.has(keyText(r.file_name))).length;
  const explicitlyExcluded = tsMatched.filter((r) => excludedCategoryIds.has(classKey(r.category_id))).length;
  const neededClassIds = new Set([...baseByClass.entries()]
    .filter(([, rows]) => rows.length < args.target).map(([id]) => id));
  for (const row of tsMatched) if (!baseByClass.has(row.category_id)) neededClassIds.add(row.category_id);
  const rawEligible = dedupeStable(tsMatched.filter((r) =>
    !excludedCategoryIds.has(classKey(r.category_id)) && neededClassIds.has(r.category_id)), selectedPriorNames);
  const contentResult = await filterInvalidCropContent(rawEligible);
  const eligible = contentResult.valid;
  const contentRejectedByClass = new Map();
  for (const row of contentResult.rejected)
    contentRejectedByClass.set(row.category_id, (contentRejectedByClass.get(row.category_id) || 0) + 1);
  console.log(`TS 후보 필터: back_color 제한 없음, 매칭된 전체 TS 사용 (고정 random seed=${RANDOM_SEED})`);
  console.log(`CLI 제외 category_id: ${args.excludeCategoryIds.length ? args.excludeCategoryIds.join(", ") : "없음"} | 제외 TS ${explicitlyExcluded}장`);
  console.log(`내용 품질 검사: 검사 ${rawEligible.length}장 | 알약 없음/검정 화면/배경만 제외 ${contentResult.rejected.length}장 | 통과 ${eligible.length}장`);
  console.log(`증분 후보 요약: 현재 TS Crop ${tsMatched.length}장 | 기존 선택과 file_name 중복 제외 ${priorOverlapCount}장 | 최종 신규 평가 가능 Crop ${eligible.length}장`);
  const tsByClass = new Map();
  for (const r of eligible) { if (!tsByClass.has(r.category_id)) tsByClass.set(r.category_id, []); tsByClass.get(r.category_id).push(r); }

  const naming = validateDrugNames(trainUnique, tsMatched);
  for (const [id, names] of naming.conflicts)
    console.warn(`WARNING: category_id=${id} 약품명 충돌 — 생성 예정 폴더를 확정하지 않음: ${[...names].join(" | ")}`);
  const classIds = [...new Set([...baseByClass.keys(), ...tsByClass.keys()])];
  const newlySelected = [];
  for (const id of classIds) {
    const currentRows = baseByClass.get(id) || [];
    const current = currentRows.length;
    const need = Math.max(0, args.target - current);
    const candidateCount = (tsByClass.get(id) || []).length;
    const match = tsMatchStats.byClass.get(id) || { matched: 0, failed: 0, duplicateIgnored: 0 };
    const names = naming.byClass.get(id) || new Set();
    const priorStat = priorReconciliation.byClass.get(id) || { excel: 0, actual: 0, missing: 0 };
    const contentRejected = contentRejectedByClass.get(id) || 0;
    const folderName = names.size === 1 ? `${safeFolderName(id)}_${safeFolderName([...names][0])}` : "(약품명 충돌/없음으로 미확정)";
    const picked = pickBalanced(tsByClass.get(id) || [], currentRows, need).map((r) => ({
      ...r, source: "TS", selection_round: nextRound,
    }));
    newlySelected.push(...picked);
    const plannedRows = currentRows.concat(picked);
    console.log(`[클래스 ${id}] Excel 기록 ${priorStat.excel}장 | 실제 파일 ${priorStat.actual}장 | 삭제 차이 ${priorStat.missing}장 | 현재 유효 ${current}장 | 부족 ${need}장 | 품질 제외 ${contentRejected}장 | TS 보충 ${picked.length}장 | 최종 예상 ${plannedRows.length}장`);
    console.log(`  생성 예정 폴더명: ${folderName}`);
  }

  const plannedRows = base.concat(newlySelected);
  const globalFileDuplicates = duplicateCount(plannedRows.map((r) => keyText(r.file_name)));
  const tsFileDuplicates = duplicateCount(plannedRows.filter((r) => !keyText(r.source).includes("train")).map((r) => keyText(r.file_name)));
  console.log(`선택 결과 file_name 중복 검증: 전체 ${globalFileDuplicates}건, TS ${tsFileDuplicates}건`);

  console.log(`TS crop_path 매칭 합계: 성공 ${tsMatchStats.matched}/${tsMatchStats.attempted}, 실패 ${tsMatchStats.failed}`);
  console.log(`동일 basename 복제본 자동 제외: 현재 TS metadata ${tsMatchStats.duplicateIgnored}개 / crop_metadata 전체 ${canonicalStats.duplicatePhysicalIgnored}개`);
  for (const [id, count] of [...canonicalStats.duplicateByClass.entries()].sort((a, b) => Number(a[0]) - Number(b[0])))
    console.log(`  - crop_metadata class ${id}: 정본 crop_path 외 동일 basename 복제본 ${count}개 자동 무시`);

  const unmatchedTrainCrops = trainCropIndex.files.filter((f) => !usedCropPaths.has(f));
  const unmatchedTsCrops = tsCropIndex.files.filter((f) => !usedCropPaths.has(f));
  console.log(`Crop 매칭 warning: metadata ${matchWarnings.length}건, 현재 --train 입력 미참조 Crop ${unmatchedTrainCrops.length}개, 현재 --ts 입력 미참조 Crop ${unmatchedTsCrops.length}개`);
  for (const warning of matchWarnings.slice(0, 20)) console.warn(`WARNING: ${warning.message}`);
  for (const file of unmatchedTrainCrops.slice(0, 5)) console.warn(`WARNING: 현재 --train 입력에서 미참조된 Crop: ${file}`);
  for (const file of unmatchedTsCrops.slice(0, 5)) console.warn(`WARNING: 현재 --ts 입력에서 미참조된 Crop(전체 crop_metadata orphan 의미 아님): ${file}`);

  if (args.dryRun) {
    console.log(`선택 이미지 출력 예정: ${selectedDir}`);
    console.log(`\nDRY-RUN: Excel/이미지를 수정하지 않았습니다. 대상 클래스 ${classIds.length}, 신규 TS 선택 예정 ${newlySelected.length}장`);
    return;
  }

  if (naming.conflicts.size) throw new Error("category_id-약품명 충돌이 있어 실제 폴더 생성/복사를 중단합니다. preflight warning을 해결하세요.");

  // 증분 실행에서는 기존 출력 파일을 건드리지 않고 이번 round 신규 선택분만 복사한다.
  const copyResult = await copySelectedImages(newlySelected, selectedDir, args.flatOutput);
  const finalRows = base.concat(newlySelected);
  const allHeaders = [...new Set(finalRows.flatMap((r) => Object.keys(r)))];
  const warningsByClass = new Map();
  for (const item of matchWarnings) {
    if (!warningsByClass.has(item.category_id)) warningsByClass.set(item.category_id, []);
    warningsByClass.get(item.category_id).push(item.message);
  }
  const report = buildReports(finalRows, trainCounts, args.target, warningsByClass);
  await fs.mkdir(path.dirname(output), { recursive: true });
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
  const temp = `${output}.tmp-${process.pid}-${stamp}.xlsx`;
  await writeWorkbook(temp, report, finalRows, allHeaders);
  await validateWorkbook(temp, finalRows.length);

  let backup = "";
  try {
    await fs.access(output);
    const ext = path.extname(output), stem = output.slice(0, -ext.length);
    backup = `${stem}.backup-${stamp}${ext}`;
    await fs.copyFile(output, backup);
  } catch (e) { if (e?.code !== "ENOENT") { await fs.rm(temp, { force: true }); throw e; } }
  await fs.rename(temp, output);
  console.log(`저장 완료: ${output}`);
  console.log(`Train Crop 구성: 신규 복사 ${copyResult.copied}장, 기존 파일 재사용 ${copyResult.reused}장 → ${selectedDir}`);
  if (backup) console.log(`기존 Excel 백업: ${backup}`);
  console.log(`검증 완료: selected_images ${finalRows.length}행, 필수 컬럼 ${REQUIRED_OUTPUT.length}개, file_name 중복 0건`);
}

main().catch((error) => { console.error(error?.stack || error); process.exitCode = 1; });
