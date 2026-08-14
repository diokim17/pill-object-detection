#!/usr/bin/env python3
"""추가 알약 이미지의 BBox/object 품질을 U2Net(rembg)으로 증분 분석한다.

입력: 원본 이미지 폴더, BBox/클래스 CSV, 기존 분석 Excel, 선택적 Annotation 폴더.
출력: 누적 Excel/CSV, suspect·bad 리뷰 이미지, good Train 후보 복사본.
처리: 기존 file_name 제외 → U2Net 객체 분할 → good/suspect/bad 판정 →
      annotation 중복 제거 → 클래스 통계 재계산 → 원자적 저장.

This is a new, read-only-input workflow. It does not modify the existing
extraction/crop scripts, source PNGs, metadata CSV, or Annotation JSON files.

Preflight (analysis only; writes nothing):
  python analyze_bbox_quality.py --image-dir IMAGES --csv METADATA.csv \
    --existing-excel bbox_quality_analysis.xlsx --u2net-home MODEL_CACHE \
    --review-root REVIEW --candidate-root CANDIDATES --preflight

Class-balanced limited preflight (for example, 100 current local images):
  .venv-rembg/bin/python analyze_additional_train_candidates_local.py --preflight --limit 100

Create review artifacts, reports, and copied Train candidates only after a
successful preflight:
  .venv-rembg/bin/python analyze_additional_train_candidates_local.py --execute

Incremental behavior: an existing bbox_quality_analysis.xlsx is read first;
file_name values already present in image_quality are skipped. Newly analyzed
rows are appended, then duplicate annotations are removed by
file_name+bbox_x+bbox_y+bbox_w+bbox_h before the workbook is atomically saved.

The CSV is the sole source of truth for bbox and existing Train
category_id/category_name. Capture conditions absent from the CSV are read only
from the matching Annotation JSON images[] record. dl_idx and K-code numbers are
never used as class IDs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw, ImageFont

new_session = None
remove = None


def load_rembg() -> None:
    """CLI 도움말은 의존성 없이 표시하고, 실제 분석 직전에 rembg를 불러온다."""
    global new_session, remove
    try:
        from rembg import new_session as rembg_new_session, remove as rembg_remove
    except ImportError as exc:
        raise SystemExit("rembg가 필요합니다: pip install rembg onnxruntime") from exc
    new_session, remove = rembg_new_session, rembg_remove

try:
    from scipy import ndimage
except ImportError as exc:
    raise SystemExit("scipy가 필요합니다. rembg 가상환경으로 실행하세요.") from exc

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(items, total=None, desc="진행", unit="개"):
        yield from items


CAPTURE_FIELDS = ("back_color", "light_color", "camera_la", "camera_lo", "drug_dir")
REQUIRED_FIELDS = {
    "json_file", "file_name", "item_seq", "category_id", "category_name",
    "image_width", "image_height", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
}
QUALITY_HEADERS = [
    "file_name", "source_path", "category_id", "category_name", "item_seq",
    "bbox_quality", "quality_reason", "mask_outside_bbox_ratio",
    "mask_edge_contact_ratio", "segmentation_confidence",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
]
SUMMARY_HEADERS = [
    "category_id", "category_name", "total_images", "good_count",
    "suspect_count", "bad_count", "good_ratio", "suspect_ratio", "bad_ratio",
]
SELECTED_HEADERS = [
    "file_name", "category_id", "category_name", "item_seq", "back_color",
    "light_color", "camera_la", "camera_lo", "drug_dir", "bbox_quality",
    "selection_reason",
]


def norm(value: object) -> str:
    return unicodedata.normalize("NFC", str(value if value is not None else "").strip())


def safe_folder(value: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", norm(value))


@dataclass
class Record:
    file_name: str
    json_file: str
    item_seq: str
    category_id: int
    category_name: str
    image_width: int
    image_height: int
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    back_color: str = ""
    light_color: str = ""
    camera_la: str = ""
    camera_lo: str = ""
    drug_dir: str = ""
    source_path: Path | None = None
    quality: str = ""
    quality_reason: str = ""
    mask_area: int = 0
    mask_outside_bbox_ratio: float = 0.0
    mask_edge_contact_ratio: float = 0.0
    segmentation_confidence: str = ""
    dhash: int = 0

    @property
    def bbox_x2(self) -> float: return self.bbox_x + self.bbox_w

    @property
    def bbox_y2(self) -> float: return self.bbox_y + self.bbox_h


def resolve_annotation_dir(explicit: Path | None) -> Path | None:
    if explicit:
        path = explicit.expanduser().resolve()
        return path if path.is_dir() else None
    return None


def image_index(root: Path) -> tuple[list[Path], dict[str, list[Path]]]:
    files = sorted(p for p in root.rglob("*.png") if p.is_file())
    by_name: dict[str, list[Path]] = defaultdict(list)
    for path in files: by_name[norm(path.name)].append(path)
    return files, by_name


def annotation_index(root: Path | None) -> dict[str, Path]:
    if root is None: return {}
    index: dict[str, Path] = {}
    duplicates = set()
    for path in root.rglob("*.json"):
        key = norm(path.name)
        if key in index: duplicates.add(key)
        else: index[key] = path
    if duplicates:
        raise RuntimeError(f"Annotation JSON basename 중복: {len(duplicates):,}개")
    return index


def load_records(csv_path: Path, ann_index: dict[str, Path]) -> tuple[list[Record], list[str]]:
    records=[]; errors=[]
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader=csv.DictReader(handle)
        missing=REQUIRED_FIELDS-set(reader.fieldnames or [])
        if missing: raise ValueError(f"CSV 필수 컬럼 누락: {sorted(missing)}")
        csv_has_capture=all(f in (reader.fieldnames or []) for f in CAPTURE_FIELDS)
        for line,row in enumerate(reader,2):
            try:
                capture={field:norm(row.get(field,"")) for field in CAPTURE_FIELDS}
                if not csv_has_capture or any(not capture[f] for f in CAPTURE_FIELDS):
                    ann_path=ann_index.get(norm(PurePosixPath(row["json_file"]).name))
                    if ann_path is None:
                        errors.append(f"{line}행 촬영조건 Annotation 없음: {row['json_file']}")
                        continue
                    data=json.loads(ann_path.read_text(encoding="utf-8"))
                    matches=[im for im in data.get("images",[]) if norm(PurePosixPath(str(im.get("file_name",""))).name)==norm(PurePosixPath(row["file_name"]).name)]
                    if len(matches)!=1:
                        errors.append(f"{line}행 images[] 매칭 {len(matches)}개: {row['file_name']}")
                        continue
                    capture.update({f:norm(matches[0].get(f,"")) for f in CAPTURE_FIELDS})
                records.append(Record(
                    file_name=norm(PurePosixPath(row["file_name"]).name),
                    json_file=norm(row["json_file"]), item_seq=norm(row["item_seq"]),
                    category_id=int(row["category_id"]), category_name=norm(row["category_name"]),
                    image_width=int(row["image_width"]), image_height=int(row["image_height"]),
                    bbox_x=float(row["bbox_x"]), bbox_y=float(row["bbox_y"]),
                    bbox_w=float(row["bbox_w"]), bbox_h=float(row["bbox_h"]), **capture,
                ))
            except Exception as exc: errors.append(f"CSV {line}행 오류: {exc}")
    return records,errors


def largest_component(mask: np.ndarray) -> np.ndarray:
    labeled,count=ndimage.label(mask,structure=np.ones((3,3),dtype=np.uint8))
    if count==0:return np.zeros_like(mask,dtype=bool)
    sizes=np.bincount(labeled.ravel());sizes[0]=0
    return labeled==int(np.argmax(sizes))


def segment_expanded_roi(image: Image.Image, record: Record, session) -> tuple[np.ndarray, tuple[int,int,int,int], str]:
    # Padding lets the model reveal pill pixels immediately outside the CSV bbox.
    pad_x=max(12,int(round(record.bbox_w*.35)));pad_y=max(12,int(round(record.bbox_h*.35)))
    left=max(0,int(math.floor(record.bbox_x-pad_x)));top=max(0,int(math.floor(record.bbox_y-pad_y)))
    right=min(image.width,int(math.ceil(record.bbox_x2+pad_x)));bottom=min(image.height,int(math.ceil(record.bbox_y2+pad_y)))
    roi=image.crop((left,top,right,bottom)).convert("RGB")
    alpha=np.asarray(remove(roi,session=session,only_mask=True).convert("L"))
    hard=largest_component(alpha>=96)
    if not hard.any(): return hard,(left,top,right,bottom),"empty"
    area=int(hard.sum());fraction=area/hard.size
    confidence="normal" if .015<=fraction<=.85 else "uncertain_area"
    return hard,(left,top,right,bottom),confidence


def analyze_one(record: Record, session) -> None:
    assert record.source_path is not None
    try:
        with Image.open(record.source_path) as raw:
            raw.verify()
        with Image.open(record.source_path) as image:
            image=image.convert("RGB")
            if image.size!=(record.image_width,record.image_height):
                record.quality="bad";record.quality_reason=f"이미지 크기 불일치 actual={image.size} csv={(record.image_width,record.image_height)}";return
            values=(record.bbox_x,record.bbox_y,record.bbox_w,record.bbox_h,record.bbox_x2,record.bbox_y2)
            valid=all(math.isfinite(x) for x in values) and record.bbox_x>=0 and record.bbox_y>=0 and record.bbox_w>1 and record.bbox_h>1 and record.bbox_x2<=image.width and record.bbox_y2<=image.height
            if not valid:
                record.quality="bad";record.quality_reason="bbox 좌표/크기 명백한 오류";return
            mask,roi_box,confidence=segment_expanded_roi(image,record,session)
            record.segmentation_confidence=confidence;record.mask_area=int(mask.sum())
            if not mask.any():
                record.quality="suspect";record.quality_reason="자동 객체 분할 불확실(empty); 수동 검토 필요";return
            left,top,right,bottom=roi_box
            bx1=max(0,int(math.floor(record.bbox_x-left)));by1=max(0,int(math.floor(record.bbox_y-top)))
            bx2=min(mask.shape[1],int(math.ceil(record.bbox_x2-left)));by2=min(mask.shape[0],int(math.ceil(record.bbox_y2-top)))
            inside=np.zeros_like(mask);inside[by1:by2,bx1:bx2]=True
            outside=int((mask & ~inside).sum());record.mask_outside_bbox_ratio=outside/max(record.mask_area,1)
            # Contact ratio measures substantial foreground along crop edges, not a single antialias pixel.
            crop_mask=mask[by1:by2,bx1:bx2]
            band=max(2,int(round(min(crop_mask.shape)*.025)))
            edge=np.zeros_like(crop_mask);edge[:band]=True;edge[-band:]=True;edge[:,:band]=True;edge[:,-band:]=True
            record.mask_edge_contact_ratio=float((crop_mask&edge).sum()/max(crop_mask.sum(),1))
            out=record.mask_outside_bbox_ratio;edge_ratio=record.mask_edge_contact_ratio
            if confidence!="normal":
                record.quality="suspect";record.quality_reason=f"분할 면적 불확실; outside={out:.3f}, edge={edge_ratio:.3f}"
            elif out>=.22:
                record.quality="bad";record.quality_reason=f"알약 mask의 {out:.1%}가 bbox 밖; 상당 부분 잘림 가능"
            elif out>=.035 or edge_ratio>=.16:
                record.quality="suspect";record.quality_reason=f"bbox 경계 접촉/외부 mask; outside={out:.3f}, edge={edge_ratio:.3f}"
            else:
                record.quality="good";record.quality_reason=f"객체가 bbox 내부에 충분히 포함; outside={out:.3f}, edge={edge_ratio:.3f}"
            crop=image.crop((int(round(record.bbox_x)),int(round(record.bbox_y)),int(round(record.bbox_x2)),int(round(record.bbox_y2))))
            record.dhash=dhash(crop)
    except Exception as exc:
        record.quality="bad";record.quality_reason=f"이미지 읽기/분석 오류: {exc}"


def dhash(image: Image.Image) -> int:
    gray=np.asarray(image.convert("L").resize((9,8),Image.Resampling.LANCZOS))
    bits=(gray[:,1:]>gray[:,:-1]).ravel();value=0
    for bit in bits:value=(value<<1)|int(bit)
    return value


def hamming(a: int,b: int) -> int: return (a^b).bit_count()


def select_candidates(good: list[Record], limit: int=100) -> list[tuple[Record,str]]:
    selected=[];selected_ids=set();condition_counts=Counter();seen_la=set();seen_lo=set();seen_dir=set();seen_tuple=set()
    def add(record,reason):
        selected.append((record,reason));selected_ids.add(record.file_name)
        condition_counts[(record.back_color,record.light_color,record.camera_la,record.camera_lo,record.drug_dir)]+=1
        seen_la.add(record.camera_la);seen_lo.add(record.camera_lo);seen_dir.add(record.drug_dir);seen_tuple.add((record.camera_la,record.camera_lo,record.drug_dir))
    combos=defaultdict(list)
    for record in good:combos[(record.back_color,record.light_color)].append(record)
    # Phase 1: up to two low-duplicate representatives per background/light pair.
    for combo in sorted(combos):
        pool=sorted(combos[combo],key=lambda r:(r.camera_la,r.camera_lo,r.drug_dir,r.file_name))
        picked=[]
        for record in pool:
            if all(hamming(record.dhash,p.dhash)>5 for p in picked):
                add(record,"back_color × light_color 조합 우선 2장") ;picked.append(record)
                if len(picked)==2 or len(selected)==limit:break
        if len(selected)==limit:return selected
    # Phase 2: greedy condition diversity plus perceptual-distance score.
    remaining=[r for r in good if r.file_name not in selected_ids]
    while remaining and len(selected)<limit:
        best=None;best_score=-10**9
        for r in remaining:
            exact=(r.back_color,r.light_color,r.camera_la,r.camera_lo,r.drug_dir)
            if condition_counts[exact]>=4:continue
            min_distance=min((hamming(r.dhash,s.dhash) for s,_ in selected),default=64)
            if min_distance<=3:continue
            score=(12*(r.camera_la not in seen_la)+12*(r.camera_lo not in seen_lo)+14*(r.drug_dir not in seen_dir)+18*((r.camera_la,r.camera_lo,r.drug_dir) not in seen_tuple)+min(min_distance,20)-8*condition_counts[exact])
            key=(score,r.file_name)
            if best is None or key>(best_score,best.file_name):best=r;best_score=score
        if best is None:break
        add(best,"camera_la/camera_lo/drug_dir 다양성 및 perceptual 중복 억제")
        remaining.remove(best)
    return selected


def quality_rows(records: list[Record]) -> list[dict]:
    rows=[]
    for r in records:
        rows.append({"file_name":r.file_name,"source_path":str(r.source_path or ""),"category_id":r.category_id,"category_name":r.category_name,"item_seq":r.item_seq,"bbox_quality":r.quality,"quality_reason":r.quality_reason,"mask_outside_bbox_ratio":round(r.mask_outside_bbox_ratio,6),"mask_edge_contact_ratio":round(r.mask_edge_contact_ratio,6),"segmentation_confidence":r.segmentation_confidence,"bbox_x":r.bbox_x,"bbox_y":r.bbox_y,"bbox_w":r.bbox_w,"bbox_h":r.bbox_h})
    return rows


def class_stats(records: list[Record]) -> list[dict]:
    groups=defaultdict(list)
    for r in records:groups[(r.category_id,r.category_name)].append(r)
    rows=[]
    for (cid,name),items in sorted(groups.items()):
        counts=Counter(r.quality for r in items);total=len(items)
        rows.append({"category_id":cid,"category_name":name,"total_images":total,"good_count":counts["good"],"suspect_count":counts["suspect"],"bad_count":counts["bad"],"good_ratio":round(counts["good"]/total,6),"suspect_ratio":round(counts["suspect"]/total,6),"bad_ratio":round(counts["bad"]/total,6)})
    return rows


def class_stats_from_detail_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(int(row["category_id"]), norm(row["category_name"]))].append(row)
    result=[]
    for (category_id,category_name),items in sorted(groups.items()):
        counts=Counter(norm(row["bbox_quality"]) for row in items);total=len(items)
        result.append({
            "category_id":category_id,"category_name":category_name,
            "total_images":total,"good_count":counts["good"],
            "suspect_count":counts["suspect"],"bad_count":counts["bad"],
            "good_ratio":round(counts["good"]/total,6),
            "suspect_ratio":round(counts["suspect"]/total,6),
            "bad_ratio":round(counts["bad"]/total,6),
        })
    return result


def excel_key_number(value: object) -> str:
    """Normalize Excel/Python numeric representations for stable dedup keys."""
    try:
        number=float(value)
        if math.isfinite(number): return format(number,".12g")
    except (TypeError,ValueError):
        pass
    return norm(value)


def annotation_key(row: dict) -> tuple[str,str,str,str,str]:
    return (
        norm(row.get("file_name","")), excel_key_number(row.get("bbox_x")),
        excel_key_number(row.get("bbox_y")), excel_key_number(row.get("bbox_w")),
        excel_key_number(row.get("bbox_h")),
    )


def deduplicate_detail_rows(rows: list[dict]) -> tuple[list[dict], int]:
    """Keep the first row for each file_name+bbox coordinate identity."""
    unique=[];seen=set();removed=0
    for row in rows:
        key=annotation_key(row)
        if key in seen:removed+=1;continue
        seen.add(key);unique.append({header:row.get(header,"") for header in QUALITY_HEADERS})
    return unique,removed


def read_sheet_rows(sheet, expected_headers: list[str]) -> list[dict]:
    actual=[cell.value for cell in sheet[1]]
    if actual!=expected_headers:
        raise RuntimeError(
            f"Excel {sheet.title} 컬럼이 기존 규격과 다릅니다. "
            f"expected={expected_headers}, actual={actual}"
        )
    return [dict(zip(actual,row)) for row in sheet.iter_rows(min_row=2,values_only=True)]


def load_existing_excel(path: Path) -> tuple[list[dict], list[dict]]:
    if not path.is_file(): return [],[]
    try: import openpyxl
    except ImportError as exc: raise RuntimeError("Excel 읽기에는 openpyxl이 필요합니다.") from exc
    workbook=openpyxl.load_workbook(path,read_only=True,data_only=False)
    try:
        required={"class_quality_summary","image_quality","selected_candidates"}
        missing=required-set(workbook.sheetnames)
        if missing:raise RuntimeError(f"기존 Excel 필수 시트 누락: {sorted(missing)}")
        details=read_sheet_rows(workbook["image_quality"],QUALITY_HEADERS)
        selected=read_sheet_rows(workbook["selected_candidates"],SELECTED_HEADERS)
        return details,selected
    finally:workbook.close()


def write_csv(path: Path,rows: list[dict]):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else []);writer.writeheader();writer.writerows(rows)


def replace_sheet_rows(sheet,headers: list[str],rows: list[dict]):
    actual=[cell.value for cell in sheet[1]]
    if actual!=headers:
        raise RuntimeError(f"Excel {sheet.title} 컬럼 변경 차단: {actual}")
    if sheet.max_row>1:sheet.delete_rows(2,sheet.max_row-1)
    for row in rows:sheet.append([row.get(header,"") for header in headers])
    sheet.freeze_panes="A2";sheet.auto_filter.ref=sheet.dimensions


def write_excel_incremental(
    path: Path, stats: list[dict], details: list[dict], selected: list[dict]
):
    try: import openpyxl
    except ImportError as exc: raise RuntimeError("Excel 저장에는 openpyxl이 필요합니다.") from exc
    if not path.is_file():
        raise RuntimeError(
            f"기존 Excel이 없어 신규 파일 생성을 차단했습니다: {path}"
        )
    workbook=openpyxl.load_workbook(path)
    replace_sheet_rows(workbook["class_quality_summary"],SUMMARY_HEADERS,stats)
    replace_sheet_rows(workbook["image_quality"],QUALITY_HEADERS,details)
    replace_sheet_rows(workbook["selected_candidates"],SELECTED_HEADERS,selected)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".tmp.xlsx")
    backup=None
    try:
        workbook.save(temporary)
        check=openpyxl.load_workbook(temporary,read_only=True,data_only=False)
        try:
            if check["image_quality"].max_row-1!=len(details):
                raise RuntimeError("임시 Excel image_quality 행 수 검증 실패")
        finally:check.close()
        if path.exists():
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
            backup=path.with_name(f"{path.stem}.backup_{timestamp}{path.suffix}")
            shutil.copy2(path,backup)
        temporary.replace(path)
    finally:
        workbook.close()
        if temporary.exists():temporary.unlink()
    return backup


def review_image(record: Record,destination: Path):
    assert record.source_path
    with Image.open(record.source_path) as source:
        source=source.convert("RGB");thumb=source.copy();thumb.thumbnail((850,650))
        sx=thumb.width/source.width;sy=thumb.height/source.height;draw=ImageDraw.Draw(thumb)
        color=(245,158,11) if record.quality=="suspect" else (220,38,38)
        draw.rectangle((record.bbox_x*sx,record.bbox_y*sy,record.bbox_x2*sx,record.bbox_y2*sy),outline=color,width=4)
        crop=source.crop((int(record.bbox_x),int(record.bbox_y),int(record.bbox_x2),int(record.bbox_y2)));crop.thumbnail((500,500))
        canvas=Image.new("RGB",(max(thumb.width,crop.width)+30,thumb.height+crop.height+100),"white");canvas.paste(thumb,(15,15));canvas.paste(crop,(15,thumb.height+45))
        d=ImageDraw.Draw(canvas);d.text((15,thumb.height+crop.height+55),f"{record.quality}: {record.quality_reason}",fill=color)
    destination.parent.mkdir(parents=True,exist_ok=True);canvas.save(destination)


def write_csv_atomic(path: Path,rows: list[dict],headers: list[str]):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+".tmp")
    with temporary.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=headers);writer.writeheader();writer.writerows(rows)
    temporary.replace(path)


def balanced_limit(records: list[Record], limit: int | None) -> list[Record]:
    """Deterministically sample current records across classes by round-robin.

    This avoids taking the first N CSV rows. Within each class, records are
    ordered by capture conditions and filename so repeated runs select the same
    test set while exposing varied acquisition conditions early.
    """
    if limit is None or limit >= len(records):
        return records
    if limit <= 0:
        raise ValueError("--limit은 1 이상의 정수여야 합니다.")
    groups: dict[tuple[int, str], list[Record]] = defaultdict(list)
    for record in records:
        groups[(record.category_id, record.category_name)].append(record)
    for items in groups.values():
        items.sort(key=lambda r: (
            r.back_color, r.light_color, r.camera_la, r.camera_lo,
            r.drug_dir, r.file_name,
        ))
    class_keys = sorted(groups)
    selected: list[Record] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for key in class_keys:
            items = groups[key]
            if offset < len(items):
                selected.append(items[offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir",type=Path,required=True)
    parser.add_argument("--csv",type=Path,required=True)
    parser.add_argument("--annotation-dir",type=Path)
    parser.add_argument("--review-root",type=Path,required=True)
    parser.add_argument("--candidate-root",type=Path,required=True)
    parser.add_argument("--existing-excel",type=Path,required=True,
                        help="기존 image_quality를 읽고 증분 갱신할 Excel")
    parser.add_argument("--u2net-home",type=Path,required=True,
                        help="기존 전체 분석과 동일한 rembg 모델 캐시 경로")
    parser.add_argument("--model",default="u2net",help="rembg foreground model")
    parser.add_argument("--limit",type=int,
                        help="새 분석 대상 중 클래스 균등 샘플 N장만 분석 (미지정 시 전체)")
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if args.preflight==args.execute:parser.error("--preflight 또는 --execute 중 하나만 지정하세요.")
    load_rembg()

    image_dir=args.image_dir.expanduser().resolve();csv_path=args.csv.expanduser().resolve()
    existing_excel=args.existing_excel.expanduser().resolve()
    u2net_home=args.u2net_home.expanduser().resolve()
    if not image_dir.is_dir():parser.error(f"이미지 폴더 없음: {image_dir}")
    if not csv_path.is_file():parser.error(f"CSV 없음: {csv_path}")
    if not existing_excel.is_file():
        parser.error(f"기존 Excel 없음(새 Excel 생성 차단): {existing_excel}")
    os.environ["U2NET_HOME"]=str(u2net_home)
    ann_dir=resolve_annotation_dir(args.annotation_dir);ann_idx=annotation_index(ann_dir)
    images,by_name=image_index(image_dir);records,load_errors=load_records(csv_path,ann_idx)
    if load_errors:
        print("메타데이터 보완 오류 예시:");[print(f"- {x}") for x in load_errors[:20]]
        raise SystemExit(f"사전검증 실패: CSV/Annotation 오류 {len(load_errors):,}건")

    existing_details,existing_selected=load_existing_excel(existing_excel)
    existing_names={norm(row.get("file_name","")) for row in existing_details
                    if norm(row.get("file_name",""))}
    duplicate_local={k:v for k,v in by_name.items() if len(v)>1};current=[];unmatched=[]
    for record in records:
        matches=by_name.get(record.file_name,[])
        if len(matches)==1:record.source_path=matches[0];current.append(record)
        elif len(matches)>1:unmatched.append(f"중복 PNG {record.file_name}: {len(matches)}개")
    csv_names={record.file_name for record in records}
    extra=[path for path in images if norm(path.name) not in csv_names]
    class_map=defaultdict(set)
    for record in current:class_map[record.item_seq].add((record.category_id,record.category_name))
    mapping_conflicts={key:value for key,value in class_map.items() if len(value)!=1}
    new_current=[record for record in current if record.file_name not in existing_names]

    print("========== 입력 사전검증 ==========")
    print(f"현재 로컬 PNG: {len(images):,}")
    print(f"CSV와 1:1 매칭 PNG: {len(current):,}")
    print(f"CSV 대상 외 PNG: {len(extra):,}")
    print(f"로컬 중복 file_name: {len(duplicate_local):,}")
    print(f"item_seq → Train 클래스 충돌: {len(mapping_conflicts):,}")
    print(f"기존 Excel image_quality 행: {len(existing_details):,}")
    print(f"기존 Excel 고유 file_name: {len(existing_names):,}")
    print(f"Excel 증분 갱신 대상(단일): {existing_excel}")
    print(f"review/CSV 출력 경로: {args.review_root.expanduser().resolve()}")
    print(f"분석 모델: {args.model} | U2NET_HOME: {u2net_home}")
    print(f"기존 분석 완료로 제외: {len(current)-len(new_current):,}장")
    print(f"새 분석 대상: {len(new_current):,}장")
    print(f"촬영조건 소스: {'CSV' if ann_dir is None else ann_dir}")
    if not current or unmatched or mapping_conflicts:
        raise SystemExit("입력 사전검증 실패; 분석을 중단했습니다.")
    try:analysis_records=balanced_limit(new_current,args.limit)
    except ValueError as exc:parser.error(str(exc))
    if args.limit is not None:
        sampled_classes=len({(record.category_id,record.category_name) for record in analysis_records})
        print(f"제한 분석: 새 분석 대상 {len(new_current):,}장 중 {len(analysis_records):,}장")
        print(f"제한 샘플 포함 클래스: {sampled_classes:,}개 (클래스 round-robin)")

    if analysis_records:
        session=new_session(args.model)
        for record in tqdm(analysis_records,total=len(analysis_records),desc="BBox 객체 품질 분석",unit="장"):
            analyze_one(record,session)
    counts=Counter(record.quality for record in analysis_records)
    new_detail_rows=quality_rows(analysis_records)
    combined_details,dedup_removed=deduplicate_detail_rows(existing_details+new_detail_rows)
    stats=class_stats_from_detail_rows(combined_details)

    existing_selected_counts=Counter(
        (int(row["category_id"]),norm(row["category_name"])) for row in existing_selected
    )
    selected=[];groups=defaultdict(list)
    for record in analysis_records:
        if record.quality=="good":groups[(record.category_id,record.category_name)].append(record)
    for key,items in sorted(groups.items()):
        remaining=max(0,100-existing_selected_counts[key])
        if remaining:selected.extend(select_candidates(items,remaining))
    selected_rows=[]
    for record,reason in selected:
        selected_rows.append({
            "file_name":record.file_name,"category_id":record.category_id,
            "category_name":record.category_name,"item_seq":record.item_seq,
            "back_color":record.back_color,"light_color":record.light_color,
            "camera_la":record.camera_la,"camera_lo":record.camera_lo,
            "drug_dir":record.drug_dir,"bbox_quality":record.quality,
            "selection_reason":reason,
        })
    combined_selected=[];selected_names=set();selected_class_counts=Counter()
    for row in existing_selected+selected_rows:
        file_name=norm(row.get("file_name",""))
        key=(int(row["category_id"]),norm(row["category_name"]))
        if not file_name or file_name in selected_names or selected_class_counts[key]>=100:continue
        combined_selected.append({header:row.get(header,"") for header in SELECTED_HEADERS})
        selected_names.add(file_name);selected_class_counts[key]+=1

    print("========== 분석/선별 예상 결과 ==========")
    print(f"신규 분석: {len(analysis_records):,}장 | good={counts['good']:,}, suspect={counts['suspect']:,}, bad={counts['bad']:,}")
    print(f"누적 image_quality: {len(combined_details):,}행 | 중복 annotation 제거: {dedup_removed:,}행")
    print(f"누적 클래스: {len(stats):,}개 | 신규 Train 후보: {len(selected_rows):,}장 | 누적 후보: {len(combined_selected):,}장")
    print("주의: 현재 로컬 PNG만 계산한 결과이며 전체 55,728장 최종 결과가 아닙니다.")
    if args.preflight:
        print("PREVIEW ONLY: 파일 복사·리뷰·Excel/CSV 저장을 수행하지 않았습니다.");return 0

    review_root=args.review_root.expanduser().resolve();candidate_root=args.candidate_root.expanduser().resolve()
    review_root.mkdir(parents=True,exist_ok=True);candidate_root.mkdir(parents=True,exist_ok=True)
    for record,reason in selected:
        folder=candidate_root/f"{record.category_id}_{safe_folder(record.category_name)}"
        folder.mkdir(parents=True,exist_ok=True);target=folder/record.file_name
        if target.exists():raise RuntimeError(f"후보 출력 충돌(덮어쓰기 차단): {target}")
        shutil.copy2(record.source_path,target)
    for record in tqdm([item for item in analysis_records if item.quality in {"suspect","bad"}],
                       desc="Review 생성",unit="장"):
        target=review_root/"review"/record.quality/f"{record.category_id}_{safe_folder(record.category_name)}"/record.file_name
        if target.exists():raise RuntimeError(f"Review 출력 충돌(덮어쓰기 차단): {target}")
        review_image(record,target)
    write_csv_atomic(review_root/"bbox_quality_class_summary.csv",stats,SUMMARY_HEADERS)
    write_csv_atomic(review_root/"bbox_quality_details.csv",combined_details,QUALITY_HEADERS)
    write_csv_atomic(candidate_root/"selected_train_candidates.csv",combined_selected,SELECTED_HEADERS)
    backup=write_excel_incremental(existing_excel,stats,combined_details,combined_selected)
    print(f"실행 완료: Excel={existing_excel}")
    if backup:print(f"갱신 전 Excel 백업: {backup}")
    print(f"review={review_root}, candidates={candidate_root}")
    print("원본 PNG/CSV/Annotation은 수정하거나 삭제하지 않았습니다.")
    return 0


if __name__=="__main__":sys.exit(main())
