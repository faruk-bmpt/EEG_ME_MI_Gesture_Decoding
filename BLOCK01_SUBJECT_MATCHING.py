# ============================================================
# PAPER 2 — BLOCK 01 FINAL
# SUBJECT MATCHING & COHORT INTEGRITY
# ============================================================
#
# FINAL CLEAN/FROZEN VERSION
#
# PURPOSE
# -------
# Establish the definitive 25-subject paired ME/MI cohort
# from the CURRENT raw-data folders.
#
# THIS SCRIPT:
#   1. Verifies BLOCK 00.
#   2. Verifies both raw-data folders.
#   3. INVENTORIES all raw files.
#   4. Counts only scientific CSV recordings:
#        CSV files outside AUDIT/
#   5. Treats TXT files as companion logs only.
#   6. Discovers subjects/gestures/sets.
#   7. Checks ME/MI matching.
#   8. Checks 7 gestures × A/B/C.
#   9. Checks 21 recordings per subject-condition.
#  10. Checks 1,050 expected logical recordings.
#  11. Detects logical duplicate files.
#  12. Detects missing/unexpected recordings.
#  13. Creates ONE clean official BLOCK 01 output set.
#
# IMPORTANT
# ---------
# RAW DATA ARE NEVER MODIFIED OR DELETED.
#
# The script DOES delete/recreate ONLY:
#     Paper2_EEG_BCI/01_SUBJECT_MATCHING/
#
# This removes stale/duplicate BLOCK 01 outputs from earlier
# versions before creating the final outputs.
#
# BLOCK 01 DOES NOT:
#   - read EEG signal values
#   - filter
#   - resample
#   - interpolate
#   - window
#   - extract features
#   - train models
#   - evaluate models
#   - perform statistics
#   - exclude subjects based on performance
#
# NEXT:
#     BLOCK 02 — SIGNAL QUALITY ASSURANCE
#
# ============================================================


# ============================================================
# 1. MOUNT GOOGLE DRIVE
# ============================================================

from google.colab import drive
drive.mount("/content/drive")


# ============================================================
# 2. IMPORTS
# ============================================================

from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import csv
import json
import re
import shutil


# ============================================================
# 3. EXACT PROJECT PATHS
# ============================================================

BASE_DIR = Path(
    "/content/drive/MyDrive/EEG_Motor_Execution_Imagery"
)

ME_RAW = BASE_DIR / "Motor_Execution_Raw_Data"
MI_RAW = BASE_DIR / "Motor_Imagery_Raw_Data"

PROJECT_ROOT = BASE_DIR / "Paper2_EEG_BCI"

BLOCK00_DIR = PROJECT_ROOT / "00_PROTOCOL"
BLOCK01_DIR = PROJECT_ROOT / "01_SUBJECT_MATCHING"

BLOCK00_COMPLETE = BLOCK00_DIR / "BLOCK00_complete.json"
BLOCK00_CONFIG = BLOCK00_DIR / "configs" / "BLOCK00_config.json"


# ============================================================
# 4. FROZEN PAPER 2 DEFINITIONS
# ============================================================

BLOCK_ID = "BLOCK01"
BLOCK_VERSION = "PAPER2-BLOCK01-FINAL-CLEAN-2.0"

CONDITIONS = ("ME", "MI")
SETS = ("A", "B", "C")

GESTURES = {
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
}

GESTURE_TO_CODE = {
    name.upper(): code
    for code, name in GESTURES.items()
}

GESTURE_FOLDER_TO_CODE = {
    "HAND_OPEN_CLOSE": 1,
    "INDEX_FINGER_MOVEMENT": 2,
    "LITTLE_FINGER_MOVEMENT": 3,
    "MIDDLE_FINGER_MOVEMENT": 4,
    "PEN_HOLDING": 5,
    "RING_FINGER_MOVEMENT": 6,
    "THUMB_FINGER_MOVEMENT": 7,
}

PROTOCOL_PRIMARY_SUBJECT_COUNT = 25
EXPECTED_RECORDINGS_PER_CONDITION = 7 * 3
EXPECTED_RECORDINGS_PER_PAIRED_SUBJECT = 2 * 7 * 3
EXPECTED_TOTAL_RECORDINGS = 25 * 2 * 7 * 3


# ============================================================
# 5. BASIC HELPERS
# ============================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def normalize_subject_id(raw_subject):
    if raw_subject is None:
        return None

    text = str(raw_subject).strip()

    if not text:
        return None

    text = re.sub(r"[\s\-]+", "_", text)

    patterns = [
        r"(?i)^S_?(\d+)$",
        r"(?i)^Subject_?(\d+)$",
    ]

    for pattern in patterns:
        m = re.fullmatch(pattern, text)
        if m:
            return f"S{int(m.group(1))}"

    return text.upper()


def extract_subject_from_text(text):
    """
    Conservative subject extraction.

    Examples:
        S01
        S_01
        Subject_01
        Subject01
    """
    if text is None:
        return None

    stem = Path(str(text)).stem

    patterns = [
        r"(?i)(?<![A-Za-z0-9])(Subject[_\-]?\d+)(?![A-Za-z0-9])",
        r"(?i)(?<![A-Za-z0-9])(S[_\-]?\d+)(?![A-Za-z0-9])",
        r"(?i)^(Subject[_\-]?\d+)",
        r"(?i)^(S[_\-]?\d+)",
    ]

    for pattern in patterns:
        m = re.search(pattern, stem)
        if m:
            token = m.group(1)
            return normalize_subject_id(token)

    return None


def extract_subject(path, raw_root):
    """
    Prefer the actual directory structure.

    Example:
        Subject_01/Hand_Open_Close/S01_HOC_SetA.csv

    Filename is used only as fallback.
    """
    try:
        rel_parts = path.relative_to(raw_root).parts
    except ValueError:
        rel_parts = path.parts

    # Prefer Subject_* directory.
    for part in rel_parts[:-1]:
        subject = normalize_subject_id(part)
        if subject and re.fullmatch(r"S\d+", subject):
            if re.fullmatch(r"(?i)(Subject[_\-]?\d+|S[_\-]?\d+)", part):
                return subject, part, "subject_directory"

    # Filename fallback.
    subject = extract_subject_from_text(path.name)
    if subject:
        return subject, subject, "filename"

    return None, None, None


def extract_gesture(path, raw_root):
    """
    Prefer the actual gesture directory.
    Filename short code is fallback.
    """
    try:
        rel_parts = path.relative_to(raw_root).parts
    except ValueError:
        rel_parts = path.parts

    # Directory-based gesture parsing.
    for part in rel_parts[:-1]:
        normalized = re.sub(r"[^A-Z0-9]+", "_", part.upper()).strip("_")

        if normalized in GESTURE_FOLDER_TO_CODE:
            code = GESTURE_FOLDER_TO_CODE[normalized]
            return code, GESTURES[code], "gesture_directory"

    # Filename fallback.
    stem = Path(path.name).stem.upper()
    stem = re.sub(r"\(\d+\)$", "", stem)

    # Long names first.
    for name, code in GESTURE_TO_CODE.items():
        pattern = (
            r"(?<![A-Z0-9])"
            + re.escape(name)
            + r"(?![A-Z0-9])"
        )
        if re.search(pattern, stem):
            return code, name, "filename"

    return None, None, None


def extract_set(filename):
    stem = Path(filename).stem

    patterns = [
        r"(?i)(?:^|[_\-\s])Set[_\-]?([ABC])(?:$|[_\-\s\(])",
        r"(?i)(?:^|[_\-\s])([ABC])(?:$|[_\-\s\(])",
    ]

    for pattern in patterns:
        matches = list(re.finditer(pattern, stem))
        if matches:
            candidate = matches[-1].group(1).upper()
            if candidate in SETS:
                return candidate

    return None


def detect_duplicate_suffix(filename):
    stem = Path(filename).stem

    for pattern in (r"\((\d+)\)$", r"_(\d+)$"):
        m = re.search(pattern, stem)
        if m:
            return m.group(1)

    return None


def make_logical_key(subject, condition, gesture_code, set_code):
    if None in (subject, condition, gesture_code, set_code):
        return None

    return f"{subject}|{condition}|{gesture_code}|{set_code}"


def companion_log_candidates(csv_path):
    """
    Recognize both:
        file.txt
        file.log.txt

    without requiring either one to be present.
    """
    return [
        csv_path.with_suffix(".txt"),
        csv_path.with_name(csv_path.stem + ".log.txt"),
    ]


# ============================================================
# 6. VERIFY BLOCK 00
# ============================================================

print("=" * 78)
print("PAPER 2 — BLOCK 01 FINAL CLEAN RUN")
print("=" * 78)
print()

print("VERIFYING BLOCK 00...")

if not BLOCK00_COMPLETE.exists():
    raise FileNotFoundError(
        f"BLOCK 00 completion marker not found:\n{BLOCK00_COMPLETE}"
    )

if not BLOCK00_CONFIG.exists():
    raise FileNotFoundError(
        f"BLOCK 00 configuration not found:\n{BLOCK00_CONFIG}"
    )

block00_complete = json.loads(
    BLOCK00_COMPLETE.read_text(encoding="utf-8")
)

block00_config = json.loads(
    BLOCK00_CONFIG.read_text(encoding="utf-8")
)

if block00_complete.get("status") != "COMPLETE_AND_FROZEN":
    raise RuntimeError(
        "BLOCK 00 is not COMPLETE_AND_FROZEN."
    )

if block00_complete.get("complete_marker") is not True:
    raise RuntimeError(
        "BLOCK 00 completion marker is invalid."
    )

if block00_complete.get("next_block") != "BLOCK01":
    raise RuntimeError(
        "BLOCK 00 does not authorize BLOCK 01."
    )

print("✓ BLOCK 00 = COMPLETE_AND_FROZEN")
print()


# ============================================================
# 7. VERIFY RAW DIRECTORIES
# ============================================================

print("VERIFYING RAW DIRECTORIES...")

for path, label in [
    (ME_RAW, "Motor Execution"),
    (MI_RAW, "Motor Imagery"),
]:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} raw directory not found:\n{path}"
        )
    if not path.is_dir():
        raise NotADirectoryError(
            f"{label} raw path is not a directory:\n{path}"
        )

print("✓ ME raw directory found")
print("✓ MI raw directory found")
print()


# ============================================================
# 8. CLEAN OLD BLOCK 01 OUTPUT
# ============================================================
#
# IMPORTANT:
# Only the official BLOCK 01 output directory is removed.
#
# NEVER remove:
#   Motor_Execution_Raw_Data/
#   Motor_Imagery_Raw_Data/
#   00_PROTOCOL/
#   any other Paper2 block
#
# ============================================================

print("=" * 78)
print("CLEANING OLD BLOCK 01 OUTPUT")
print("=" * 78)

if BLOCK01_DIR.exists():
    print(f"Removing OLD BLOCK 01 directory:")
    print(BLOCK01_DIR)

    shutil.rmtree(BLOCK01_DIR)

    print("✓ OLD BLOCK 01 output removed")
else:
    print("✓ No previous BLOCK 01 directory found")

BLOCK01_DIR.mkdir(parents=True, exist_ok=True)

print("✓ Clean BLOCK 01 directory created")
print()


# ============================================================
# 9. DISCOVER ALL FILES
# ============================================================

def discover_files(raw_root, condition):

    rows = []

    all_paths = sorted(
        p for p in raw_root.rglob("*")
        if p.is_file()
    )

    for path in all_paths:

        relative_path = path.relative_to(raw_root)
        parts_upper = {
            part.upper()
            for part in relative_path.parts[:-1]
        }

        in_audit = "AUDIT" in parts_upper
        extension = path.suffix.lower()

        subject_id, raw_subject, subject_source = extract_subject(
            path,
            raw_root,
        )

        gesture_code, gesture_name, gesture_source = extract_gesture(
            path,
            raw_root,
        )

        set_code = extract_set(path.name)
        duplicate_suffix = detect_duplicate_suffix(path.name)

        if in_audit:
            category = "audit_artifact"
            scientific_recording = False

        elif extension == ".csv":
            category = "scientific_csv_candidate"
            scientific_recording = True

        elif extension == ".txt":
            category = "companion_txt"
            scientific_recording = False

        else:
            category = "supporting_or_non_signal"
            scientific_recording = False

        parse_status = "NOT_A_RECORDING"

        if scientific_recording:
            problems = []

            if subject_id is None:
                problems.append("SUBJECT_UNPARSED")

            if gesture_code is None:
                problems.append("GESTURE_UNPARSED")

            if set_code is None:
                problems.append("SET_UNPARSED")

            parse_status = (
                "PARSED"
                if not problems
                else ";".join(problems)
            )

        logical_key = (
            make_logical_key(
                subject_id,
                condition,
                gesture_code,
                set_code,
            )
            if scientific_recording
            and parse_status == "PARSED"
            else None
        )

        rows.append({
            "condition": condition,
            "relative_path": str(relative_path),
            "filename": path.name,
            "extension": extension,
            "file_size_bytes": path.stat().st_size,
            "file_category": category,
            "scientific_recording": scientific_recording,
            "in_audit_directory": in_audit,
            "subject_id": subject_id,
            "raw_subject_token": raw_subject,
            "subject_parse_source": subject_source,
            "gesture_code": gesture_code,
            "gesture_name": gesture_name,
            "gesture_parse_source": gesture_source,
            "set_code": set_code,
            "duplicate_suffix": duplicate_suffix,
            "logical_key": logical_key,
            "parse_status": parse_status,
            "absolute_path": str(path),
        })

    return rows


print("=" * 78)
print("INVENTORYING CURRENT RAW DATA")
print("=" * 78)

me_all = discover_files(ME_RAW, "ME")
mi_all = discover_files(MI_RAW, "MI")

all_files = me_all + mi_all

print(f"ME total filesystem files : {len(me_all):,}")
print(f"MI total filesystem files : {len(mi_all):,}")
print(f"TOTAL filesystem files    : {len(all_files):,}")
print()


# ============================================================
# 10. FILE CATEGORY COUNTS
# ============================================================

def category_counts(rows):
    return Counter(row["file_category"] for row in rows)


me_counts = category_counts(me_all)
mi_counts = category_counts(mi_all)

print("FILE CATEGORIES")
print("-" * 78)

for category in sorted(
    set(me_counts) | set(mi_counts)
):
    print(
        f"{category:32s} "
        f"ME={me_counts.get(category, 0):4d} "
        f"MI={mi_counts.get(category, 0):4d}"
    )

print()


# ============================================================
# 11. SCIENTIFIC RECORDINGS
# ============================================================

parsed_rows = [
    row
    for row in all_files
    if (
        row["scientific_recording"]
        and row["parse_status"] == "PARSED"
    )
]

unparsed_rows = [
    row
    for row in all_files
    if (
        row["scientific_recording"]
        and row["parse_status"] != "PARSED"
    )
]

me_recordings = [
    row for row in parsed_rows
    if row["condition"] == "ME"
]

mi_recordings = [
    row for row in parsed_rows
    if row["condition"] == "MI"
]

print("=" * 78)
print("SCIENTIFIC RECORDING DISCOVERY")
print("=" * 78)

print(f"ME scientific CSV recordings : {len(me_recordings):,}")
print(f"MI scientific CSV recordings : {len(mi_recordings):,}")
print(f"TOTAL scientific recordings  : {len(parsed_rows):,}")
print(f"Unparsed scientific CSVs      : {len(unparsed_rows):,}")
print()


# ============================================================
# 12. SUBJECT MATCHING
# ============================================================

me_subjects = sorted({
    row["subject_id"]
    for row in me_recordings
    if row["subject_id"] is not None
})

mi_subjects = sorted({
    row["subject_id"]
    for row in mi_recordings
    if row["subject_id"] is not None
})

me_set = set(me_subjects)
mi_set = set(mi_subjects)

paired_subjects = sorted(me_set & mi_set)
me_only_subjects = sorted(me_set - mi_set)
mi_only_subjects = sorted(mi_set - me_set)

unexpected_subjects = sorted(
    {
        s
        for s in (me_set | mi_set)
        if not re.fullmatch(r"S\d+", str(s))
        or not (1 <= int(str(s)[1:]) <= 25)
    }
)

print("=" * 78)
print("SUBJECT MATCHING")
print("=" * 78)

print(f"ME subjects       : {len(me_subjects):,}")
print(f"MI subjects       : {len(mi_subjects):,}")
print(f"Paired subjects   : {len(paired_subjects):,}")
print(f"ME-only subjects  : {len(me_only_subjects):,}")
print(f"MI-only subjects  : {len(mi_only_subjects):,}")
print(f"Unexpected IDs    : {len(unexpected_subjects):,}")
print()

print("Paired subjects:")
print(paired_subjects)
print()


# ============================================================
# 13. EXPECTED 25-SUBJECT LOGICAL KEYS
# ============================================================

expected_subjects = [
    f"S{i}"
    for i in range(1, PROTOCOL_PRIMARY_SUBJECT_COUNT + 1)
]

expected_rows = []

for subject in expected_subjects:
    for condition in CONDITIONS:
        for gesture_code in sorted(GESTURES):
            for set_code in SETS:

                expected_rows.append({
                    "subject_id": subject,
                    "condition": condition,
                    "gesture_code": gesture_code,
                    "gesture_name": GESTURES[gesture_code],
                    "set_code": set_code,
                    "logical_key": make_logical_key(
                        subject,
                        condition,
                        gesture_code,
                        set_code,
                    ),
                })

expected_keys = {
    row["logical_key"]
    for row in expected_rows
}


# ============================================================
# 14. ACTUAL LOGICAL KEY GROUPS
# ============================================================

logical_groups = defaultdict(list)

for row in parsed_rows:
    if row["logical_key"] is not None:
        logical_groups[row["logical_key"]].append(row)

actual_keys = set(logical_groups)


# ============================================================
# 15. MISSING / UNEXPECTED / DUPLICATE LOGICAL RECORDINGS
# ============================================================

missing_keys = sorted(expected_keys - actual_keys)
unexpected_keys = sorted(actual_keys - expected_keys)

duplicate_groups = {
    key: rows
    for key, rows in logical_groups.items()
    if len(rows) > 1
}

duplicate_rows = []

for key, rows in sorted(duplicate_groups.items()):
    for row in rows:
        duplicate_rows.append({
            "logical_key": key,
            "condition": row["condition"],
            "subject_id": row["subject_id"],
            "gesture_code": row["gesture_code"],
            "gesture_name": row["gesture_name"],
            "set_code": row["set_code"],
            "relative_path": row["relative_path"],
            "filename": row["filename"],
            "file_size_bytes": row["file_size_bytes"],
            "duplicate_suffix": row["duplicate_suffix"],
        })

missing_rows = []

for expected in expected_rows:
    if expected["logical_key"] in missing_keys:
        missing_rows.append({
            **expected,
            "status": "MISSING",
        })


unexpected_key_rows = []

for key in unexpected_keys:
    rows = logical_groups[key]
    for row in rows:
        unexpected_key_rows.append({
            "logical_key": key,
            "condition": row["condition"],
            "subject_id": row["subject_id"],
            "gesture_code": row["gesture_code"],
            "gesture_name": row["gesture_name"],
            "set_code": row["set_code"],
            "relative_path": row["relative_path"],
            "filename": row["filename"],
            "reason": "OUTSIDE_FROZEN_25_SUBJECT_COHORT",
        })


# ============================================================
# 16. SUBJECT COMPLETENESS
# ============================================================

subject_rows = []

for subject in expected_subjects:

    me_keys = {
        row["logical_key"]
        for row in me_recordings
        if row["subject_id"] == subject
    }

    mi_keys = {
        row["logical_key"]
        for row in mi_recordings
        if row["subject_id"] == subject
    }

    expected_me = {
        make_logical_key(subject, "ME", g, s)
        for g in GESTURES
        for s in SETS
    }

    expected_mi = {
        make_logical_key(subject, "MI", g, s)
        for g in GESTURES
        for s in SETS
    }

    me_missing = sorted(expected_me - me_keys)
    mi_missing = sorted(expected_mi - mi_keys)

    me_complete = (
        len(me_keys) == EXPECTED_RECORDINGS_PER_CONDITION
        and not me_missing
    )

    mi_complete = (
        len(mi_keys) == EXPECTED_RECORDINGS_PER_CONDITION
        and not mi_missing
    )

    subject_rows.append({
        "subject_id": subject,
        "me_recording_count": len(me_keys),
        "mi_recording_count": len(mi_keys),
        "expected_per_condition": EXPECTED_RECORDINGS_PER_CONDITION,
        "expected_paired": EXPECTED_RECORDINGS_PER_PAIRED_SUBJECT,
        "me_missing_count": len(me_missing),
        "mi_missing_count": len(mi_missing),
        "me_missing_keys": ";".join(me_missing),
        "mi_missing_keys": ";".join(mi_missing),
        "me_complete": me_complete,
        "mi_complete": mi_complete,
        "paired_complete": me_complete and mi_complete,
    })

complete_subjects = [
    row["subject_id"]
    for row in subject_rows
    if row["paired_complete"]
]


# ============================================================
# 17. GESTURE COMPLETENESS
# ============================================================

gesture_rows = []

for subject in expected_subjects:
    for condition in CONDITIONS:
        for gesture_code, gesture_name in GESTURES.items():

            observed_sets = {
                row["set_code"]
                for row in parsed_rows
                if (
                    row["subject_id"] == subject
                    and row["condition"] == condition
                    and row["gesture_code"] == gesture_code
                )
            }

            missing_sets = sorted(set(SETS) - observed_sets)

            gesture_rows.append({
                "subject_id": subject,
                "condition": condition,
                "gesture_code": gesture_code,
                "gesture_name": gesture_name,
                "observed_set_count": len(observed_sets),
                "observed_sets": ";".join(sorted(observed_sets)),
                "missing_set_count": len(missing_sets),
                "missing_sets": ";".join(missing_sets),
                "complete": len(missing_sets) == 0,
            })


# ============================================================
# 18. SET COMPLETENESS
# ============================================================

set_rows = []

for subject in expected_subjects:
    for condition in CONDITIONS:
        for set_code in SETS:

            observed_gestures = {
                row["gesture_code"]
                for row in parsed_rows
                if (
                    row["subject_id"] == subject
                    and row["condition"] == condition
                    and row["set_code"] == set_code
                )
            }

            missing_gestures = sorted(
                set(GESTURES) - observed_gestures
            )

            set_rows.append({
                "subject_id": subject,
                "condition": condition,
                "set_code": set_code,
                "observed_gesture_count": len(observed_gestures),
                "observed_gestures": ";".join(
                    str(x) for x in sorted(observed_gestures)
                ),
                "missing_gesture_count": len(missing_gestures),
                "missing_gestures": ";".join(
                    str(x) for x in missing_gestures
                ),
                "complete": len(missing_gestures) == 0,
            })


# ============================================================
# 19. COMPANION TXT AUDIT
# ============================================================
#
# TXT files are NOT scientific recordings.
# Their presence/absence does not affect the cohort.
#
# Recognizes:
#     S01_HOC_SetA.txt
#     S01_HOC_SetA.log.txt
#
# ============================================================

companion_rows = []

for row in parsed_rows:

    csv_path = Path(row["absolute_path"])

    candidates = companion_log_candidates(csv_path)

    existing = [
        p for p in candidates
        if p.exists() and p.is_file()
    ]

    companion_rows.append({
        "condition": row["condition"],
        "subject_id": row["subject_id"],
        "gesture_code": row["gesture_code"],
        "gesture_name": row["gesture_name"],
        "set_code": row["set_code"],
        "csv_filename": row["filename"],
        "companion_present": bool(existing),
        "companion_filenames": ";".join(
            p.name for p in existing
        ),
    })


# ============================================================
# 20. RECORDING INVENTORY
# ============================================================

recording_inventory = []

for row in sorted(
    parsed_rows,
    key=lambda r: (
        r["condition"],
        r["subject_id"],
        r["gesture_code"],
        r["set_code"],
    ),
):

    recording_inventory.append({
        "condition": row["condition"],
        "subject_id": row["subject_id"],
        "gesture_code": row["gesture_code"],
        "gesture_name": row["gesture_name"],
        "set_code": row["set_code"],
        "logical_key": row["logical_key"],
        "relative_path": row["relative_path"],
        "filename": row["filename"],
        "file_size_bytes": row["file_size_bytes"],
        "duplicate_suffix": row["duplicate_suffix"],
        "absolute_path": row["absolute_path"],
    })


# ============================================================
# 21. ALL-FILE INVENTORY
# ============================================================

all_file_inventory = []

for row in sorted(
    all_files,
    key=lambda r: (
        r["condition"],
        r["relative_path"],
    ),
):
    all_file_inventory.append(row)


# ============================================================
# 22. SUBJECT MATCHING MANIFEST
# ============================================================

subject_matching_manifest = []

for subject in expected_subjects:

    item = next(
        row for row in subject_rows
        if row["subject_id"] == subject
    )

    subject_matching_manifest.append({
        "subject_id": subject,
        "in_me": item["me_recording_count"] > 0,
        "in_mi": item["mi_recording_count"] > 0,
        "paired": (
            item["me_recording_count"] > 0
            and item["mi_recording_count"] > 0
        ),
        "me_complete": item["me_complete"],
        "mi_complete": item["mi_complete"],
        "paired_complete": item["paired_complete"],
    })


# ============================================================
# 23. QA REVIEW MANIFEST
# ============================================================

review_rows = []

for row in subject_rows:
    if not row["paired_complete"]:
        review_rows.append({
            "issue_type": "STRUCTURAL_INCOMPLETENESS",
            "subject_id": row["subject_id"],
            "logical_key": "",
            "relative_path": "",
            "decision": "REVIEW",
            "reason": "ME/MI recording structure is incomplete.",
        })

for row in duplicate_rows:
    review_rows.append({
        "issue_type": "DUPLICATE_LOGICAL_RECORDING",
        "subject_id": row["subject_id"],
        "logical_key": row["logical_key"],
        "relative_path": row["relative_path"],
        "decision": "REVIEW",
        "reason": (
            "More than one scientific CSV maps to the same "
            "Subject+Condition+Gesture+Set logical key."
        ),
    })

for row in unexpected_rows:
    review_rows.append({
        "issue_type": "UNPARSED_SCIENTIFIC_CSV",
        "subject_id": row["subject_id"] or "",
        "logical_key": "",
        "relative_path": row["relative_path"],
        "decision": "REVIEW",
        "reason": row["parse_status"],
    })

for row in unexpected_key_rows:
    review_rows.append({
        "issue_type": "UNEXPECTED_LOGICAL_KEY",
        "subject_id": row["subject_id"],
        "logical_key": row["logical_key"],
        "relative_path": row["relative_path"],
        "decision": "REVIEW",
        "reason": row["reason"],
    })


# ============================================================
# 24. FINAL PASS LOGIC
# ============================================================

pass_conditions = {
    "me_scientific_csv_count": len(me_recordings) == 525,
    "mi_scientific_csv_count": len(mi_recordings) == 525,
    "me_subject_count": len(me_subjects) == 25,
    "mi_subject_count": len(mi_subjects) == 25,
    "paired_subject_count": len(paired_subjects) == 25,
    "complete_paired_subject_count": len(complete_subjects) == 25,
    "me_only_subject_count": len(me_only_subjects) == 0,
    "mi_only_subject_count": len(mi_only_subjects) == 0,
    "unexpected_subject_count": len(unexpected_subjects) == 0,
    "parsed_recording_count": len(parsed_rows) == 1050,
    "unparsed_scientific_csv_count": len(unparsed_rows) == 0,
    "logical_recording_count": len(actual_keys) == 1050,
    "missing_recording_count": len(missing_keys) == 0,
    "unexpected_logical_key_count": len(unexpected_keys) == 0,
    "duplicate_logical_group_count": len(duplicate_groups) == 0,
}

block01_status = (
    "PASS"
    if all(pass_conditions.values())
    else "REVIEW"
)


# ============================================================
# 25. WRITE CLEAN OFFICIAL OUTPUTS
# ============================================================

print("=" * 78)
print("WRITING CLEAN BLOCK 01 OUTPUTS")
print("=" * 78)

write_csv(
    BLOCK01_DIR / "all_file_inventory.csv",
    all_file_inventory,
    [
        "condition",
        "relative_path",
        "filename",
        "extension",
        "file_size_bytes",
        "file_category",
        "scientific_recording",
        "in_audit_directory",
        "subject_id",
        "raw_subject_token",
        "subject_parse_source",
        "gesture_code",
        "gesture_name",
        "gesture_parse_source",
        "set_code",
        "duplicate_suffix",
        "logical_key",
        "parse_status",
        "absolute_path",
    ],
)

write_csv(
    BLOCK01_DIR / "recording_inventory.csv",
    recording_inventory,
    [
        "condition",
        "subject_id",
        "gesture_code",
        "gesture_name",
        "set_code",
        "logical_key",
        "relative_path",
        "filename",
        "file_size_bytes",
        "duplicate_suffix",
        "absolute_path",
    ],
)

write_csv(
    BLOCK01_DIR / "subject_matching_manifest.csv",
    subject_matching_manifest,
    [
        "subject_id",
        "in_me",
        "in_mi",
        "paired",
        "me_complete",
        "mi_complete",
        "paired_complete",
    ],
)

write_csv(
    BLOCK01_DIR / "subject_completeness.csv",
    subject_rows,
    [
        "subject_id",
        "me_recording_count",
        "mi_recording_count",
        "expected_per_condition",
        "expected_paired",
        "me_missing_count",
        "mi_missing_count",
        "me_missing_keys",
        "mi_missing_keys",
        "me_complete",
        "mi_complete",
        "paired_complete",
    ],
)

write_csv(
    BLOCK01_DIR / "gesture_completeness.csv",
    gesture_rows,
    [
        "subject_id",
        "condition",
        "gesture_code",
        "gesture_name",
        "observed_set_count",
        "observed_sets",
        "missing_set_count",
        "missing_sets",
        "complete",
    ],
)

write_csv(
    BLOCK01_DIR / "set_completeness.csv",
    set_rows,
    [
        "subject_id",
        "condition",
        "set_code",
        "observed_gesture_count",
        "observed_gestures",
        "missing_gesture_count",
        "missing_gestures",
        "complete",
    ],
)

write_csv(
    BLOCK01_DIR / "missing_recordings.csv",
    missing_rows,
    [
        "subject_id",
        "condition",
        "gesture_code",
        "gesture_name",
        "set_code",
        "logical_key",
        "status",
    ],
)

write_csv(
    BLOCK01_DIR / "duplicate_recordings.csv",
    duplicate_rows,
    [
        "logical_key",
        "condition",
        "subject_id",
        "gesture_code",
        "gesture_name",
        "set_code",
        "relative_path",
        "filename",
        "file_size_bytes",
        "duplicate_suffix",
    ],
)

write_csv(
    BLOCK01_DIR / "unexpected_recordings.csv",
    unexpected_rows,
    [
        "condition",
        "relative_path",
        "filename",
        "subject_id",
        "gesture_code",
        "gesture_name",
        "set_code",
        "parse_status",
    ],
)

write_csv(
    BLOCK01_DIR / "unexpected_logical_keys.csv",
    unexpected_key_rows,
    [
        "logical_key",
        "condition",
        "subject_id",
        "gesture_code",
        "gesture_name",
        "set_code",
        "relative_path",
        "filename",
        "reason",
    ],
)

write_csv(
    BLOCK01_DIR / "companion_log_audit.csv",
    companion_rows,
    [
        "condition",
        "subject_id",
        "gesture_code",
        "gesture_name",
        "set_code",
        "csv_filename",
        "companion_present",
        "companion_filenames",
    ],
)

write_csv(
    BLOCK01_DIR / "BLOCK01_review_manifest.csv",
    review_rows,
    [
        "issue_type",
        "subject_id",
        "logical_key",
        "relative_path",
        "decision",
        "reason",
    ],
)


# ============================================================
# 26. SUMMARY JSON
# ============================================================

summary = {
    "block": BLOCK_ID,
    "block_version": BLOCK_VERSION,
    "status": block01_status,
    "created_utc": utc_now(),

    "raw_data_modified": False,
    "raw_data_deleted": False,

    "old_block01_output_deleted_before_run": True,

    "signal_processing_performed": False,
    "windowing_performed": False,
    "feature_extraction_performed": False,
    "model_training_performed": False,
    "statistical_testing_performed": False,
    "performance_based_subject_exclusion": False,

    "protocol": {
        "primary_subject_count": 25,
        "conditions": list(CONDITIONS),
        "gestures": GESTURES,
        "sets": list(SETS),
        "recordings_per_subject_condition": 21,
        "recordings_per_paired_subject": 42,
        "total_recordings": 1050,
    },

    "filesystem": {
        "me_total_files": len(me_all),
        "mi_total_files": len(mi_all),
        "total_files": len(all_files),
        "me_scientific_csv": len(me_recordings),
        "mi_scientific_csv": len(mi_recordings),
        "scientific_csv_total": len(parsed_rows),
        "unparsed_scientific_csv": len(unparsed_rows),
    },

    "subjects": {
        "me_subject_count": len(me_subjects),
        "mi_subject_count": len(mi_subjects),
        "paired_subject_count": len(paired_subjects),
        "complete_paired_subject_count": len(complete_subjects),
        "me_only_subjects": me_only_subjects,
        "mi_only_subjects": mi_only_subjects,
        "unexpected_subjects": unexpected_subjects,
        "paired_subjects": paired_subjects,
    },

    "logical_integrity": {
        "expected_logical_recordings": len(expected_keys),
        "actual_logical_recordings": len(actual_keys),
        "missing_recordings": len(missing_keys),
        "unexpected_logical_keys": len(unexpected_keys),
        "duplicate_logical_groups": len(duplicate_groups),
        "duplicate_recording_rows": len(duplicate_rows),
    },

    "pass_conditions": pass_conditions,

    "cohort_lock": (
        "STRUCTURALLY_COMPLETE"
        if block01_status == "PASS"
        else "PROVISIONAL_PENDING_REVIEW"
    ),

    "next_block": "BLOCK02",
}


write_json(
    BLOCK01_DIR / "BLOCK01_summary.json",
    summary,
)


# ============================================================
# 27. CONFIGURATION
# ============================================================

config = {
    "block": BLOCK_ID,
    "block_version": BLOCK_VERSION,

    "protocol_source": str(BLOCK00_CONFIG),

    "raw_paths": {
        "base": str(BASE_DIR),
        "me": str(ME_RAW),
        "mi": str(MI_RAW),
    },

    "output_path": str(BLOCK01_DIR),

    "frozen_cohort": {
        "subjects": [f"S{i}" for i in range(1, 26)],
        "conditions": list(CONDITIONS),
        "gestures": GESTURES,
        "sets": list(SETS),
    },

    "recording_definition": (
        "Scientific recording = .csv file outside any AUDIT directory."
    ),

    "companion_definition": (
        "TXT files are companion logs and are not counted as recordings."
    ),

    "subject_parsing": (
        "Prefer Subject_* directory; filename fallback."
    ),

    "gesture_parsing": (
        "Prefer gesture directory; filename short-code fallback."
    ),

    "set_parsing": (
        "SetA/SetB/SetC filename token."
    ),

    "duplicate_definition": (
        "More than one parsed scientific CSV maps to the same "
        "Subject+Condition+Gesture+Set logical key."
    ),

    "raw_data_modification": False,
    "raw_data_deletion": False,

    "old_block01_output_cleanup": (
        "Only Paper2_EEG_BCI/01_SUBJECT_MATCHING is deleted/recreated."
    ),

    "signal_processing": False,
    "windowing": False,
    "feature_extraction": False,
    "model_training": False,
    "statistics": False,
}


write_json(
    BLOCK01_DIR / "BLOCK01_config.json",
    config,
)


# ============================================================
# 28. COMPLETION MARKER
# ============================================================

completion = {
    "block": BLOCK_ID,
    "block_version": BLOCK_VERSION,
    "status": block01_status,
    "complete_marker": (
        block01_status == "PASS"
    ),
    "created_utc": utc_now(),

    "raw_data_scanned": True,
    "raw_data_modified": False,
    "raw_data_deleted": False,

    "old_block01_output_deleted_before_run": True,

    "scientific_analysis_performed": False,
    "signal_processing_performed": False,
    "model_training_performed": False,
    "statistical_testing_performed": False,
    "performance_based_subject_exclusion": False,

    "paired_subject_count": len(paired_subjects),
    "complete_paired_subject_count": len(complete_subjects),

    "scientific_recording_count": len(parsed_rows),
    "missing_recording_count": len(missing_keys),
    "duplicate_logical_group_count": len(duplicate_groups),
    "unexpected_logical_key_count": len(unexpected_keys),
    "unparsed_scientific_csv_count": len(unparsed_rows),

    "output_directory": str(BLOCK01_DIR),

    "next_block": (
        "BLOCK02"
        if block01_status == "PASS"
        else "BLOCK01_REVIEW"
    ),

    "completion_meaning": (
        "PASS means the current raw filesystem contains exactly "
        "25 structurally complete paired subjects and 1,050 expected "
        "logical recordings with no missing, unexpected, or duplicate "
        "logical recordings. No EEG values were processed."
    ),
}


write_json(
    BLOCK01_DIR / "BLOCK01_complete.json",
    completion,
)


# ============================================================
# 29. HUMAN-READABLE SUMMARY
# ============================================================

summary_text = f"""
PAPER 2 — BLOCK 01 FINAL CLEAN
SUBJECT MATCHING & COHORT INTEGRITY
===================================

STATUS
------
{block01_status}

RAW DATA
--------
ME:
{ME_RAW}

MI:
{MI_RAW}

RAW DATA MODIFIED:
NO

RAW DATA DELETED:
NO

OLD BLOCK 01 OUTPUT CLEANED:
YES

CURRENT BLOCK 01 OUTPUT:
{BLOCK01_DIR}


FILES
-----
ME scientific CSV recordings:
{len(me_recordings)}

MI scientific CSV recordings:
{len(mi_recordings)}

Total scientific recordings:
{len(parsed_rows)}

Unparsed scientific CSVs:
{len(unparsed_rows)}


SUBJECTS
--------
ME subjects:
{len(me_subjects)}

MI subjects:
{len(mi_subjects)}

Paired subjects:
{len(paired_subjects)}

Complete paired subjects:
{len(complete_subjects)}

ME-only:
{me_only_subjects}

MI-only:
{mi_only_subjects}

Unexpected subject IDs:
{unexpected_subjects}


EXPECTED STRUCTURE
------------------
Subjects:
25

Conditions:
ME, MI

Gestures:
1
2
3
4
5
6
7

Sets:
A, B, C

Recordings per subject-condition:
21

Recordings per paired subject:
42

Expected total recordings:
1050


LOGICAL INTEGRITY
-----------------
Expected logical recordings:
{len(expected_keys)}

Actual logical recordings:
{len(actual_keys)}

Missing logical recordings:
{len(missing_keys)}

Unexpected logical keys:
{len(unexpected_keys)}

Duplicate logical groups:
{len(duplicate_groups)}

Duplicate recording rows:
{len(duplicate_rows)}


SCOPE
-----
No EEG values were processed.

No timestamp analysis was performed.

No filtering was performed.

No resampling was performed.

No interpolation was performed.

No windowing was performed.

No features were extracted.

No model was trained.

No statistics were performed.

No subject was excluded based on performance.


NEXT
----
BLOCK 02 — SIGNAL QUALITY ASSURANCE
"""


(BLOCK01_DIR / "BLOCK01_summary.txt").write_text(
    summary_text,
    encoding="utf-8",
)


# ============================================================
# 30. FINAL CONSOLE REPORT
# ============================================================

print()
print("=" * 78)
print("BLOCK 01 — FINAL CLEAN RESULT")
print("=" * 78)

print(f"STATUS                         : {block01_status}")
print(f"ME scientific CSVs             : {len(me_recordings):,}")
print(f"MI scientific CSVs             : {len(mi_recordings):,}")
print(f"Total scientific recordings    : {len(parsed_rows):,}")
print(f"ME subjects                    : {len(me_subjects):,}")
print(f"MI subjects                    : {len(mi_subjects):,}")
print(f"Paired subjects                : {len(paired_subjects):,}")
print(f"Complete paired subjects       : {len(complete_subjects):,}")
print(f"Unparsed scientific CSVs       : {len(unparsed_rows):,}")
print(f"Missing logical recordings     : {len(missing_keys):,}")
print(f"Unexpected logical keys        : {len(unexpected_keys):,}")
print(f"Duplicate logical groups       : {len(duplicate_groups):,}")
print(f"Unexpected subject IDs         : {len(unexpected_subjects):,}")

print()
print("PASS CONDITIONS:")
for name, passed in pass_conditions.items():
    print(f"  {'✓' if passed else '✗'} {name}")

print()
print("PAIRED SUBJECTS:")
print(paired_subjects)

print()
print("OUTPUT DIRECTORY:")
print(BLOCK01_DIR)

print()
print("RAW DATA MODIFIED: NO")
print("RAW DATA DELETED : NO")
print("OLD BLOCK 01 OUTPUT CLEANED: YES")

print()

if block01_status == "PASS":

    print("=" * 78)
    print("✓ BLOCK 01 PASS")
    print("✓ CURRENT 25-SUBJECT PAIRED COHORT IS STRUCTURALLY COMPLETE")
    print("✓ 525 ME + 525 MI SCIENTIFIC RECORDINGS VERIFIED")
    print("✓ 1,050 LOGICAL RECORDINGS VERIFIED")
    print("✓ NO MISSING LOGICAL RECORDINGS")
    print("✓ NO DUPLICATE LOGICAL RECORDINGS")
    print("✓ NO UNEXPECTED LOGICAL RECORDINGS")
    print("✓ READY TO PROCEED TO BLOCK 02 SIGNAL QA")
    print("=" * 78)

elif block01_status == "REVIEW":

    print("=" * 78)
    print("BLOCK 01 = REVIEW")
    print("DO NOT PROCEED TO BLOCK 02 YET.")
    print("Inspect BLOCK01_review_manifest.csv and related manifests.")
    print("=" * 78)

else:

    raise RuntimeError(
        "Unexpected BLOCK 01 status."
    )
