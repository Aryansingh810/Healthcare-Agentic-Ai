import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class OCRResult:
    raw_text: str
    fields: dict[str, str]
    confidence: float


def _avg_confidence_from_tesseract(image_path: Path) -> float:
    import pytesseract

    cmd = os.environ.get("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd

    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    confs = [int(c) for c in data["conf"] if str(c).isdigit() and int(c) >= 0]
    if not confs:
        return 0.0
    return sum(confs) / len(confs)


def _extract_email(text: str) -> str | None:
    m = re.search(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        text,
    )
    return m.group(0).strip() if m else None


def _extract_phone(text: str) -> str | None:
    patterns = [
        r"\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        r"\d{10}",
        r"\d{3}[-.\s]\d{3}[-.\s]\d{4}",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return re.sub(r"\s+", " ", m.group(0)).strip()
    return None


def _extract_id_number(text: str) -> str | None:
    m = re.search(r"(?:ID|License|No\.?|#)\s*[:\-]?\s*([A-Z0-9\-]{4,})", text, re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"\b([A-Z]{1,3}\d{6,12})\b", text)
    return m.group(1).strip() if m else None


def _extract_specialty(text: str) -> str | None:
    common = [
        "cardiology",
        "dermatology",
        "endocrinology",
        "neurology",
        "oncology",
        "pediatrics",
        "psychiatry",
        "radiology",
        "surgery",
        "internal medicine",
        "family medicine",
        "orthopedic",
    ]
    lower = text.lower()
    for s in common:
        if s in lower:
            return s.title()
    m = re.search(r"(?:Specialty|Specialisation|Specialization)\s*[:\-]\s*([^\n]+)", text, re.I)
    return m.group(1).strip()[:120] if m else None


def _extract_name(text: str) -> str | None:
    m = re.search(r"(?:Dr\.?|Name)\s*[:\-]\s*([A-Za-z][A-Za-z\s.'\-]{2,60})", text, re.I)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:5]:
        if re.match(r"^[A-Za-z][A-Za-z\s.'\-]{2,50}$", ln) and len(ln.split()) <= 5:
            return ln
    return None


def run_ocr(image_path: str | Path) -> OCRResult:
    import pytesseract

    cmd = os.environ.get("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    img = Image.open(path)
    raw = pytesseract.image_to_string(img)
    conf = _avg_confidence_from_tesseract(path)

    fields: dict[str, str] = {}
    name = _extract_name(raw)
    if name:
        fields["name"] = name
    spec = _extract_specialty(raw)
    if spec:
        fields["specialty"] = spec
    phone = _extract_phone(raw)
    if phone:
        fields["phone"] = phone
    email = _extract_email(raw)
    if email:
        fields["email"] = email
    id_no = _extract_id_number(raw)
    if id_no:
        fields["id_number"] = id_no

    return OCRResult(raw_text=raw.strip(), fields=fields, confidence=conf)


def ocr_result_to_dict(result: OCRResult) -> dict[str, Any]:
    return {
        "raw_text": result.raw_text,
        "fields": result.fields,
        "confidence": round(result.confidence, 2),
    }
