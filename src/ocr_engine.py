import sys
import re
import numpy as np
from PIL import Image

_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        if sys.stdout.encoding != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8")
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader

def preprocess_image(image: Image.Image) -> np.ndarray:
    img = np.array(image)
    try:
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        scale = 1200 / max(gray.shape)
        if scale < 1:
            w = int(gray.shape[1] * scale)
            h = int(gray.shape[0] * scale)
            gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        denoised = cv2.fastNlMeansDenoising(gray, h=20)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    except Exception:
        return np.array(image)

def extract_text(image: Image.Image) -> str:
    processed = preprocess_image(image)
    reader = _get_reader()
    results = reader.readtext(processed, paragraph=False, width_ths=0.7)
    lines = {}
    for bbox, text, conf in results:
        if conf < 0.3:
            continue
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        y_key = round(y_center / 15) * 15
        if y_key not in lines:
            lines[y_key] = []
        lines[y_key].append((bbox[0][0], text, conf))
    sorted_lines = []
    for y_key in sorted(lines.keys()):
        sorted_line = sorted(lines[y_key], key=lambda x: x[0])
        line_text = " ".join(t for _, t, _ in sorted_line)
        sorted_lines.append(line_text)
    return "\n".join(sorted_lines)

def _normalize_text(text: str) -> str:
    text = re.sub(r"\b[A-Za-z]+\d+[A-Za-z0-9]+\b", " ", text)
    text = re.sub(r"\b\d+[A-Za-z]+[A-Za-z0-9]*\b", " ", text)
    text = re.sub(r"\b\d{8,}\b", " ", text)
    text = text.replace(",", ".").replace("-", ".").replace("x", ".")
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
    return text

def _get_clean_numbers(text: str) -> list[float]:
    text = _normalize_text(text)
    raw = re.findall(r"\d+\.?\d*", text)
    seen = set()
    out = []
    for r in raw:
        if len(r) >= 7 and "." not in r:
            continue
        if r in ("0", "0.", "0.0"):
            continue
        try:
            v = float(r)
        except ValueError:
            continue
        if v <= 0:
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out

def _parse_dates(text: str) -> tuple[int | None, int | None]:
    for pat in [r"(\d{2})/(\d{2})/(\d{4})", r"(\d{2})-(\d{2})-(\d{4})"]:
        for m in re.finditer(pat, text):
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1 <= mo <= 12 and 2020 <= y <= 2035:
                return mo, y
    return None, None

from src.ml_models import calculate_bescom_bill as _calc_bill

def parse_bill_data(text: str) -> dict:
    result = {}
    nums = _get_clean_numbers(text)

    month, year = _parse_dates(text)
    if month: result["month"] = month
    if year: result["year"] = year

    result["fixed_charges"] = 200.0

    candidates = sorted([v for v in nums if 2000 <= v <= 9999], reverse=True)
    readings = []
    for i, v in enumerate(candidates):
        if not readings:
            readings.append(v)
        elif len(readings) == 1:
            if abs(readings[0] - v) < 1000 and v != readings[0]:
                readings.append(v)
        else:
            break
    readings = sorted(readings)
    if len(readings) >= 2:
        result["present_reading"] = readings[-1]
        result["previous_reading"] = readings[-2]
    elif len(readings) == 1:
        result["present_reading"] = readings[0]

    readings_vals = set(readings) | {2026.0, 2025.0, 2024.0, 2023.0}
    units = None
    for v in nums:
        if 20 <= v <= 500 and v not in readings_vals:
            units = v
            result["units_consumed"] = units
            break
    if units is None and "present_reading" in result and "previous_reading" in result:
        diff = round(result["present_reading"] - result["previous_reading"], 1)
        if 20 <= diff <= 500:
            units = diff
            result["units_consumed"] = units

    pf = None
    for v in nums:
        if 0.85 <= v <= 1.0:
            pf = v
            result["power_factor"] = v
            break

    md_raw = None
    md_match = re.search(r"([LIl1])?(\d{2,3})\.?\s*(?:KW|OKW|MD|md|kw)", text)
    if md_match:
        prefix = md_match.group(1)
        digits = md_match.group(2)
        try:
            if prefix and prefix in "LIl" and len(digits) == 2:
                md_raw = float(f"1.{digits}")
            else:
                md_raw = float(digits)
        except ValueError:
            pass
    if md_raw is not None:
        if 0.5 <= md_raw <= 3.0:
            md = round(md_raw, 3)
            result["recorded_md"] = md
    if md is None:
        for v in nums:
            if 0.5 <= v <= 3.0 and v != pf:
                md = round(v, 3)
                result["recorded_md"] = md
                break
    if md is None:
        md = 0.5
        result["recorded_md"] = md

    net_payable_ocr = None
    for v in sorted(nums, reverse=True):
        if 1500 <= v <= 5000 and v not in readings:
            net_payable_ocr = v
            break

    if units and md is not None:
        calc = _calc_bill(units, md)
        for k, v in calc.items():
            if k == "net_payable":
                result[k] = net_payable_ocr or v
            else:
                result[k] = v
    else:
        result["net_payable"] = net_payable_ocr or 0

    if units and md is not None and result.get("net_payable"):
        base = _calc_bill(units, md)
        base_total = base["net_payable"]
        extracted_total = next((v for v in sorted(nums, reverse=True)
                                if 1500 <= v <= 5000 and v not in readings), 0)
        if extracted_total > base_total:
            result["true_up_charges"] = round(extracted_total - base_total, 2)
            result["net_payable"] = extracted_total

    return result

def parse_uploaded_bill(image: Image.Image) -> dict:
    text = extract_text(image)
    data = parse_bill_data(text)
    data["_raw_text"] = text[:800]
    return data
