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

def _find_nearby(text: str, keyword: str) -> float | None:
    pats = [
        rf"{keyword}[:\s]*(\d+[,.]?\d*)",
        rf"{keyword}[:\s]+(?:Rs\.?|INR|₹)?\s*(\d+[,.]?\d*)",
        rf"(\d+[,.]?\d*)\s*{keyword}",
    ]
    for pat in pats:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = m.group(1).replace(",", "")
            return float(val)
    return None

def parse_bill_data(text: str) -> dict:
    result = {}
    tl = text.lower()
    lines = text.split("\n")

    amount = (_find_nearby(tl, "net payable") or _find_nearby(tl, "net amount")
              or _find_nearby(tl, "total amount") or _find_nearby(tl, "bill amount"))
    if not amount:
        for line in lines:
            if re.search(r'(?:rs\.?|inr|₹)\s*\d+', line, re.IGNORECASE):
                m = re.search(r'(\d+[,.]?\d*)', line)
                if m:
                    v = float(m.group(1).replace(",", ""))
                    if 100 < v < 50000:
                        amount = v
                        break
    if amount:
        result["net_payable"] = amount

    units = (_find_nearby(tl, "units consumed") or _find_nearby(tl, "consumption")
             or _find_nearby(tl, "energy consumed"))
    if not units:
        for line in lines:
            if re.search(r'\d+[,.]?\d*\s*(?:kWh|kwh|KWH)', line):
                m = re.search(r'(\d+[,.]?\d*)', line)
                if m:
                    v = float(m.group(1).replace(",", ""))
                    if 20 < v < 2000:
                        units = v
                        break
    if not units:
        units = _find_nearby(tl, "kwh")
    if units:
        result["units_consumed"] = round(units, 1)

    fixed = (_find_nearby(tl, "fixed charges") or _find_nearby(tl, "fixed"))
    if fixed:
        result["fixed_charges"] = fixed

    energy = (_find_nearby(tl, "energy charges") or _find_nearby(tl, "energy"))
    if energy:
        result["energy_charges"] = energy

    fppca = _find_nearby(tl, "fppca")
    if fppca:
        result["fppca_charges"] = fppca

    pg = _find_nearby(tl, "pg surcharge") or _find_nearby(tl, "p&g")
    if pg:
        result["pg_surcharge"] = pg

    tax = (_find_nearby(tl, "tax") or _find_nearby(tl, "gst"))
    if tax:
        result["tax_amount"] = tax

    penalty = (_find_nearby(tl, "md penalty") or _find_nearby(tl, "ex load")
               or _find_nearby(tl, "penalty"))
    if penalty:
        result["md_penalty"] = penalty

    true_up = (_find_nearby(tl, "true.up") or _find_nearby(tl, "trueup")
               or _find_nearby(tl, "fy adjustment"))
    if true_up:
        result["true_up_charges"] = true_up

    arrears = _find_nearby(tl, "arrears")
    if arrears:
        result["arrears"] = arrears

    month_map = {
        "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
        "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
        "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
        "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12
    }
    for line in lines:
        ll = line.lower()
        for name, num in month_map.items():
            if name in ll:
                result["month"] = num
                break
        if "month" in result:
            break
    for pat in [r"\b(20\d{2})\b"]:
        m = re.search(pat, text)
        if m:
            yr = int(m.group(1))
            if 2020 <= yr <= 2035:
                result["year"] = yr
                break

    present = _find_nearby(tl, "present reading") or _find_nearby(tl, "present")
    if present:
        result["present_reading"] = present
    previous = _find_nearby(tl, "previous reading") or _find_nearby(tl, "previous")
    if previous:
        result["previous_reading"] = previous
    md = _find_nearby(tl, "recorded md") or _find_nearby(tl, "recorded")
    if md:
        result["recorded_md"] = md
    pf = _find_nearby(tl, "power factor")
    if pf:
        result["power_factor"] = pf

    return result

def parse_uploaded_bill(image: Image.Image) -> dict:
    text = extract_text(image)
    data = parse_bill_data(text)
    data["_raw_text"] = text[:800]
    return data
