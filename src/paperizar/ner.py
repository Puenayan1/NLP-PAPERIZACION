import re, json
from typing import List, Dict, Tuple
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.scorer import Scorer

UNITS = [
    "ms", "s", "W", "mW", "kW", "V", "mV", "kV", "A", "mA", "uA",
    "Hz", "kHz", "MHz", "GHz", "THz", "B", "KB", "MB", "GB", "TB",
    "bps", "kbps", "Mbps", "Gbps", "fps", "mm", "cm", "m", "km",
    "g", "kg", "dB", "dBm", "dBi", "mAh", "Wh", "FLOPs", "MFLOPs", "GFLOPs", "TFLOPs",
    "nm", "um", "ns", "us", "rad", "deg"
]

TECH_LEXICON = [
    "F1-score", "F1 score", "ROUGE-L", "BLEU", "BERTScore", "PSNR", "SSIM",
    "RMSE", "MAE", "MSE", "mAP", "IoU", "BER", "SNR", "SINR", "PAPR", "EVM",
    "Throughput", "Raspberry Pi 4", "Raspberry Pi", "Jetson Nano", "Jetson Xavier",
    "Jetson Orin", "Arduino", "STM32", "ESP32", "RTX 2080 Ti", "RTX 3090", "RTX 4090",
    "NVIDIA", "Intel", "Xilinx", "Zynq", " Virtex", "ImageNet", "CIFAR-10", "CIFAR-100",
    "COCO", "KITTI", "MNIST", "SDNET2018", "MobileNet V2", "MobileNetV2", "ResNet-50",
    "ResNet50", "YOLOv5", "YOLOv8", "PyTorch", "TensorFlow", "Kalman filter",
    "Monte Carlo", "Wi-Fi", "Bluetooth", "5G", "6G", "LoRa", "Zigbee"
]

STOP_ACRONYMS = {
    "THE", "AND", "FOR", "ARE", "NOT", "BUT", "ALL", "CAN", "HER", "WAS",
    "ONE", "OUR", "OUT", "YOU", "HAS", "HIS", "HOW", "ITS", "MAY", "NEW",
    "NOW", "OLD", "SEE", "TWO", "WAY", "WHO", "USE", "VIA", "III", "II", "IV"
}

# Expresiones regulares sobre texto completo para máxima precisión en casos con punto final o tokens pegados
UNITS_SORTED = sorted(UNITS, key=len, reverse=True)
UNIT_ALT = "|".join(re.escape(u) for u in UNITS_SORTED)

REGEX_PATTERNS = [
    # 1. TECH compuestos conocidos primero (prioridad para no romper 'Raspberry Pi 4' o 'RTX 2080 Ti')
    ("TECH", re.compile(r'\b(?:' + "|".join(re.escape(t.strip()) for t in sorted(TECH_LEXICON, key=len, reverse=True)) + r')\b', re.IGNORECASE)),
    # 2. TECH alfanuméricos tipo MobileNetV2, SDNET2018, IEEE 802.11ax, FP16, INT8, ResNet-18
    ("TECH", re.compile(r'\b(?:IEEE\s+802\.\d+[a-z]*|[A-Za-z]{2,}[A-Za-z0-9\-]*\d+[A-Za-z0-9\-\.]*)\b')),
    # 3. TECH CamelCase (ej. PointNet, DeepLab, GraphSAGE)
    ("TECH", re.compile(r'\b[A-Z][a-z]+[A-Z][A-Za-z0-9]*\b')),
    # 4. TECH Siglas en mayúsculas de 2 a 8 letras (con plural opcional 'CNNs', 'FPGAs', 'UAVs')
    ("TECH", re.compile(r'\b[A-Z]{2,8}s?\b')),
    # 5. MEASURE: Número (entero o decimal, rango opcional) seguido de unidad (con o sin espacio; punto final excluido del span)
    ("MEASURE", re.compile(rf'(?<![A-Za-z0-9\.\-])\d+(?:\.\d+)?\s*(?:{UNIT_ALT})(?=(?:\s|$|[\.,;:\)\]]))')),
    # 6. MEASURE: Porcentajes pegados o separados ('95.3%', '95.3 %')
    ("MEASURE", re.compile(r'(?<![A-Za-z0-9\.\-])\d+(?:\.\d+)?\s*%')),
    # 7. MEASURE: Escalas en palabras ('3.4 million', '100 thousand')
    ("MEASURE", re.compile(r'(?<![A-Za-z0-9\.\-])\d+(?:\.\d+)?\s+(?:million|billion|thousand)\b', re.IGNORECASE)),
    # 8. MEASURE: Decimales sueltos (ej. '0.95', '3.14', excluyendo versiones precedidas de v/V)
    ("MEASURE", re.compile(r'(?<![A-Za-z0-9\.\-])\d+\.\d+\b')),
]

def get_spacy_ruler_patterns() -> List[Dict]:
    unit_regex = rf"^({UNIT_ALT})\.?$"
    patterns = []
    # MEASURE: número + unidad en 2 tokens (acepta punto pegado como 'W.')
    patterns.append({"label": "MEASURE", "pattern": [{"LIKE_NUM": True}, {"TEXT": {"REGEX": unit_regex}}]})
    # MEASURE: número y unidad pegados en 1 token ('12ms', '2.4GHz', '50W.')
    patterns.append({"label": "MEASURE", "pattern": [{"TEXT": {"REGEX": rf"^\d+(?:\.\d+)?({UNIT_ALT})\.?$"}}]})
    # MEASURE: porcentaje en 2 tokens ('95.3', '%') o 1 token ('95.3%')
    patterns.append({"label": "MEASURE", "pattern": [{"LIKE_NUM": True}, {"TEXT": "%"}]})
    patterns.append({"label": "MEASURE", "pattern": [{"TEXT": {"REGEX": r"^\d+(?:\.\d+)?%$"}}]})
    # MEASURE: million / billion / thousand
    patterns.append({"label": "MEASURE", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["million", "billion", "thousand"]}}]})
    # MEASURE: decimales sueltos
    patterns.append({"label": "MEASURE", "pattern": [{"TEXT": {"REGEX": r"^\d+\.\d+$"}}]})

    # TECH: siglas, CamelCase, alfanuméricos y lexicón
    patterns.append({"label": "TECH", "pattern": [{"TEXT": {"REGEX": r"^[A-Z]{2,8}s?$"}}]})
    patterns.append({"label": "TECH", "pattern": [{"TEXT": {"REGEX": r"^[A-Z][a-z]+[A-Z][A-Za-z0-9]*$"}}]})
    patterns.append({"label": "TECH", "pattern": [{"TEXT": {"REGEX": r"^[A-Za-z]{2,}[A-Za-z0-9\-]*\d+[A-Za-z0-9\-]*$"}}]})
    for item in TECH_LEXICON:
        toks = item.strip().split()
        patterns.append({"label": "TECH", "pattern": [{"LOWER": t.lower()} for t in toks]})
    return patterns

def build_rules_only_nlp() -> spacy.Language:
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
    ruler.add_patterns(get_spacy_ruler_patterns())
    return nlp

def build_hybrid_nlp(trained_model_path: str) -> spacy.Language:
    nlp = spacy.load(trained_model_path)
    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")
    # Prioridad a las reglas: entity_ruler ANTES de ner, y el ner entrenado respeta spans ya fijados;
    # además según Sección 5-A ponemos entity_ruler antes de ner con overwrite_ents=True.
    ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
    ruler.add_patterns(get_spacy_ruler_patterns())
    return nlp

def preannotate_text(text: str, nlp_tok: spacy.Language) -> List[Dict]:
    """Genera spans limpios alineados a bordes de tokens de spaCy sin solapamientos."""
    doc = nlp_tok.make_doc(text)
    raw_spans = []
    occupied = [False] * len(text)

    for label, regex in REGEX_PATTERNS:
        for m in regex.finditer(text):
            s, e = m.start(), m.end()
            frag = text[s:e]
            if label == "TECH" and frag.rstrip('s') in STOP_ACRONYMS:
                continue
            if any(occupied[s:e]):
                continue
            span = doc.char_span(s, e, label=label, alignment_mode="contract")
            if span is None:
                span = doc.char_span(s, e, label=label, alignment_mode="expand")
            if span is None or not span.text.strip():
                continue
            # Quitar punto final si quedó expandido por 'W.'
            clean_end = span.end_char
            if text[span.start_char:clean_end].endswith(".") and not text[span.start_char:clean_end].endswith(".."):
                sub = doc.char_span(span.start_char, clean_end - 1, label=label, alignment_mode="contract")
                if sub is not None:
                    span = sub
            if any(occupied[span.start_char:span.end_char]):
                continue
            for i in range(span.start_char, span.end_char):
                occupied[i] = True
            raw_spans.append({
                "start": span.start_char,
                "end": span.end_char,
                "label": label,
                "text": text[span.start_char:span.end_char]
            })
    raw_spans.sort(key=lambda x: x["start"])
    return raw_spans

def compute_inter_annotator_f1(ann_a: List[List[Tuple[int, int, str]]], ann_b: List[List[Tuple[int, int, str]]]) -> Dict[str, float]:
    tp = fp = fn = 0
    per_label = {"MEASURE": [0, 0, 0], "TECH": [0, 0, 0]}
    for sa, sb in zip(ann_a, ann_b):
        set_a, set_b = set(sa), set(sb)
        tp += len(set_a & set_b)
        fp += len(set_b - set_a)
        fn += len(set_a - set_b)
        for lbl in per_label:
            la = {x for x in set_a if x[2] == lbl}
            lb = {x for x in set_b if x[2] == lbl}
            per_label[lbl][0] += len(la & lb)
            per_label[lbl][1] += len(lb - la)
            per_label[lbl][2] += len(la - lb)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    out = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    for lbl, (t, p, n) in per_label.items():
        f_lbl = (2 * t) / (2 * t + p + n) if (2 * t + p + n) else 0.0
        out[f"f1_{lbl}"] = round(f_lbl, 4)
    return out

def export_spacy_docbin(records: List[Dict], split_name: str, out_path: str):
    nlp = spacy.blank("en")
    db = DocBin()
    skipped = 0
    for rec in records:
        if rec["ner_split"] != split_name:
            continue
        doc = nlp.make_doc(rec["A"])
        ents = []
        for sp in rec["entities"]:
            span = doc.char_span(sp["start"], sp["end"], label=sp["label"], alignment_mode="contract")
            if span is not None:
                ents.append(span)
            else:
                skipped += 1
        doc.ents = spacy.util.filter_spans(ents)
        db.add(doc)
    db.to_disk(out_path)
    return skipped

def evaluate_nlp_on_records(nlp: spacy.Language, records: List[Dict], split_name: str = "test") -> Dict:
    examples = []
    for rec in records:
        if rec["ner_split"] != split_name:
            continue
        pred_doc = nlp(rec["A"])
        gold_doc = nlp.make_doc(rec["A"])
        ents = []
        for sp in rec["entities"]:
            s = gold_doc.char_span(sp["start"], sp["end"], label=sp["label"], alignment_mode="contract")
            if s is not None:
                ents.append(s)
        gold_doc.ents = spacy.util.filter_spans(ents)
        examples.append(Example(pred_doc, gold_doc))
    scorer = Scorer()
    scores = scorer.score(examples)
    return {
        "ents_p": round(scores["ents_p"] or 0.0, 4),
        "ents_r": round(scores["ents_r"] or 0.0, 4),
        "ents_f": round(scores["ents_f"] or 0.0, 4),
        "ents_per_type": {
            k: {m: round(v or 0.0, 4) for m, v in vals.items()}
            for k, vals in (scores.get("ents_per_type") or {}).items()
        }
    }

def extract_entities_list(nlp: spacy.Language, text: str) -> List[Dict]:
    doc = nlp(text)
    return [{"text": ent.text.rstrip("."), "label": ent.label_, "start": ent.start_char, "end": ent.end_char} for ent in doc.ents]
