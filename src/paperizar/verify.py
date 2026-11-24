import os, re, json
from typing import Dict, List, Tuple
import numpy as np
import spacy
import textstat
from src.paperizar.informalize import entity_present, norm_ws, clean_measure

UNIT_SPLIT_RE = re.compile(r'^(\d+(?:\.\d+)?)\s*(%|million|billion|thousand|[A-Za-z]+)$')

def compute_corpus_flesch_bounds(
    corpus_path: str = "data/processed/corpus_A.jsonl",
    cache_path: str = "configs/flesch_bounds.json"
) -> Tuple[float, float]:
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            d = json.load(f)
        return float(d["p10"]), float(d["p90"])

    scores = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") == "train":
                scores.append(textstat.flesch_reading_ease(r["A"]))

    p10, p90 = float(np.percentile(scores, 10)), float(np.percentile(scores, 90))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"p10": round(p10, 4), "p90": round(p90, 4), "n_train": len(scores)}, f, indent=2)
    return p10, p90

def parse_num_unit(measure_txt: str) -> Tuple[str, str]:
    c = clean_measure(measure_txt)
    m = UNIT_SPLIT_RE.match(c)
    if m:
        return m.group(1), m.group(2)
    return "", ""

def detect_units_changed(ents_in: List[Dict], ents_out: List[Dict]) -> List[Dict]:
    in_map = {}
    for e in ents_in:
        if e["label"] == "MEASURE":
            num, unit = parse_num_unit(e["text"])
            if num and unit:
                in_map.setdefault(num, set()).add(unit)

    changes = []
    for e in ents_out:
        if e["label"] == "MEASURE":
            num, unit_out = parse_num_unit(e["text"])
            if num and unit_out and num in in_map:
                if unit_out not in in_map[num]:
                    changes.append({
                        "number": num,
                        "input_units": sorted(in_map[num]),
                        "output_unit": unit_out,
                        "span": e["text"]
                    })
    return changes

def verify_single_rewrite(
    input_text: str,
    output_text: str,
    nlp: spacy.Language,
    flesch_bounds: Tuple[float, float]
) -> Dict:
    """Genera el reporte JSON oficial de verificación (Sección 7 / Página 18)."""
    doc_in = nlp(input_text)
    doc_out = nlp(output_text)
    ents_in = [{"text": e.text.rstrip("."), "label": e.label_} for e in doc_in.ents]
    ents_out = [{"text": e.text.rstrip("."), "label": e.label_} for e in doc_out.ents]

    lost = [e["text"] for e in ents_in if not entity_present(e, ents_out, output_text)]
    added = [e["text"] for e in ents_out if not entity_present(e, ents_in, input_text)]
    u_changed = detect_units_changed(ents_in, ents_out)

    fl_in = round(float(textstat.flesch_reading_ease(input_text)), 2)
    fl_out = round(float(textstat.flesch_reading_ease(output_text)), 2)
    p10, p90 = flesch_bounds
    in_range = bool(p10 <= fl_out <= p90)

    warnings = []
    if lost:
        warnings.append(f"Se omitieron {len(lost)} entidades técnicas de la entrada: {', '.join(lost[:5])}.")
    if added:
        warnings.append(f"Aparecen {len(added)} entidades nuevas no detectadas en la entrada (posible expansión de siglas o alucinación): {', '.join(added[:5])}.")
    if u_changed:
        desc = ", ".join(f"{c['number']} ({'/'.join(c['input_units'])} -> {c['output_unit']})" for c in u_changed)
        warnings.append(f"Alerta crítica de unidades alteradas: {desc}.")
    if fl_out > fl_in:
        warnings.append(f"El texto de salida no aumentó su complejidad académica (Flesch subió de {fl_in} a {fl_out}).")
    if not in_range:
        warnings.append(f"Legibilidad Flesch ({fl_out}) fuera del rango típico [p10={p10:.1f}, p90={p90:.1f}] de abstracts de ingeniería.")

    return {
        "output": output_text,
        "entities": {
            "lost": lost,
            "added": added
        },
        "units_changed": u_changed,
        "flesch": {
            "input": fl_in,
            "output": fl_out,
            "reference_p10_p90": [round(p10, 2), round(p90, 2)],
            "in_range": in_range
        },
        "warnings": warnings
    }

def compute_pair_entity_metrics(
    text_a: str,
    text_i: str,
    text_aprime: str,
    nlp: spacy.Language,
    flesch_bounds: Tuple[float, float]
) -> Dict:
    doc_a = nlp(text_a)
    doc_i = nlp(text_i)
    doc_ap = nlp(text_aprime)

    ents_a = [{"text": e.text.rstrip("."), "label": e.label_} for e in doc_a.ents]
    ents_i = [{"text": e.text.rstrip("."), "label": e.label_} for e in doc_i.ents]
    ents_ap = [{"text": e.text.rstrip("."), "label": e.label_} for e in doc_ap.ents]

    ent_cov_A = sum(1 for e in ents_a if entity_present(e, ents_ap, text_aprime)) / len(ents_a) if ents_a else 1.0
    ent_prec_A = sum(1 for e in ents_ap if entity_present(e, ents_a, text_a)) / len(ents_ap) if ents_ap else 1.0
    ent_cov_I = sum(1 for e in ents_i if entity_present(e, ents_ap, text_aprime)) / len(ents_i) if ents_i else 1.0

    # unit_preserved: todas las cifras con unidad de I (o A presentes en I) se conservan idénticas sin cambio de unidad
    measures_ref = [e for e in ents_a if e["label"] == "MEASURE" and entity_present(e, ents_i, text_i)]
    all_meas_kept = all(entity_present(m, ents_ap, text_aprime) for m in measures_ref)
    no_unit_swap = (len(detect_units_changed(ents_i, ents_ap)) == 0)
    unit_preserved = bool(all_meas_kept and no_unit_swap)

    fl_a = float(textstat.flesch_reading_ease(text_a))
    fl_i = float(textstat.flesch_reading_ease(text_i))
    fl_ap = float(textstat.flesch_reading_ease(text_aprime))
    p10, p90 = flesch_bounds

    return {
        "ent_cov_A": round(ent_cov_A, 4),
        "ent_prec_A": round(ent_prec_A, 4),
        "ent_cov_I": round(ent_cov_I, 4),
        "unit_preserved": unit_preserved,
        "flesch_A": round(fl_a, 2),
        "flesch_I": round(fl_i, 2),
        "flesch_Aprime": round(fl_ap, 2),
        "flesch_harder_than_I": bool(fl_ap < fl_i),
        "flesch_in_range": bool(p10 <= fl_ap <= p90)
    }
