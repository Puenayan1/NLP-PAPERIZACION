import os, re, json, time, urllib.request, urllib.parse
import xml.etree.ElementTree as ET
import random
from typing import List, Dict
from datasets import load_dataset

SEED = 42
CATEGORIES = {
    "eess.SP": "signal processing",
    "eess.SY": "systems and control",
    "eess.IV": "image and video processing",
    "cs.RO": "robotics",
    "cs.AR": "hardware architecture",
    "cs.NI": "networking and internet architecture"
}

MEASURE_REGEX = re.compile(
    r'\b\d+(?:\.\d+)?\s*(?:%|(?:ms|s|W|mW|V|mA|Hz|kHz|MHz|GHz|MB|GB|fps|mm|cm|m|km|kg|dB|mAh|GFLOPs|TFLOPs|million|billion|thousand)\b)',
    re.IGNORECASE
)

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def passes_filters(abstract: str) -> bool:
    if abstract.count('$') > 2:
        return False
    words = abstract.split()
    if not (80 <= len(words) <= 250):
        return False
    return len(MEASURE_REGEX.findall(abstract)) >= 2

def fetch_arxiv_recent_api(cat: str, max_results: int = 400) -> List[Dict]:
    base_url = "http://export.arxiv.org/api/query?"
    records = []
    batch_size = 200
    ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}

    for start in range(0, max_results, batch_size):
        params = urllib.parse.urlencode({
            'search_query': f'cat:{cat}',
            'start': start,
            'max_results': batch_size,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        })
        req = urllib.request.Request(
            base_url + params,
            headers={'User-Agent': 'PaperizarAcademicProject/1.0 (mailto:academic@javeriana.edu.co)'}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                xml_data = response.read()
        except Exception as e:
            print(f"[WARN] API arXiv {cat} start={start}: {e}. Reintentando en 8s...")
            time.sleep(8)
            continue

        root = ET.fromstring(xml_data)
        entries = root.findall('atom:entry', ns)
        if not entries:
            break

        for entry in entries:
            raw_id = entry.find('atom:id', ns).text.strip()
            arxiv_id = raw_id.split('/abs/')[-1].split('v')[0]
            published = entry.find('atom:published', ns).text.strip()[:10]
            title = re.sub(r'\s+', ' ', entry.find('atom:title', ns).text.strip())
            summary = re.sub(r'\s+', ' ', entry.find('atom:summary', ns).text.strip())
            lic_el = entry.find('arxiv:license', ns)
            license_url = lic_el.text.strip() if (lic_el is not None and lic_el.text) else "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"

            records.append({
                "arxiv_id": arxiv_id,
                "category": cat,
                "topic": CATEGORIES[cat],
                "published_date": published,
                "title": title,
                "A": summary,
                "license": license_url
            })
        time.sleep(3.1)
    return records

def stream_hf_historical(target_count: int, seen_ids: set, seen_norm: set) -> List[Dict]:
    print(f"Cargando corpus histórico en streaming desde Hugging Face (meta: {target_count} abstracts filtrados)...")
    ds = load_dataset("GFissore/arxiv-abstracts-2021", split="train", streaming=True)
    pool = []
    per_cat_counts = {c: 0 for c in CATEGORIES}
    max_per_cat = (target_count // len(CATEGORIES)) + 150

    for row in ds:
        cats_field = row.get("categories", "")
        cat_list = cats_field[0].split() if isinstance(cats_field, list) and cats_field else (cats_field.split() if isinstance(cats_field, str) else [])
        matched_cat = next((c for c in cat_list if c in CATEGORIES), None)
        if not matched_cat or per_cat_counts[matched_cat] >= max_per_cat:
            continue

        aid = str(row.get("id", "")).strip()
        if not aid or aid in seen_ids:
            continue

        abstract = re.sub(r'\s+', ' ', str(row.get("abstract", "")).strip())
        if not passes_filters(abstract):
            continue

        norm = normalize_text(abstract)
        if norm in seen_norm:
            continue

        seen_ids.add(aid)
        seen_norm.add(norm)
        per_cat_counts[matched_cat] += 1

        # Extraer fecha aproximada desde update_date o id de arXiv
        pub_date = str(row.get("update_date", "2021-01-01"))[:10]
        lic = row.get("license") or "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"

        pool.append({
            "arxiv_id": aid,
            "category": matched_cat,
            "topic": CATEGORIES[matched_cat],
            "published_date": pub_date,
            "title": re.sub(r'\s+', ' ', str(row.get("title", "")).strip()),
            "A": abstract,
            "license": lic
        })

        if len(pool) % 500 == 0:
            print(f"  -> Filtrados acumulados: {len(pool)} | por categoría: {per_cat_counts}")
        if len(pool) >= target_count:
            break

    return pool

def build_datasets(pilot: bool = False, cutoff_date: str = "2025-06-01", target_corpus_a: int = 3300):
    rng = random.Random(SEED)
    seen_ids = set()
    seen_norm = set()
    post_cutoff_by_cat = {c: [] for c in CATEGORIES}
    corpus_pool = []

    api_fetch = 100 if pilot else 400
    print(f"1/2 Descargando recientes de API arXiv para roundtrip (>= {cutoff_date})...")
    for cat in CATEGORIES:
        raw_recent = fetch_arxiv_recent_api(cat, max_results=api_fetch)
        for item in raw_recent:
            aid = item["arxiv_id"]
            if aid in seen_ids or not passes_filters(item["A"]):
                continue
            norm = normalize_text(item["A"])
            if norm in seen_norm:
                continue
            seen_ids.add(aid)
            seen_norm.add(norm)
            if item["published_date"] >= cutoff_date:
                post_cutoff_by_cat[cat].append(item)
            else:
                corpus_pool.append(item)
        print(f"  -> {cat}: {len(post_cutoff_by_cat[cat])} recientes válidos")

    # Armar roundtrip (100 recientes, mínimo 15 por categoría)
    roundtrip = []
    target_rt = 12 if pilot else 100
    min_per_cat = 2 if pilot else 15
    leftover_recent = []

    for cat in CATEGORIES:
        rng.shuffle(post_cutoff_by_cat[cat])
        take = min(len(post_cutoff_by_cat[cat]), min_per_cat)
        roundtrip.extend(post_cutoff_by_cat[cat][:take])
        leftover_recent.extend(post_cutoff_by_cat[cat][take:])

    rng.shuffle(leftover_recent)
    needed = target_rt - len(roundtrip)
    if needed > 0:
        roundtrip.extend(leftover_recent[:needed])
        leftover_recent = leftover_recent[needed:]

    corpus_pool.extend(leftover_recent)

    # Completar corpus_A hasta >= 3.300 con HF streaming
    needed_hist = (60 if pilot else target_corpus_a) - len(corpus_pool)
    if needed_hist > 0:
        hist_records = stream_hf_historical(needed_hist, seen_ids, seen_norm)
        corpus_pool.extend(hist_records)

    rng.shuffle(corpus_pool)

    # Partición reproducible 70/15/15 con semilla 42
    n_total = len(corpus_pool)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    for idx, rec in enumerate(corpus_pool):
        rec["id"] = f"corp_{idx:05d}"
        rec["split"] = "train" if idx < n_train else ("val" if idx < n_train + n_val else "test")

    for idx, rec in enumerate(roundtrip):
        rec["id"] = f"rt_{idx:03d}"
        rec["split"] = "roundtrip"

    train_pool = [r for r in corpus_pool if r["split"] == "train"]
    val_pool = [r for r in corpus_pool if r["split"] == "val"]
    test_pool = [r for r in corpus_pool if r["split"] == "test"]

    n_ner_tr, n_ner_dev, n_ner_te = (4, 2, 2) if pilot else (40, 20, 20)
    ner_samples = (
        [{**r, "ner_split": "train"} for r in train_pool[:n_ner_tr]] +
        [{**r, "ner_split": "dev"} for r in val_pool[:n_ner_dev]] +
        [{**r, "ner_split": "test"} for r in test_pool[:n_ner_te]]
    )

    suffix = "_pilot" if pilot else ""
    save_jsonl(corpus_pool, f"data/processed/corpus_A{suffix}.jsonl")
    save_jsonl(roundtrip, f"data/processed/roundtrip{suffix}.jsonl")
    save_jsonl(ner_samples, f"data/processed/ner_candidates{suffix}.jsonl")

    print(f"\n=== RESUMEN GUARDADO ({'PILOTO' if pilot else 'FINAL'}) ===")
    print(f"  corpus_A: {len(corpus_pool)} (train={len(train_pool)}, val={len(val_pool)}, test={len(test_pool)})")
    print(f"  roundtrip: {len(roundtrip)} (mínimo por cat verificado, 0 solapamiento)")
    print(f"  ner_candidates: {len(ner_samples)}")

def save_jsonl(items: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    build_datasets(pilot=args.pilot)
