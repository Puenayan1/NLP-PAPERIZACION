import os, json, random
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import spacy
from scipy.stats import wilcoxon
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from tqdm.auto import tqdm
from src.paperizar.verify import compute_corpus_flesch_bounds, compute_pair_entity_metrics

SEED = 42

def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 42) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)

def holm_bonferroni(p_values: List[float]) -> List[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.zeros(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = p_values[idx] * (m - rank)
        running_max = max(running_max, min(1.0, adj))
        adjusted[idx] = running_max
    return adjusted.tolist()

def evaluate_all_systems(
    paperized_jsonl: str = "data/processed/paperized_roundtrip.jsonl",
    out_roundtrip_jsonl: str = "results/roundtrip.jsonl",
    strict_keep_only: bool = True
) -> pd.DataFrame:
    nlp = spacy.load("models/ner_hybrid_v1")
    flesch_bounds = compute_corpus_flesch_bounds()
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    with open(paperized_jsonl, encoding="utf-8") as f:
        raw_rows = [json.loads(line) for line in f if line.strip()]

    if strict_keep_only:
        rows = [r for r in raw_rows if not r.get("fallback_100_coverage", False)]
    else:
        rows = raw_rows

    # Añadir control de identidad (A' = I) exigido en Paso P7 una vez por borrador único
    seen_drafts = {}
    for r in rows:
        if r["id"] not in seen_drafts:
            ctrl = dict(r)
            ctrl["A_prime"] = r["I"]
            ctrl["model_id"] = "control_identity"
            ctrl["prompt_id"] = "CTRL-IDENTITY"
            ctrl["prompt_hash"] = "000000000000"
            seen_drafts[r["id"]] = ctrl

    all_eval_rows = rows + list(seen_drafts.values())
    print(f"Calculando BERTScore sobre {len(all_eval_rows)} salidas (Herramienta + Línea Base + Control Identidad)...")

    cands = [r["A_prime"] for r in all_eval_rows]
    refs = [r["A"] for r in all_eval_rows]
    _, _, F1_bert = bert_score_fn(
        cands,
        refs,
        lang="en",
        rescale_with_baseline=True,
        batch_size=32,
        verbose=True
    )
    f1_list = F1_bert.tolist()

    enriched = []
    for r, bf1 in tqdm(zip(all_eval_rows, f1_list), total=len(all_eval_rows), desc="Métricas NER + Flesch + ROUGE"):
        rl = rouge.score(r["A"], r["A_prime"])["rougeL"].fmeasure
        m_dict = compute_pair_entity_metrics(r["A"], r["I"], r["A_prime"], nlp, flesch_bounds)
        rec_out = {
            **r,
            "bertscore_f1": round(float(bf1), 4),
            "rougeL_f1": round(float(rl), 4),
            **m_dict
        }
        enriched.append(rec_out)

    os.makedirs(os.path.dirname(out_roundtrip_jsonl), exist_ok=True)
    with open(out_roundtrip_jsonl, "w", encoding="utf-8") as fout:
        for r in enriched:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    return pd.DataFrame(enriched)

def run_statistical_tests(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "ent_cov_A", "ent_prec_A", "ent_cov_I", "unit_preserved",
        "bertscore_f1", "rougeL_f1", "flesch_Aprime", "flesch_in_range", "flesch_harder_than_I"
    ]

    # 1. Resumen global por sistema con IC 95% Bootstrap (10.000 remuestreos)
    sys_rows = []
    for sys_id, grp in df.groupby("prompt_id"):
        row = {"Sistema": sys_id, "N_pares": len(grp)}
        for m in metrics:
            mean_v, lo, hi = bootstrap_ci(grp[m].astype(float).values, n_boot=10000, seed=SEED)
            row[f"{m}_mean"] = round(mean_v, 4)
            row[f"{m}_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
        sys_rows.append(row)
    df_sys = pd.DataFrame(sys_rows)

    # 2. Comparación pareada Herramienta (P-PAP-v1) vs Línea Base (P-BASE-v1) con Wilcoxon + Holm (Paso P9)
    piv_pap = df[df["prompt_id"] == "P-PAP-v1"].set_index("id")
    piv_bas = df[df["prompt_id"] == "P-BASE-v1"].set_index("id")
    common_ids = piv_pap.index.intersection(piv_bas.index)
    piv_pap = piv_pap.loc[common_ids]
    piv_bas = piv_bas.loc[common_ids]

    test_metrics = ["ent_cov_A", "ent_prec_A", "unit_preserved", "bertscore_f1", "rougeL_f1", "flesch_in_range"]
    raw_pvals, comp_rows = [], []

    for m in test_metrics:
        diff = piv_pap[m].astype(float).values - piv_bas[m].astype(float).values
        d_mean, d_lo, d_hi = bootstrap_ci(diff, n_boot=10000, seed=SEED)
        if np.allclose(diff, 0):
            stat_w, p_val = 0.0, 1.0
        else:
            res_w = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            stat_w, p_val = float(res_w.statistic), float(res_w.pvalue)
        raw_pvals.append(p_val)
        comp_rows.append({
            "Metrica": m,
            "Media_Herramienta": round(float(piv_pap[m].astype(float).mean()), 4),
            "Media_LineaBase": round(float(piv_bas[m].astype(float).mean()), 4),
            "Dif_Pareada_Media": round(d_mean, 4),
            "Dif_IC95_Bootstrap": f"[{d_lo:.4f}, {d_hi:.4f}]",
            "Wilcoxon_stat": round(stat_w, 2),
            "p_valor_crudo": p_val
        })

    adj_pvals = holm_bonferroni(raw_pvals)
    for r, p_adj in zip(comp_rows, adj_pvals):
        r["p_valor_Holm"] = round(p_adj, 6)
        r["Significativo_0.05"] = bool(p_adj < 0.05)

    return df_sys, pd.DataFrame(comp_rows)

def export_blind_human_sheet(df: pd.DataFrame, out_csv: str = "results/hoja_revision_humana_40_ciega.csv", key_json: str = "results/llave_revision_humana_40.json"):
    """Paso P6: 40 pares estratificados por generador y nivel, a ciegas y en orden aleatorio."""
    rng = random.Random(SEED)
    pap = df[df["prompt_id"] == "P-PAP-v1"].set_index("id")
    bas = df[df["prompt_id"] == "P-BASE-v1"].set_index("id")
    common = list(pap.index.intersection(bas.index))

    by_strata = {}
    for pid in common:
        g = pap.loc[pid, "generator"]
        l = int(pap.loc[pid, "level"])
        by_strata.setdefault((g, l), []).append(pid)

    chosen_ids = []
    strata_keys = sorted(by_strata.keys())
    for sk in strata_keys:
        pool = by_strata[sk][:]
        rng.shuffle(pool)
        chosen_ids.extend(pool[:4])  # 9 estratos x 4 = 36

    leftovers = [pid for pid in common if pid not in chosen_ids]
    rng.shuffle(leftovers)
    chosen_ids.extend(leftovers[:max(0, 40 - len(chosen_ids))])
    chosen_ids = chosen_ids[:40]

    blind_rows, secret_key = [], []
    for idx, pid in enumerate(chosen_ids, 1):
        # Asignar aleatoriamente cuál es Texto_X y cuál es Texto_Y para cegar al revisor
        flip = rng.choice([True, False])
        sys_x, sys_y = ("P-PAP-v1", "P-BASE-v1") if not flip else ("P-BASE-v1", "P-PAP-v1")
        txt_x = pap.loc[pid, "A_prime"] if sys_x == "P-PAP-v1" else bas.loc[pid, "A_prime"]
        txt_y = pap.loc[pid, "A_prime"] if sys_y == "P-PAP-v1" else bas.loc[pid, "A_prime"]

        blind_rows.append({
            "item_id": f"HUM_{idx:02d}",
            "original_A": pap.loc[pid, "A"],
            "draft_I": pap.loc[pid, "I"],
            "candidate_X": txt_x,
            "X_significado_1a5": "",
            "X_registro_1a5": "",
            "X_fluidez_1a5": "",
            "candidate_Y": txt_y,
            "Y_significado_1a5": "",
            "Y_registro_1a5": "",
            "Y_fluidez_1a5": ""
        })
        secret_key.append({"item_id": f"HUM_{idx:02d}", "pair_id": pid, "generator": pap.loc[pid, "generator"], "level": int(pap.loc[pid, "level"]), "candidate_X": sys_x, "candidate_Y": sys_y})

    pd.DataFrame(blind_rows).to_csv(out_csv, index=False)
    with open(key_json, "w", encoding="utf-8") as f:
        json.dump(secret_key, f, indent=2)
