import os, re, gc, json, hashlib
from typing import List, Dict, Tuple
import torch
import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from tqdm.auto import tqdm

SEED = 42

GENERATORS = {
    "G1": "microsoft/Phi-3.5-mini-instruct",
    "G2": "ibm-granite/granite-3.1-2b-instruct",
    "G3": "tiiuae/Falcon3-3B-Instruct",
}

LEVEL_DESCRIPTIONS = {
    1: "Mostly neutral wording; uses 'we' and simple verbs; one or two contractions or loose connectors.",
    2: "First person ('we', 'I'), contractions, vague words ('pretty good', 'a lot', 'kind of'), shorter sentences, phrasal verbs ('ran into', 'came up with').",
    3: "Chatty, spoken tone, run-on sentences, filler words, informal comparisons; still technically correct and still no new facts."
}

ROUGE = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def norm_ws(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip().lower())

def clean_measure(s: str) -> str:
    # Normaliza '95.3 %' -> '95.3%' y quita puntuación colgante para comparar cifras+unidad exactamente
    s = norm_ws(s).rstrip(".,;:")
    s = re.sub(r'(\d)\s+%', r'\1%', s)
    return s

def entity_present(ent: Dict, target_ents: List[Dict], target_text_lower: str) -> bool:
    lbl = ent["label"]
    txt = ent["text"]
    if lbl == "MEASURE":
        c_src = clean_measure(txt)
        c_targets = {clean_measure(e["text"]) for e in target_ents if e["label"] == "MEASURE"}
        if c_src in c_targets:
            return True
        # Respaldo sobre texto crudo normalizado por si el tokenizador partió distinto
        norm_target_raw = re.sub(r'(\d)\s+%', r'\1%', norm_ws(target_text_lower))
        return c_src in norm_target_raw
    else:
        # TECH: coincidencia en minúsculas, espacios normalizados y aceptación por contención (Sección 5-F)
        c_src = norm_ws(txt).rstrip(".,;:")
        if not c_src:
            return True
        for e in target_ents:
            if e["label"] == "TECH":
                c_tgt = norm_ws(e["text"]).rstrip(".,;:")
                if c_src == c_tgt or c_src in c_tgt or (len(c_tgt) >= 3 and c_tgt in c_src):
                    return True
        return c_src in norm_ws(target_text_lower)

def compute_entity_coverage(ents_src: List[Dict], ents_tgt: List[Dict], tgt_text: str) -> float:
    if not ents_src:
        return 1.0
    hits = sum(1 for e in ents_src if entity_present(e, ents_tgt, tgt_text))
    return hits / len(ents_src)

def evaluate_draft_quality(text_a: str, text_i: str, ents_a: List[Dict], ents_i: List[Dict]) -> Tuple[bool, str, Dict]:
    words_a = len(text_a.split())
    words_i = len(text_i.split())
    ratio = words_i / max(words_a, 1)
    cov = compute_entity_coverage(ents_a, ents_i, text_i)
    rl = ROUGE.score(text_a, text_i)['rougeL'].fmeasure

    reasons = []
    if cov < 0.90:
        reasons.append(f"low_entity_cov({cov:.2f}<0.90)")
    if not (0.60 <= ratio <= 1.40):
        reasons.append(f"length_ratio_out({ratio:.2f})")
    if rl > 0.60:
        reasons.append(f"too_close_rougeL({rl:.2f}>0.60)")

    keep = (len(reasons) == 0)
    discard_reason = ";".join(reasons) if reasons else ""
    return keep, discard_reason, {"ent_cov": round(cov, 4), "len_ratio": round(ratio, 4), "rougeL_AI": round(rl, 4)}

def clean_llm_output(raw: str) -> str:
    txt = raw.strip()
    # Remover posibles prefijos conversacionales si el modelo los emite
    txt = re.sub(r'^(?:Here is[^:]*:|Sure[^:]*:|Rewritten abstract:|Draft:)\s*', '', txt, flags=re.IGNORECASE).strip()
    txt = txt.strip('"').strip()
    # Forzar un solo párrafo limpio como pide la regla 4 de P-INF-v1
    txt = re.sub(r'\s+', ' ', txt)
    return txt

def pair_seed(pair_id: str, attempt: int = 0) -> int:
    h = int(hashlib.md5(f"{SEED}_{pair_id}_{attempt}".encode()).hexdigest()[:8], 16)
    return h % (2**31 - 1)

def load_llm(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    return tok, model

def generate_batch(tok, model, prompts: List[str], seeds: List[int], temp: float = 0.8, top_p: float = 0.95, max_new_tokens: int = 450) -> List[str]:
    formatted = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        formatted.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    inputs = tok(formatted, return_tensors="pt", padding=True, truncation=True, max_length=1536).to(model.device)
    torch.manual_seed(seeds[0])
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True
        )
    in_len = inputs["input_ids"].shape[1]
    results = []
    for i in range(len(prompts)):
        gen_tokens = out[i][in_len:]
        results.append(clean_llm_output(tok.decode(gen_tokens, skip_special_tokens=True)))
    return results

def run_informalization(
    input_records: List[Dict],
    out_jsonl: str,
    generators_to_run: List[str],
    levels: List[int] = [1, 2, 3],
    batch_size: int = 8,
    max_attempts: int = 2
):
    nlp = spacy.load("models/ner_hybrid_v1")
    with open("prompts/P-INF-v1.txt", encoding="utf-8") as f:
        prompt_tpl = f.read().strip()
    prompt_hash = hashlib.sha256(prompt_tpl.encode("utf-8")).hexdigest()[:12]

    # Pre-extraer entidades de A una sola vez para acelerar
    ents_a_cache = {}
    for rec in input_records:
        doc_a = nlp(rec["A"])
        ents_a_cache[rec["id"]] = [{"text": e.text.rstrip("."), "label": e.label_} for e in doc_a.ents]

    # Cargar ya procesados para reanudar sin repetir
    done_keys = set()
    if os.path.exists(out_jsonl):
        with open(out_jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done_keys.add((r["arxiv_id"], r["generator"], r["level"]))
    print(f"[{out_jsonl}] Registros previos detectados para reanudar: {len(done_keys)}")

    for gen_key in generators_to_run:
        model_id = GENERATORS[gen_key]
        pending_tasks = []
        for rec in input_records:
            for lvl in levels:
                if (rec["arxiv_id"], gen_key, lvl) not in done_keys:
                    pending_tasks.append((rec, lvl))

        if not pending_tasks:
            print(f"✔ {gen_key} ({model_id}) ya está 100% completado en {out_jsonl}.")
            continue

        print(f"\nCargando {gen_key}: {model_id} ({len(pending_tasks)} borradores pendientes)...")
        tok, model = load_llm(model_id)

        with open(out_jsonl, "a", encoding="utf-8") as fout:
            for b_start in tqdm(range(0, len(pending_tasks), batch_size), desc=f"{gen_key}"):
                batch = pending_tasks[b_start:b_start + batch_size]
                active_indices = list(range(len(batch)))
                best_candidates = [None] * len(batch)

                for attempt in range(max_attempts):
                    if not active_indices:
                        break
                    sub_prompts, sub_seeds = [], []
                    for idx in active_indices:
                        rec, lvl = batch[idx]
                        ents_hint = ", ".join(e["text"] for e in ents_a_cache[rec["id"]])
                        p_text = prompt_tpl.format(
                            level=lvl,
                            level_description=LEVEL_DESCRIPTIONS[lvl],
                            topic=rec.get("topic", rec["category"]),
                            abstract=rec["A"]
                        )
                        # En un 2do intento (si falló el filtro), reforzamos únicamente conservar los tokens exactos y parafrasear más
                        if attempt > 0:
                            p_text += f"\nReminder: Keep these exact technical items verbatim ({ents_hint}) while paraphrasing sentence structure."
                        pid = f"{rec['id']}_{gen_key}_L{lvl}"
                        sub_prompts.append(p_text)
                        sub_seeds.append(pair_seed(pid, attempt))

                    temp_use = 0.8 if attempt == 0 else 0.9
                    drafts = generate_batch(tok, model, sub_prompts, sub_seeds, temp=temp_use, top_p=0.95, max_new_tokens=450)

                    still_failing = []
                    for idx, draft_i, used_seed in zip(active_indices, drafts, sub_seeds):
                        rec, lvl = batch[idx]
                        ents_a = ents_a_cache[rec["id"]]
                        doc_i = nlp(draft_i)
                        ents_i = [{"text": e.text.rstrip("."), "label": e.label_} for e in doc_i.ents]
                        keep, reason, stats = evaluate_draft_quality(rec["A"], draft_i, ents_a, ents_i)

                        cand = {
                            "id": f"{rec['id']}_{gen_key}_L{lvl}",
                            "base_id": rec["id"],
                            "arxiv_id": rec["arxiv_id"],
                            "category": rec["category"],
                            "topic": rec.get("topic", rec["category"]),
                            "published_date": rec["published_date"],
                            "split": rec["split"],
                            "A": rec["A"],
                            "I": draft_i,
                            "generator": gen_key,
                            "generator_model": model_id,
                            "level": lvl,
                            "seed": used_seed,
                            "prompt_id": "P-INF-v1",
                            "prompt_hash": prompt_hash,
                            "entities_A": ents_a,
                            "entities_I": ents_i,
                            "filter_stats": stats,
                            "keep": keep,
                            "discard_reason": reason
                        }
                        # Conservar el mejor intento si aún no había uno o si mejoró cobertura
                        if best_candidates[idx] is None or keep or (stats["ent_cov"] > best_candidates[idx]["filter_stats"]["ent_cov"]):
                            best_candidates[idx] = cand
                        if not keep:
                            still_failing.append(idx)

                    active_indices = still_failing

                for cand in best_candidates:
                    fout.write(json.dumps(cand, ensure_ascii=False) + "\n")
                fout.flush()

        # Liberar VRAM antes de cargar el siguiente LLM en la T4
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()
