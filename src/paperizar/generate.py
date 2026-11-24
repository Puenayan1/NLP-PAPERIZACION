import os, re, gc, json, yaml, hashlib
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

def load_gen_config(cfg_path: str = "configs/generacion.yaml") -> Dict:
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def clean_academic_output(raw: str) -> str:
    txt = raw.strip()
    txt = re.sub(
        r'^(?:Here is[^:]*:|Sure[^:]*:|Abstract:|Academic abstract:|Rewritten text:)\s*',
        '',
        txt,
        flags=re.IGNORECASE
    ).strip()
    txt = txt.strip('"').strip()
    return re.sub(r'\s+', ' ', txt)

def derive_pair_seed(pair_id: str, global_seed: int = 42) -> int:
    h = int(hashlib.md5(f"{global_seed}_{pair_id}".encode("utf-8")).hexdigest()[:8], 16)
    return h % (2**31 - 1)

class PaperizerEngine:
    def __init__(self, cfg_path: str = "configs/generacion.yaml"):
        self.cfg = load_gen_config(cfg_path)
        self.model_id = self.cfg["model_id"]

        with open(self.cfg["prompts"]["tool"], encoding="utf-8") as f:
            self.tpl_tool = f.read().strip()
        with open(self.cfg["prompts"]["baseline"], encoding="utf-8") as f:
            self.tpl_base = f.read().strip()

        self.hash_tool = hashlib.sha256(self.tpl_tool.encode("utf-8")).hexdigest()[:12]
        self.hash_base = hashlib.sha256(self.tpl_base.encode("utf-8")).hexdigest()[:12]

        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

    def generate_batch(self, drafts: List[str], pair_ids: List[str], mode: str = "tool") -> List[Dict]:
        tpl = self.tpl_tool if mode == "tool" else self.tpl_base
        prompt_id = "P-PAP-v1" if mode == "tool" else "P-BASE-v1"
        prompt_hash = self.hash_tool if mode == "tool" else self.hash_base

        formatted = []
        seeds = [derive_pair_seed(pid, self.cfg["seed"]) for pid in pair_ids]
        for d in drafts:
            p_txt = tpl.format(draft=d)
            msgs = [{"role": "user", "content": p_txt}]
            formatted.append(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

        inputs = self.tok(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1536
        ).to(self.model.device)

        torch.manual_seed(seeds[0])
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=bool(self.cfg["do_sample"]),
                temperature=float(self.cfg["temperature"]),
                top_p=float(self.cfg["top_p"]),
                top_k=int(self.cfg["top_k"]),
                max_new_tokens=int(self.cfg["max_new_tokens"]),
                repetition_penalty=float(self.cfg["repetition_penalty"]),
                pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.tok.eos_token_id,
                use_cache=True
            )

        in_len = inputs["input_ids"].shape[1]
        outputs = []
        for i in range(len(drafts)):
            gen_ids = out[i][in_len:]
            text_out = clean_academic_output(self.tok.decode(gen_ids, skip_special_tokens=True))
            outputs.append({
                "A_prime": text_out,
                "model_id": self.model_id,
                "prompt_id": prompt_id,
                "prompt_hash": prompt_hash,
                "gen_seed": seeds[i]
            })
        return outputs

    def close(self):
        del self.model, self.tok
        gc.collect()
        torch.cuda.empty_cache()

def select_pairs_for_paperization(pairs: List[Dict]) -> List[Dict]:
    """Selecciona todos los keep=True; si algún arxiv_id de los 100 quedó sin representante,
    agrega su borrador de mayor cobertura marcado con fallback_100_coverage=True."""
    selected = []
    covered_arxiv = set()
    for p in pairs:
        if p.get("keep", False):
            p_copy = dict(p)
            p_copy["fallback_100_coverage"] = False
            selected.append(p_copy)
            covered_arxiv.add(p["arxiv_id"])

    by_arxiv = {}
    for p in pairs:
        by_arxiv.setdefault(p["arxiv_id"], []).append(p)

    for aid, group in by_arxiv.items():
        if aid not in covered_arxiv:
            best_p = max(group, key=lambda x: x.get("filter_stats", {}).get("ent_cov", 0.0))
            p_copy = dict(best_p)
            p_copy["fallback_100_coverage"] = True
            selected.append(p_copy)
    return selected

def run_paperization_pipeline(
    input_records: List[Dict],
    out_jsonl: str,
    batch_size: int = 16,
    engine: PaperizerEngine = None
):
    own_engine = False
    if engine is None:
        engine = PaperizerEngine()
        own_engine = True

    done_keys = set()
    if os.path.exists(out_jsonl):
        with open(out_jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done_keys.add((r["id"], r["prompt_id"]))
    print(f"[{out_jsonl}] Salidas previas detectadas para reanudar: {len(done_keys)}")

    for mode, pid_tag in [("tool", "P-PAP-v1"), ("baseline", "P-BASE-v1")]:
        pending = [r for r in input_records if (r["id"], pid_tag) not in done_keys]
        if not pending:
            print(f"✔ Sistema {pid_tag} ya está 100% completado en {out_jsonl}.")
            continue

        print(f"\nEjecutando {pid_tag} sobre {len(pending)} borradores (batch_size={batch_size})...")
        with open(out_jsonl, "a", encoding="utf-8") as fout:
            for b_start in tqdm(range(0, len(pending), batch_size), desc=pid_tag):
                batch = pending[b_start:b_start + batch_size]
                drafts = [r["I"] for r in batch]
                pids = [f"{r['id']}_{pid_tag}" for r in batch]
                gen_res = engine.generate_batch(drafts, pids, mode=mode)

                for rec, g in zip(batch, gen_res):
                    out_rec = {
                        **rec,
                        "A_prime": g["A_prime"],
                        "model_id": g["model_id"],
                        "prompt_id": g["prompt_id"],
                        "prompt_hash": g["prompt_hash"],
                        "gen_seed": g["gen_seed"]
                    }
                    fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                fout.flush()

    if own_engine:
        engine.close()
