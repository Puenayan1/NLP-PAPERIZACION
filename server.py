"""
AbstractBot · servidor local
============================
Expone el pipeline de `src/paperizar` (NER híbrido + generación + verificación)
como una API HTTP local y sirve la interfaz web de `web/index.html`.

Uso (desde la raíz del repositorio, donde están configs/, models/ y src/):

    pip install fastapi uvicorn
    python server.py                 # abre http://localhost:8000
    python server.py --port 8080 --no-nli

Endpoints
    GET  /                 -> interfaz web (web/index.html)
    GET  /api/health       -> estado del servidor y de los modelos
    GET  /api/examples     -> ejemplos para los botones de la interfaz
    POST /api/ner          -> {text} -> entidades MEASURE / TECH del NER híbrido
    POST /api/paperizar    -> {text, protect, max_retries} -> texto académico + reporte

Todo corre en la máquina local: el texto nunca sale a servicios externos.
La primera vez, Hugging Face descarga el LLM (y el modelo NLI si está activo);
después queda en la caché local.
"""
import argparse, json, os, re, sys, threading, time
from typing import Dict, List, Optional

import spacy
import textstat
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)                      # el código del equipo usa rutas relativas a la raíz
sys.path.insert(0, ROOT)

from src.paperizar.verify import compute_corpus_flesch_bounds, verify_single_rewrite  # noqa: E402
from src.paperizar.informalize import entity_present  # noqa: E402

# ---------------------------------------------------------------- configuración
ARGS = argparse.Namespace(
    host="127.0.0.1", port=8000, config="configs/generacion.yaml",
    ner_model="models/ner_hybrid_v1", nli=True,
    nli_model="cross-encoder/nli-deberta-v3-xsmall", preload=False,
)

THRESHOLDS = {
    "ent_cov": (0.95, 0.85),      # verde >= 0.95, amarillo >= 0.85 (meta del protocolo: 0.95)
    "ent_prec": (0.95, 0.85),
    "semantic": (0.90, 0.80),
    "nli": (0.80, 0.60),
    "copy_rougeL": (0.60, 0.80),  # verde <= 0.60 (no es copia), amarillo <= 0.80
}

# ---------------------------------------------------------------- modelos (carga perezosa)
_lock = threading.Lock()
_state: Dict = {"nlp": None, "engine": None, "engine_error": None, "nli": None, "nli_error": None, "flesch": None}


def repair_spacy_model(path: str) -> None:
    """Google Drive añade '.txt' a los archivos sin extensión al descargar en zip ('cfg' -> 'cfg.txt'),
    y spaCy no encuentra el modelo. Restaura los nombres originales si hace falta."""
    import shutil
    for dirpath, _, files in os.walk(path):
        if "cfg.txt" in files and "cfg" not in files:
            shutil.copyfile(os.path.join(dirpath, "cfg.txt"), os.path.join(dirpath, "cfg"))


def get_nlp():
    if _state["nlp"] is None:
        repair_spacy_model(ARGS.ner_model)
        _state["nlp"] = spacy.load(ARGS.ner_model)
        _state["flesch"] = compute_corpus_flesch_bounds()
    return _state["nlp"]


def get_engine():
    """Carga una sola vez el LLM de la herramienta (Qwen2.5-3B-Instruct según configs/generacion.yaml)."""
    if _state["engine"] is None and _state["engine_error"] is None:
        try:
            from src.paperizar.generate import PaperizerEngine
            _state["engine"] = PaperizerEngine(ARGS.config)
        except Exception as e:  # sin GPU, sin torch, sin descarga…
            _state["engine_error"] = f"{type(e).__name__}: {e}"
    if _state["engine"] is None:
        raise HTTPException(503, f"No se pudo cargar el modelo de generación. {_state['engine_error']}")
    return _state["engine"]


def get_nli():
    if not ARGS.nli:
        return None
    if _state["nli"] is None and _state["nli_error"] is None:
        try:
            from transformers import pipeline
            _state["nli"] = pipeline("text-classification", model=ARGS.nli_model, top_k=None, device_map="auto")
        except Exception as e:
            _state["nli_error"] = f"{type(e).__name__}: {e}"
    return _state["nli"]


# ---------------------------------------------------------------- generación con reintentos
STRICT_SUFFIX = (
    "\n\nThe following terms are protected. Each one MUST appear in your answer exactly as written, "
    "character by character: {terms}"
)


def generate_once(engine, draft: str, protected: List[str], attempt: int) -> Dict:
    """Intento 0: P-PAP-v1 tal cual (igual que la CLI). Reintentos: P-PAP-v1 + lista de términos protegidos."""
    if attempt == 0 or not protected:
        return engine.generate_batch([draft], [f"web_{abs(hash(draft))}"], mode="tool")[0]

    import torch
    from src.paperizar.generate import clean_academic_output, derive_pair_seed
    prompt = engine.tpl_tool.format(draft=draft) + STRICT_SUFFIX.format(terms="; ".join(protected))
    msgs = [{"role": "user", "content": prompt}]
    text = engine.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = engine.tok([text], return_tensors="pt", truncation=True, max_length=1536).to(engine.model.device)
    seed = derive_pair_seed(f"web_{abs(hash(draft))}_r{attempt}", engine.cfg["seed"])
    torch.manual_seed(seed)
    cfg = engine.cfg
    with torch.no_grad():
        out = engine.model.generate(
            **inputs, do_sample=bool(cfg["do_sample"]),
            temperature=max(0.05, float(cfg["temperature"]) - 0.1 * attempt),  # más conservador en cada reintento
            top_p=float(cfg["top_p"]), top_k=int(cfg["top_k"]),
            max_new_tokens=int(cfg["max_new_tokens"]), repetition_penalty=float(cfg["repetition_penalty"]),
            pad_token_id=engine.tok.pad_token_id, eos_token_id=engine.tok.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return {
        "A_prime": clean_academic_output(engine.tok.decode(gen, skip_special_tokens=True)),
        "model_id": engine.model_id, "prompt_id": f"P-PAP-v1+protect(r{attempt})",
        "prompt_hash": engine.hash_tool, "gen_seed": seed,
    }


# ---------------------------------------------------------------- métricas adicionales
def ents_of(doc) -> List[Dict]:
    seen, out = set(), []
    for e in doc.ents:
        t = e.text.rstrip(".")
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append({"text": t, "label": e.label_, "start": e.start_char, "end": e.start_char + len(t)})
    return out


def rouge_l_f(a: str, b: str) -> float:
    x, y = a.lower().split(), b.lower().split()
    if not x or not y:
        return 0.0
    prev = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        cur = [0] * (len(y) + 1)
        for j in range(1, len(y) + 1):
            cur[j] = prev[j - 1] + 1 if x[i - 1] == y[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    p, r = lcs / len(y), lcs / len(x)
    return 0.0 if lcs == 0 else 2 * p * r / (p + r)


def nli_score(premise: str, hypothesis: str) -> Optional[float]:
    clf = get_nli()
    if clf is None:
        return None
    def entail(a, b):
        res = clf({"text": a, "text_pair": b}, truncation=True)
        res = res[0] if isinstance(res, list) and res and isinstance(res[0], list) else res
        return next((r["score"] for r in res if "entail" in r["label"].lower()), 0.0)
    return round(min(entail(premise, hypothesis), entail(hypothesis, premise)), 4)


def status(value, ok, warn, higher_is_better=True):
    if value is None:
        return "skip"
    if higher_is_better:
        return "ok" if value >= ok else "warn" if value >= warn else "bad"
    return "ok" if value <= ok else "warn" if value <= warn else "bad"


def build_report(draft: str, output: str, attempts: int, meta: Dict) -> Dict:
    nlp = get_nlp()
    base = verify_single_rewrite(draft, output, nlp, _state["flesch"])  # reporte oficial del equipo
    doc_in, doc_out = nlp(draft), nlp(output)
    e_in, e_out = ents_of(doc_in), ents_of(doc_out)
    lbl_in = {e["text"]: e["label"] for e in e_in}
    lbl_out = {e["text"]: e["label"] for e in e_out}

    cov = sum(1 for e in e_in if entity_present(e, e_out, output)) / len(e_in) if e_in else 1.0
    prec = sum(1 for e in e_out if entity_present(e, e_in, draft)) / len(e_out) if e_out else 1.0
    semantic = round(float(doc_in.similarity(doc_out)), 4) if doc_in.vector_norm and doc_out.vector_norm else None
    nli = nli_score(draft, output)
    copy = round(rouge_l_f(draft, output), 4)
    fl = base["flesch"]
    p10, p90 = fl["reference_p10_p90"]

    lost_measures = [t for t in base["entities"]["lost"] if lbl_in.get(t) == "MEASURE"]
    units_ok = not base["units_changed"] and not lost_measures
    reg_status = "ok" if fl["output"] < fl["input"] and fl["in_range"] else "warn" if fl["output"] < fl["input"] else "bad"
    layers = [
        {"id": "ner", "name": "Entidades preservadas", "detail": "NER híbrido MEASURE + TECH",
         "value": f"{len(e_in) - len(base['entities']['lost'])}/{len(e_in)}", "status": status(cov, *THRESHOLDS["ent_cov"])},
        {"id": "added", "name": "Nada inventado", "detail": "Precisión de entidades",
         "value": f"{prec:.2f}", "status": status(prec, *THRESHOLDS["ent_prec"])},
        {"id": "units", "name": "Unidades intactas", "detail": "Cifra + unidad idénticas",
         "value": "idénticas" if units_ok else f"{len(base['units_changed']) + len(lost_measures)} con problema",
         "status": "ok" if units_ok else "bad"},
        {"id": "sem", "name": "Similitud semántica", "detail": "Vectores del modelo spaCy (coseno)",
         "value": "—" if semantic is None else f"{semantic:.2f}", "status": status(semantic, *THRESHOLDS["semantic"])},
        {"id": "nli", "name": "Implicación (NLI)", "detail": ARGS.nli_model.split("/")[-1] if ARGS.nli else "desactivado",
         "value": "—" if nli is None else f"{nli:.2f}", "status": status(nli, *THRESHOLDS["nli"])},
        {"id": "reg", "name": "Registro académico", "detail": f"Flesch baja y queda en [{p10:.0f}, {p90:.0f}]",
         "value": f"{fl['input']:.1f} → {fl['output']:.1f}", "status": reg_status},
        {"id": "copy", "name": "No es una copia", "detail": "ROUGE-L entrada vs. salida",
         "value": f"{copy:.2f}", "status": status(copy, *THRESHOLDS["copy_rougeL"], higher_is_better=False)},
    ]
    return {
        "output": output,
        "entities": {
            "input": e_in, "output": e_out,
            "lost": [{"text": t, "label": lbl_in.get(t, "TECH")} for t in base["entities"]["lost"]],
            "added": [{"text": t, "label": lbl_out.get(t, "TECH")} for t in base["entities"]["added"]],
        },
        "units_changed": base["units_changed"],
        "flesch": fl,
        "similarity": {"semantic": semantic, "nli": nli, "rougeL": copy},
        "coverage": round(cov, 4), "precision": round(prec, 4),
        "layers": layers, "attempts": attempts, "warnings": base["warnings"], "meta": meta,
    }


# ---------------------------------------------------------------- API
app = FastAPI(title="AbstractBot", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class TextIn(BaseModel):
    text: str


class PaperizarIn(BaseModel):
    text: str
    mode: str = "paperizar"
    protect: bool = True
    max_retries: int = 2
    model: Optional[str] = None


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "web", "index.html"))


@app.get("/api/health")
def health():
    import yaml
    with open(ARGS.config, encoding="utf-8") as f:
        model_id = yaml.safe_load(f)["model_id"]
    return {
        "status": "ok", "model": model_id.split("/")[-1], "ner_model": ARGS.ner_model,
        "llm_loaded": _state["engine"] is not None, "llm_error": _state["engine_error"],
        "nli": ARGS.nli, "nli_error": _state["nli_error"],
    }


@app.get("/api/examples")
def examples():
    path = os.path.join(ROOT, "web", "examples.json")
    return JSONResponse(json.load(open(path, encoding="utf-8")) if os.path.exists(path) else [])


@app.post("/api/ner")
def ner(body: TextIn):
    return {"entities": ents_of(get_nlp()(body.text[:20000]))}


@app.post("/api/paperizar")
def paperizar(body: PaperizarIn):
    draft = re.sub(r"\s+", " ", body.text).strip()
    if len(draft.split()) < 15:
        raise HTTPException(400, "El texto es muy corto: pega al menos un párrafo.")
    if body.mode != "paperizar":
        raise HTTPException(400, "Esta versión solo admite el modo 'paperizar'.")
    max_retries = max(0, min(int(body.max_retries), 3))

    with _lock:  # un solo modelo en memoria: las peticiones se atienden en orden
        t0 = time.time()
        nlp = get_nlp()
        protected = [e["text"] for e in ents_of(nlp(draft))]
        engine = get_engine()
        best, attempt = None, 0
        while True:
            gen = generate_once(engine, draft, protected if body.protect else [], attempt)
            rep = build_report(draft, gen["A_prime"], attempt + 1,
                               {"model_id": gen["model_id"], "prompt_id": gen["prompt_id"],
                                "prompt_hash": gen["prompt_hash"], "gen_seed": gen.get("gen_seed")})
            score = (rep["coverage"] + rep["precision"]) - (1 if rep["units_changed"] else 0)
            if best is None or score > best[0]:
                best = (score, rep)
            # reintenta si se perdió o se inventó una entidad, o si cambió una unidad
            failed = bool(rep["entities"]["lost"] or rep["entities"]["added"] or rep["units_changed"])
            if not failed or attempt >= max_retries:
                break
            attempt += 1
        rep = best[1]
        rep["attempts"] = attempt + 1
        rep["meta"]["seconds"] = round(time.time() - t0, 2)
        if any(l["status"] == "bad" for l in rep["layers"]) and attempt >= max_retries and max_retries:
            rep["warnings"].append(f"Se agotaron los {max_retries} reintentos: revisa las capas en rojo antes de usar el texto.")
        return rep


def main():
    p = argparse.ArgumentParser(description="Servidor local de AbstractBot")
    p.add_argument("--host", default=ARGS.host)
    p.add_argument("--port", type=int, default=ARGS.port)
    p.add_argument("--config", default=ARGS.config)
    p.add_argument("--ner-model", default=ARGS.ner_model)
    p.add_argument("--no-nli", action="store_true", help="Desactiva la capa NLI (no descarga el modelo NLI).")
    p.add_argument("--nli-model", default=ARGS.nli_model)
    p.add_argument("--preload", action="store_true", help="Carga el LLM al arrancar en vez de en la primera petición.")
    a = p.parse_args()
    ARGS.host, ARGS.port, ARGS.config, ARGS.ner_model = a.host, a.port, a.config, a.ner_model
    ARGS.nli, ARGS.nli_model, ARGS.preload = not a.no_nli, a.nli_model, a.preload

    print("Cargando NER híbrido…"); get_nlp()
    if ARGS.preload:
        print("Cargando LLM (puede tardar)…"); get_engine() if _state["engine_error"] is None else None
    import uvicorn
    print(f"\n  AbstractBot listo en  http://{'localhost' if ARGS.host in ('127.0.0.1', '0.0.0.0') else ARGS.host}:{ARGS.port}\n")
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning")


if __name__ == "__main__":
    main()
