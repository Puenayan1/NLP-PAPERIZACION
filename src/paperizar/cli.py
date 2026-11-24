import sys, json, argparse
import spacy
from src.paperizar.generate import PaperizerEngine
from src.paperizar.verify import compute_corpus_flesch_bounds, verify_single_rewrite

def main():
    parser = argparse.ArgumentParser(
        prog="python -m paperizar.cli",
        description="Paperización verificable de abstracts de ingeniería en inglés."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Borrador informal en inglés entre comillas.")
    group.add_argument("--input", type=str, help="Ruta a un archivo .txt con el borrador informal.")
    parser.add_argument("--output", type=str, default=None, help="Ruta opcional para guardar el reporte JSON.")
    parser.add_argument("--config", type=str, default="configs/generacion.yaml", help="Configuración de generación.")
    parser.add_argument("--ner-model", type=str, default="models/ner_hybrid_v1", help="Ruta al modelo NER híbrido.")
    args = parser.parse_args()

    draft = args.text.strip() if args.text else open(args.input, encoding="utf-8").read().strip()
    if not draft:
        sys.exit("Error: El texto de entrada está vacío.")

    nlp = spacy.load(args.ner_model)
    flesch_bounds = compute_corpus_flesch_bounds()

    engine = PaperizerEngine(args.config)
    gen_out = engine.generate_batch([draft], ["cli_interactive"], mode="tool")[0]
    engine.close()

    report = verify_single_rewrite(draft, gen_out["A_prime"], nlp, flesch_bounds)
    report["meta"] = {
        "model_id": gen_out["model_id"],
        "prompt_id": gen_out["prompt_id"],
        "prompt_hash": gen_out["prompt_hash"]
    }

    formatted_json = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(formatted_json + "\n")
    print(formatted_json)

if __name__ == "__main__":
    main()
