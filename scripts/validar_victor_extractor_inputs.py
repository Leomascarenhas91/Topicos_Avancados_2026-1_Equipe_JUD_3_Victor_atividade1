"""
Valida se os arquivos da Atividade 1 do Victor estão prontos para o VictorExtractor.
Execute na raiz do repositório:

    python scripts/validar_victor_extractor_inputs.py
"""
from pathlib import Path
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = {
    "J1_curadoria_Victor_FINAL.csv": {
        "question_id", "Complexidade_Raciocinio_LLM", "Subdominio_Semantico", "Corpus_Aterramento"
    },
    "J1_inferencia_Victor_ANALITICA.csv": {
        "question_id", "Resposta_Llama_3.1_8B", "Resposta_Mistral_7B", "Resposta_DeepSeek_R1_8B"
    },
    "J2_curadoria_Victor_FINAL.csv": {
        "id", "question", "answerKey", "Complexidade_Raciocinio_LLM", "Subdominio_Semantico", "Corpus_Aterramento"
    },
    "Resultados_J2_Inferencias.csv": {
        "id", "Resposta_llama3.1", "Resposta_mistral", "Resposta_deepseek-r1:8b"
    },
}

def read_header_and_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return set(reader.fieldnames or []), rows

def main() -> int:
    ok = True
    cache = {}
    for filename, cols in REQUIRED.items():
        path = ROOT / filename
        if not path.exists():
            print(f"[ERRO] Arquivo ausente: {filename}")
            ok = False
            continue
        header, rows = read_header_and_rows(path)
        cache[filename] = rows
        missing = sorted(cols - header)
        if missing:
            print(f"[ERRO] {filename}: colunas ausentes: {', '.join(missing)}")
            ok = False
        else:
            print(f"[OK] {filename}: {len(rows)} linhas e colunas esperadas encontradas.")

    if "J2_curadoria_Victor_FINAL.csv" in cache and "Resultados_J2_Inferencias.csv" in cache:
        cur_ids = {r.get("id") for r in cache["J2_curadoria_Victor_FINAL.csv"]}
        ans_ids = {r.get("id") for r in cache["Resultados_J2_Inferencias.csv"]}
        missing_answers = sorted(cur_ids - ans_ids)
        if missing_answers:
            print(f"[AVISO] Há {len(missing_answers)} questão(ões) de J2 sem resposta de inferência: {missing_answers[:10]}")
            print("        O extractor vai importar a pergunta, mas não criará resposta para ela.")
        else:
            print("[OK] Todas as questões J2 possuem respostas de inferência.")

    print("\nResultado:", "PRONTO para o VictorExtractor." if ok else "AJUSTAR os erros acima antes do push.")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
