# Guia rápido — Victor / Atividade 2

Este pacote junta o seu repositório da Atividade 1 com a estrutura da Atividade 2.
Os CSVs da Atividade 1 foram mantidos na raiz porque o `src/services/extractors/victor_extractor.py` busca os arquivos diretamente pela URL raw do seu GitHub pessoal.

## Arquivos que precisam ficar na raiz

O `VictorExtractor` espera encontrar estes arquivos na raiz do repositório:

- `J1_curadoria_Victor_FINAL.csv`
- `J1_inferencia_Victor_ANALITICA.csv`
- `J2_curadoria_Victor_FINAL.csv`
- `Resultados_J2_Inferencias.csv`

Neste pacote, o extractor foi ajustado para usar `J1_curadoria_Victor_FINAL.csv` como curadoria do J1 e `J1_inferencia_Victor_ANALITICA.csv` como respostas do J1.

## Antes do push

Na raiz do repositório, rode:

```bash
python scripts/validar_victor_extractor_inputs.py
```

Resultado esperado: todos os arquivos principais devem aparecer com `[OK]`. Pode aparecer um aviso sobre uma questão J2 sem resposta, porque o arquivo `J2_curadoria_Victor_FINAL.csv` tem 119 linhas e `Resultados_J2_Inferencias.csv` tem 118 linhas. Isso não quebra o extractor; ele apenas não cria resposta para a questão sem inferência.

## Sequência para subir ao GitHub

```bash
git status
git add .
git commit -m "feat: adiciona estrutura da Atividade 2 e prepara arquivos do VictorExtractor"
git push origin main
```

## Sequência local da Atividade 2

Com Docker aberto:

```bash
docker compose up -d
uv sync
uv run python main.py db migrate
uv run python main.py db seed all
```

Para não reprocessar tudo depois:

```bash
uv run python main.py db seed export --type all
```

Para executar uma avaliação pequena de teste:

```bash
uv run python main.py db judge evaluate --judge ollama:llama3.1:8b --limit 5
```

Para exportar avaliações:

```bash
uv run python main.py db judge export --all
```

Para rodar a análise estatística:

```bash
uv run python main.py db analysis run
```

## Observação importante

O professor exige a pasta `Atividade_2` com artefatos da entrega, incluindo exports/backup e documentação. Este pacote mantém a pasta `Atividade_2/exports` já usada pelo projeto para os JSONs portáveis.
