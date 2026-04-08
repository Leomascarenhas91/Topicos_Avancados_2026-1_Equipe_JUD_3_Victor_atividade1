<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/commons/1/1c/Ufs_principal_positiva-nova.png" alt="ufs-logo" width="20%">

<h1>Tópicos Avançados ES e SI</h1>

<h3>Atividade Avaliativa 1 — Curadoria de Datasets e Inferência Analítica com LLMs Locais</h3>

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/badge.svg)

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="Licença MIT">
  </a>
  <img src="https://img.shields.io/badge/GPU-RTX%202000-green.svg" alt="RTX 2000">
  <img src="https://img.shields.io/badge/RAM-64GB-red.svg" alt="64GB RAM">
</p>

</div>

## Sobre

Repositório individual de **Victor Mascarenhas** para a primeira atividade avaliativa da disciplina **Tópicos Avançados em Engenharia de Software e Sistemas de Informação I** (UFS — 2026.1). O projeto consiste na curadoria de datasets jurídicos e na realização de inferência analítica utilizando Modelos de Linguagem (LLMs) executados localmente, com foco em questões discursivas (J1) e objetivas (J2) do Exame da OAB.

## Onde está a documentação

A documentação completa do projeto e os notebooks de execução estão disponíveis na raiz do repositório. Os resultados consolidados podem ser encontrados em:
- [J1 — Métricas Analíticas](J1_metricas_finais_Victor.csv)
- [Gráficos de Performance](Grafico_Comparativo_OAB_Victor.png)

## Domínio de atuação

Este projeto atua no **Domínio Jurídico**, trabalhando com os seguintes datasets:

| Dataset | Tipo | Quantidade | Fonte |
|---|---|---|---|
| **J1 — OAB Bench** | Questões Abertas | 10 questões (201 a 210) | [maritaca-ai/oab-bench](https://github.com/maritaca-ai/oab-bench) |
| **J2 — OAB Exams** | Múltipla Escolha | 120 questões ( 2092 - 2210)| [eduagarcia/oab_exams](https://huggingface.co/datasets/eduagarcia/oab_exams) |

## Vídeo Demonstrativo

> **Link do vídeo coletivo (Equipe 3):** [A ser adicionado](#)

## Colaborador

<div align="center">
<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/Leomascarenhas91)">
        <img src="https://github.com/Leomascarenhas91.png" height="64" width="64" alt="Victor Mascarenhas"/>
      </a><br/>
      <a href="https://github.com/Leomascarenhas91">Victor Mascarenhas</a>
    </td>
  </tr>
</table>
</div>

---

## 1. Ambiente de execução

### 1.1 Configuração de hardware

Os experimentos de inferência foram executados em máquina local com foco em estabilidade e precisão:

| Componente | Especificação |
|---|---|
| **GPU** | NVIDIA RTX 2000 |
| **VRAM dedicada** | 8 GB |
| **RAM** | 64 GB DDR4 |
| **SO** | Windows 11 |

### 1.2 Modelos de linguagem selecionados

Foram utilizados modelos via [Ollama](https://ollama.com/), selecionados pela diversidade de arquitetura:

| # | Modelo | Desenvolvedor | Comando Ollama |
|---|---|---|---|
| 1 | Llama 3.1 8B | Meta | `ollama pull llama3.1` |
| 2 | Mistral 7B | Mistral AI | `ollama pull mistral` |
| 3 | DeepSeek-R1 8B | DeepSeek | `ollama pull deepseek-r1:8b` |


---


## 2. Instruções de execução

### 2.1 Pré-requisitos

- **Python** 3.11 ou superior
- **Ollama** com os modelos `llama3.1`, `mistral` e `deepseek-r1:8b` instalados
- **pip** para instalação de dependências

### 2.2 Instalação e execução

```bash
# Clonar o repositório
git clone [https://github.com/Leonardomascarenhas91/Topicos_Avancados_2026-1_Equipe_JUD_3_Victor_atividade1.git](https://github.com/Leonardomascarenhas91/Topicos_Avancados_2026-1_Equipe_JUD_3_Victor_atividade1.git)
cd Topicos_Avancados_2026-1_Equipe_JUD_3_Victor_atividade1

# Instalar dependências
pip install pandas requests evaluate rouge_score bert_score sacrebleu matplotlib seaborn scikit-learn

# Executar o notebook de avaliação
# Certifique-se de que o Ollama está rodando
jupyter notebook Atividade1_Victor.ipynb


---


## 3. Mapeamento das questões

### 3.1 Dataset J1 — Questões abertas (`maritaca-ai/oab-bench`)

O dataset J1 contém **210 registros**. As questões designadas para minha análise correspondem a um intervalo de **10 questões abertas** (discursivas), focadas na avaliação de fundamentação jurídica, síntese e análise de casos.

### 3.2 Dataset J2 — Questões objetivas (`eduagarcia/oab_exams`)

O dataset J2 contém **2210 questões objetivas**. As questões designadas para minha análise correspondem ao intervalo de índices **2091 a 2130** (Python, base zero), totalizando **39 questões de múltipla escolha**.

---
