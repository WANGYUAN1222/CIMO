# AUTO-CIMO Replication Package

Replication materials for "Deciding What to Study Next: Design and Evaluation of
an LLM-Based Decision Support Artifact for Research-Gap Discovery" (under review).

## Mapping to the paper

| Paper section | Files |
| --- | --- |
| 3.3 Two-stage extraction prompts and JSON schemas | CIMO_Batch_Extraction_v2_2020_2025.py |
| 3.5 Judge rubric, weighting scheme, human verification | LLM-as-a-Judge.py, Cimo_cohen_kappa.py, kappa_report.txt, kappa_summary.csv |
| 3.6 CIMO-SGD implementation and threshold values | research_gap_cimo.py |
| 3.6 TEXT-RAG baseline | research_gap_rag.py |
| 4.1 Tables 2–3, benchmark fidelity | Evaluate_cimo_v2.py, cimo_full_evaluation_results.csv |
| 4.2 Table 4, LoRA distillation | step1_split_dataset.py, step2_train_lora.py, step3_evaluate_lora.py |
| 5.1 Table 5, gap quality comparison | research_gap_evaluate.py, human_precision_sample_v1.csv, human_recall_sample_v1.csv |
| 5.2 Tables 6–7, forward corroboration and leakage check | research_gap_cimo_2023_2025.py |

## Environment

`requirements.txt`. Other `requirements*.txt` files are historical versions and were
not used for the reported results.

## Data

The source articles are copyrighted journal publications and are not redistributed;
their DOIs are listed in `source_articles.csv`. The candidate-gap portfolios
(CIMO-SGD, n = 230; TEXT-RAG, n = 240) and their de-identified human and LLM
evaluation scores are included.

## Licence

Code: Apache-2.0. Derived data: CC BY 4.0.
