# Clinical Eligibility ML

This project implements a **clinical trial eligibility engine**
designed for use in **secondary-care NHS research environments**.

The primary use case is **identifying and prioritising patients**
for potential inclusion in clinical trials, with an initial focus on
**stroke-related studies**.

---

## Background

Clinical trial screening in hospital settings is often:

- time-consuming
- based on incomplete or inconsistently coded data
- constrained by limited manual review capacity

This project explores how **rule-based heuristics** and **classical
machine learning models** can support trial feasibility and screening
workflows without replacing clinical judgement.

---

## Design principles

- Secondary-care focus (hospital admissions)
- Conservative, recall-oriented eligibility logic
- Transparent heuristics and interpretable models
- No assumption of modern DevOps or cloud infrastructure
- Compatible with locked-down NHS Windows environments

---

## Project structure

````

src/
ingestion/        # Load and validate raw hospital data
code_mapping/     # ICD-based clinical code handling
phenotypes/       # Admission-level clinical phenotypes
heuristics/       # Rule-based eligibility logic
features/         # Modelling feature construction
models/           # Baseline ML models
evaluation/       # Ranking and error analysis
config/           # Study-specific configuration

```

---

## Study configuration

Study-specific eligibility criteria are defined using YAML files.

See:

```

src/config/study_config.yaml
src/config/study_schema.md

```

This allows different trials to reuse the same engine with
different eligibility definitions.

---

## Outputs

The engine produces outputs suitable for downstream clinical review:

- CSV tables
- ranked candidate lists
- summary metrics

No automated clinical decisions are made.

---

## Status

The **core eligibility engine is complete**.

Current work focuses on:
- study-level configuration
- usability improvements
- integration into simple browser-based interfaces
```