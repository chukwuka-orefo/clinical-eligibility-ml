
# Study Configuration Schema

This document describes the structure and meaning of the
`study_config.yaml` file used for trial eligibility screening.

The purpose of this file is to allow **study-specific eligibility criteria**
to be adjusted **without modifying Python code**.

This reflects how clinical research teams typically work in
secondary-care NHS environments.

---

## Overview

Each study is defined by a single YAML configuration file.

The core eligibility engine (data ingestion, phenotypes, heuristics,
and machine learning models) remains unchanged across studies.

Only the parameters in this file vary.

---

## Top-level structure

```yaml
study:
age:
stroke_signal:
cardiovascular_context:
admission:
ml_scoring:
exclusions:
screening:
````

---

## `study`

Metadata describing the study.

```yaml
study:
  name: "default_stroke_trial"
  description: "Default eligibility configuration for stroke-related trials"
  version: "1.0"
  created: "2015-09-25"
```

These fields are informational and do not affect eligibility logic.

---

## `age`

Age-based inclusion and exclusion rules.

```yaml
age:
  min: 18
  max: 85
  hard_exclude: 90
```

* `min` – minimum eligible age
* `max` – upper bound for standard inclusion
* `hard_exclude` – absolute exclusion threshold

---

## `stroke_signal`

Defines how stroke evidence is interpreted.

```yaml
stroke_signal:
  min_code_count: 1
  require_any_signal: true
  prefer_primary_dx: true
```

* `min_code_count` – minimum number of stroke-related diagnosis codes
* `require_any_signal` – whether at least one stroke signal is required
* `prefer_primary_dx` – preference (not requirement) for primary diagnosis

---

## `cardiovascular_context`

Supporting cardiovascular context.

```yaml
cardiovascular_context:
  min_code_count: 1
  required: false
```

Cardiovascular context is typically **supporting**, not mandatory.

---

## `admission`

Admission-level constraints.

```yaml
admission:
  emergency_only: false
```

If set to `true`, restricts eligibility to emergency admissions only.

---

## `ml_scoring`

Optional machine-learning score thresholds.

```yaml
ml_scoring:
  enabled: false
  min_score: 0.0
```

* If `enabled` is `false`, ML scores are used only for ranking
* If `enabled` is `true`, `min_score` can be used as an additional filter

---

## `exclusions`

Hard exclusion rules.

```yaml
exclusions:
  exclude_without_stroke_signal: true
  exclude_if_age_above_hard_limit: true
```

These rules remove admissions from consideration entirely.

---

## `screening`

Defaults for screening and reporting.

```yaml
screening:
  default_k_values:
    - 25
    - 50
    - 100
    - 200
```

These values represent typical manual screening capacities.

---

## Notes

* This schema is intentionally conservative
* All fields are optional; sensible defaults are applied
* Different studies should use different YAML files
* No clinical decisions should be made solely based on this tool

```

---

# 2. Initial `README.md` (project-level)

Replace or create `README.md` at the **repo root**:

```


