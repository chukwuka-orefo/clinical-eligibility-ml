Clinical Eligibility Tool – Data Inputs

This folder contains input and output data used by the
Clinical Eligibility Tool.

The tool is designed for BATCH screening of patients
for clinical trial eligibility. It is not intended for
single-patient or real-time decision support.

------------------------------------------------------------------
Who prepares the data
------------------------------------------------------------------

Clinical or research staff do NOT prepare data manually.

Structured CSV files are expected to be produced by:
- data analysts
- audit teams
- clinical informatics staff

These files typically originate from:
- hospital data warehouses
- EHR extracts
- audit datasets

------------------------------------------------------------------
Expected input format
------------------------------------------------------------------

The tool operates on structured CSV files representing
secondary-care hospital admissions.

At minimum, the following logical tables are expected:

1. patients.csv
   - subject_id
   - sex
   - date_of_birth (or equivalent)
   - other demographic fields as available

2. admissions.csv
   - subject_id
   - hadm_id
   - admission_time
   - discharge_time
   - admission_type

3. diagnoses.csv
   - subject_id
   - hadm_id
   - diagnosis_code
   - code_system (ICD-9 or ICD-10)

Exact column names may vary by site. Analysts are expected
to align local extracts with the engine’s ingestion scripts.

Clinical staff are not expected to edit or transform these files.

------------------------------------------------------------------
What the tool does automatically
------------------------------------------------------------------

Once data is provided, the tool will:

- derive stroke-related phenotypes
- derive cardiovascular context
- apply conservative eligibility heuristics
- train or load an eligibility model (depending on mode)
- rank patients for manual screening
- write outputs to CSV format

No feature engineering or parameter tuning is required
from end users.

------------------------------------------------------------------
Outputs
------------------------------------------------------------------

Outputs are written to the outputs/ directory and include:

- ranked eligibility lists
- summary counts
- intermediate files as required for review

All outputs are provided in CSV format and are intended
for downstream clinical review.

------------------------------------------------------------------
Important notes
------------------------------------------------------------------

- This tool does NOT make clinical decisions.
- It is intended to support screening and feasibility work.
- All results must be reviewed by qualified clinical staff.
- The tool assumes local execution; no data is sent externally.
