# Run Card

- **Task ID**: 9
- **Timestamp**: 2025-10-04T20:44:50

## Request
- **Prompt**: Vertical load sweep
- **Objective**: Vertical load sweep
- **Params:**
  - tire_size: 315/80 R22.5
  - rim: 9.00x22.5
  - inflation_bar: 8.0
  - mesh_mm: 8.0
  - load_sweep: [0, 10000, 20000, 30000, 40000, 50000]

## Citations
- Template_Guidelines.md
- Prior_Runs.md
- Vertical_Stiffness_Notes.md

## Metrics
- **stiffness_est_N_per_m**: 251437

## Artifacts
- **deck**: `artifacts/deck.txt`
- **log**: `artifacts/solver.log`
- **csv**: `artifacts/result.csv`
- **plot**: `artifacts/curve.png`

## Validation
- **Loads strictly increasing**: True
- **Deflection nonnegative**: True
- **Has citation**: True
- **Samples**: 6
- **Load range (N)**: 0.0 → 50000.0
- **Deflection range (m)**: 0.000218 → 0.198927

### Overall Status: **PASS**
