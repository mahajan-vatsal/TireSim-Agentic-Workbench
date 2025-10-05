# Template Guidelines
- Parameters to fill: `tire_size`, `rim`, `inflation_bar`, `mesh_mm`, `load_sweep`.
- Units:
  - load_sweep: Newtons
  - inflation_bar: bar
  - mesh_mm: millimeters
- Boundary conditions: rim constrained; rigid ground plane contact.
- Solver: static step is sufficient for vertical stiffness (no inertia effects required).
- Keep the deck self-documented with a header block for traceability.
