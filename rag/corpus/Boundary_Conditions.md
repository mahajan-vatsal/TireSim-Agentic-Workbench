# Boundary Conditions for Vertical Stiffness
- Constrain all rim nodes: fixed translation in X, Y, Z; rotation constrained as needed.
- Rigid ground plane with contact to tread elements.
- Normal contact with penalty formulation; friction coefficient typically 0.8 for dry asphalt (mock study).
- Loads applied as vertical force sweep on the rim or as imposed displacement with reaction force readout.
- Static step; ensure small increments at low loads for contact stabilization.
