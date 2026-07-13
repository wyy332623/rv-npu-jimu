# cosminscn_130p weights

The 130p model uses hand-crafted weights (130 floats), not a trained checkpoint.
All weight values are hardcoded directly in the DRAM layout file:

    adderboard/layout/layout_130p.py

There is no .pt file to store — the weights are computed inline by the
layout builder. All 130 parameter values are deterministic and documented
in `adderboard/docs/compatibility.md`.
