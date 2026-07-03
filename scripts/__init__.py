"""Composition root — the only place concrete classes are constructed and
credentials are read. Lives OUTSIDE src/: it legitimately imports every
module, so it cannot belong to any of them (see PHASE_8_ENTRYPOINTS.md)."""
