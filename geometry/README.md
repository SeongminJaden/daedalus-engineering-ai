# geometry

Geometry representations and CAD output.

| module | role |
|---|---|
| `implicit.py`, `topology.py`, `mesh.py`, `cad.py` | stubs for later phases |
| `cad_export/` | parametric B-rep and STEP output (Phase 9) |

## CAD export is an optional dependency

The analysis stack does not need a CAD kernel, and the OpenCascade build behind
one is several hundred megabytes. A user who only runs simulations should not
have to download it.

```bash
env -u PYTHONPATH .venv/bin/pip install -r requirements-cad.txt
```

`build123d` is preferred, `cadquery` is the fallback; both wrap the same OCCT
kernel. With neither installed, the CAD tests skip and `dae export` prints an
install hint instead of a traceback.

## The exported part must be the analysed part

A STEP file whose geometry disagrees with what the physics integrated would
mean handing a manufacturer a part nobody simulated. So `export_step` checks
before it writes:

1. B-rep volume against the analytic cross-section times length.
2. That volume times material density against the mass the analysis used.

A mismatch raises and **no file is written**. Measured on the Phase 7.5 optimum,
the volume agrees to 6.0e-16 relative. The mass residual is ~2.4e-07, and that
number comes from the **fp32 physics kernel**, not the CAD: the B-rep volume is
float64 and exact for a box, so a tolerance tighter than fp32 would be measuring
the wrong thing.

## Where the "always STEP" promise holds, and where it does not

**Parametric solids: guaranteed.** A hollow box is a handful of planar faces.
Its B-rep is exact and the STEP file is clean.

**Organic and topology-optimized shapes: not guaranteed.** Those arrive as a
density field or an implicit surface, not as faces and edges. Recovering a clean
B-rep needs surface reconstruction and refitting, and done carelessly it yields
either a NURBS patchwork no downstream CAD system accepts or thousands of facets
pretending to be a solid. `mesh_from_density_field` is therefore a deliberate
`NotImplementedError` rather than something that emits geometry which looks
manufacturable and is not.

`export_stl` tessellates an existing B-rep, which is useful for visualisation
and printing. It is an approximation with a stated tolerance and is not a
substitute for STEP where dimensions matter.

## This is analysis geometry, not a manufacturing-ready part

The exported solid has no fillets, no fastener features and no tolerances. The
sharp root corner is exactly where Phase 7 located the stress concentration, and
a real part would need a fillet there. Design-for-manufacture features are a
later phase, and multi-part assembly STEP is Phase 10.
