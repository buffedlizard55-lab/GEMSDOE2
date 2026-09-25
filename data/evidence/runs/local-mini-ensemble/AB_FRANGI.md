# Frangi A/B on the blended ensemble map (fixture, 2026-09-15) - negative result

Same 6-fold maps, same pooled calibration; the ONLY difference is `--frangi`
(`src/postprocess.frangi_enhance`, sigmas 1..6 step 2, weight 0.35) applied to the
blended mean before shaping, with calibration receiving the identical transform on the
fold crops (without that consistency - the first attempt - calibration and map
distributions disagree and the shaped map can collapse to all zeros; that failure mode
is itself worth knowing, and is now guarded by construction).

| arm | pooled held-out DTI | chosen (t0, thin) | final shaped map |
|---|---:|---|---|
| frangi OFF | **0.1286** | (0.46, True) | 7,842 thin px, informational DTI 0.1033 |
| frangi ON | 0.0991 | (0.02, False) = blanket | every valid px, informational DTI 0.0956 ≈ blanket floor |

Frangi at these parameters flattens the selectivity of a low-quality probability field:
the search could not beat blanket coverage, so it retreated to blanket. **Decision: keep
`frangi_filter: false` in the CI ensemble config** (it already is); revisit on better
models, ideally as a *residual* only above a low probability floor, not a global blend.
Reports: `report_frangi2.json` (in /tmp on the sandbox, summarized here) vs
`blend6_report_equal.json`.
