# FIELD-selection rule — pre-registered (v1, 2026-09-19)

> **This document is a pre-registration, not an analysis.** It was committed **before** the next
> re-blend so that the field axis of the emission decision is ruled the same way the policy axis
> already is. `data/evidence/emission_decision.json` ranks POLICIES (floor × thinning × width) by
> worst-case contrast across ensembles; `scripts/compare_emission_fields.py` measured the FIELD axis
> at matched support and said, verbatim, that *"no rule in this repository currently selects the
> field … acting on this needs either a rule committed BEFORE the next re-blend, or a scored
> submission."* This rule is that rule.

## Definitions

- **Field** — the per-pixel probability map obtained by averaging the live fold probability maps of
  one or more committed Actions training runs (e.g. `mean12` = runs 35042805806 + 35249562910,
  11 live folds; `ens123` = + 35263581931, 16 live folds).
- **Shipped field** — the field the current shipped submission
  (`data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif`, sha256 `7f00890a…` — bytes re-written 2026-09-25 by the template-conformance sanitation; the blended *field* is unchanged) was blended
  from: **`mean12`**.
- **Shipped support** — the pixel count the shipped submission actually writes: **172,974 px**
  (cross-checked: the adopted-policy row of `eval_sweep-mean12.json` and
  `data/evidence/block_holdout/block_stratified.json` agree — `emission_field_axis.json`).
- **Matched-support candidate of a field** — the hard-band candidate (thin, not a ramp) of that
  field's committed sweep whose emission support is within the support window, with the **same rule
  applied to every field** (identical to `scripts/compare_emission_fields.py`):
  - windows ±15 %, ±25 %, ±40 % of the shipped support;
  - **primary window ±25 %**; best-in-window = highest proxy DTI among hard candidates in the window.
- **Adopted policy** — floor 0.1 / thin / width 0 px (`data/evidence/emission_decision.json`).
- **Proxy population** — the 61,664 px of USGS SGMC trace absent from `labels.tif`
  (`data/evidence/proxy/proxy_catalogue.tif`), the standing surrogate for the scored new faults.

## Eligibility (a field must pass both before any condition is even evaluated)

| id | condition | how it is checked |
|---|---|---|
| **F1** | **Fold completeness.** Every constituent run of the field is a committed Actions run whose fold artifacts are reachable (committed run directory, or Actions artifact still available), and the field's live-fold count meets the `MIN_FOLDS` pinned in the re-blend trigger. | `sweep_source-<field>.json` records the run ids; run directories carry `usable_folds.txt` / `fold_provenance.txt`. The re-blend workflow's minimum-fold gate enforces this at blend time; this rule additionally requires the sweep to have been produced from exactly those artifacts. |
| **F2** | **Committed sweep.** The field has a committed proxy sweep `data/evidence/proxy/eval_sweep-<field>.json` produced by `scripts/eval_proxy_catalogue.py --sweep` on the field's own mean raster, with `reference_t0` recorded in `inputs` (the field's own in-domain calibrated policy). A field without its own sweep has no floor calibrated to its value scale — and a floor is a threshold on the field. | The sweep file exists, has ≥ 1 hard candidates, and `inputs.reference_t0` is present. |

## Adoption conditions (a NEW field F may replace the shipped field S only if ALL hold)

| id | condition | threshold |
|---|---|---|
| **R1** | **Matched-support contrast.** The proxy DTI of F's best-in-window(±25 %) candidate exceeds that of S's best-in-window(±25 %) candidate. | **strictly greater than S + 0.010** — the same margin the policy rule pre-registered (`emission_decision.json` condition 1), because the field comparison inherits the same proxy-population noise. |
| **R2** | **Window stability.** F is ranked **first** at matched support in **all three** windows (±15 %, ±25 %, ±40 %), the same window rule `compare_emission_fields.py` applies to every field. A field that wins only in the widest window is a support-mass effect, not detection. | rank 1 in 3/3 windows. |
| **R3** | **Block-bootstrap support.** Paired block bootstrap over the 51.2 km blocks (`src/blocks.py` partition, same as training) of the per-block DTI difference (F − S) on the proxy population. | **P(F > S) ≥ 0.95** over 2000 paired resamples. R3 is `NOT_MEASURABLE_FROM_COMMITTED_BYTES` unless both fields' matched-support emission rasters are committed; in that state the verdict **cannot** be ADOPT — this is deliberate, not a fallback. |
| **R4** | **Policy transfer.** On F's own sweep, the adopted policy beats F's own in-domain reference policy — the same +0.01 contrast the policy rule required on every ensemble. A field that "wins" because the adopted floor is miscalibrated for its value scale is a floor artifact, not a field gain. | `sweep_verdict` of `eval_sweep-<F>.json`: adopted-policy-row DTI − `current_policy_dti` > 0.010, and the adopted-policy row exists in the sweep. |
| **R5** | **Window sampling.** The primary window of F is actually populated: at least 3 hard candidates lie within ±25 % of the shipped support. A window with 0–1 candidates under-samples the field's support curve. | `n_candidates_in_window[0.25] ≥ 3`. |

**Default.** If F fails **any** of R1–R5 (or either eligibility test), the decision is **KEEP S**.
A field change is a claim and requires positive evidence on every axis; the default is to do
nothing. This is the exact symmetric of the policy rule's "no candidate → keep shipped" behaviour.

## Enforceability

- `scripts/check_field_selection.py` evaluates F1–F2 and R1–R5 from the committed evidence and
  writes `data/evidence/field_selection.json` with the per-condition PASS/FAIL/NOT_MEASURABLE
  status for every committed field and the **derived** verdict (`KEEP <field>` / `ADOPT <field>`).
  The verdict string is computed from the condition table, never typed.
- `.github/workflows/reblend.yml` runs a gate step before blending: if the re-blend is for a field
  **other than the shipped field**, the gate re-runs the checker and **fails the job** unless
  `field_selection.json` says `ADOPT <that field>` with R3 measured. Re-blending the shipped field
  itself (the current 11-fold mean) passes the gate with no measurement, because it changes nothing
  about which field ships.
- Tests in `tests/test_field_selection.py` pin the rule to the committed evidence and fail on
  mutation (weakened margin, changed window set, verdict decoupled from conditions).

## Why these thresholds (provenance)

- **0.010** — the policy rule's pre-registered margin, reused so the two axes cannot be tightened
  or loosened independently by hand.
- **±25 % primary, ±15/±40 flanks** — `compare_emission_fields.py` measured the knife-edge on 2026-09-18
  (at ±15 % the `ens123` field is represented by floor 0.8 / 12 px, at ±25 % by floor 0.1 / 0 px —
  different candidates, different DTI, same field); the three-window rule is the fix it recorded.
- **0.95 bootstrap** — the same standard the block-holdout evaluation applies to emission-policy
  alternatives (the best alternative beats the adopted policy with P = 0.008, i.e. the adopted
  policy holds at 0.992; 0.95 is the acceptance side of the same test).
- **R4 exists because** the field comparison is confounded by the floor: floor 0.1 emits 464,736 px
  on the 6-fold field but 172,974 px on the 11-fold mean. Matching support removes most of the
  confound; R4 removes the rest by requiring the adopted floor to still be the right floor for F.
- **R3 cannot currently be measured for any pair other than (S, S)** because only the shipped field's
  emission raster is committed. That is the intended strictness: the rule makes ADOPT
  unachievable until the measurement exists, which is exactly what "pre-registered" must mean.

## Caveats (inherited, not softened)

- The proxy population is a **surrogate** for the scored truth; every condition inherits that
  limitation (`emission_decision.json` caveats).
- Matched support equalises the **number** of emitted pixels, not their spatial distribution.
- The unconstrained best candidate of a field is a maximum over ~132 candidates on the judging
  population — upward biased, never a shipping recommendation.
