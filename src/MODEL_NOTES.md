# Model notes

Everything here used to live in code comments. It does not belong there.

## Comment convention

Code comments are **construction comments only**: what an array holds, what
shape it is, what units it is in, cell ordering, and non-obvious invariants a
reader could otherwise break. Brief.

Not in code:

- **performance and measurement** — timings, node-count convergence, measured
  sensitivities, "this is X% faster", "measured at Y EUR/capita";
- **modelling content** — justification of a functional form, why a parameter
  takes its value, literature, interpretation. That belongs in `main.tex`;
- **changelog** — what a line used to be, what it replaced, what was previously
  missing. That is what git is for.

Anything measured or argued goes here or in the paper.

## Parameter provenance (HEALTH_MODEL 9)

| Parameter | Status |
|---|---|
| `MU_C_MEAN` | **calibrated**, one moment (uptake 0.35 at zero incentive, age-banded FIT arm) |
| `STAGE_TARIFF` (+SE) | Corral et al. (2016), BMC Health Serv Res 16:56, Table 3: 16-year direct medical cost per patient, discounted at 3%, 2005 € — stage I 31,757 (27,172–36,731), II 41,116 (38,313–43,979), III 47,681 (43,439–52,551), IV 28,061 (26,533–29,694); ×1.45 Spanish CPI to 2025; SE = CI width / 3.92. Independent Gammas. Stage IV is cheapest (short survival), so at the mean stage mixes a screen-detected cancer (≈54,900 €) costs about the same as a clinically detected one (≈54,700 €): early detection saves no treatment money in expectation. Missed cancers are clinically detected (same mix and lag). Transportability not represented |
| `STAGE_DISUTILITY` (+YEARS) | Djalalov et al. (2014), Med Decis Making 34:809, meta-analysis of 351 CRC utilities: utilities at 3 months after surgery are 0.05 below those beyond 1 year, and stage IV utilities 0.19 below stages I–III. Hence −0.05 for one year for stages I–III and −0.24 until death (T_DEATH) for stage IV, applied to both survivors and deaths of that stage. Absolute decrements, not scaled by EQ5D(age); not lagged. Borne by both agents: PM at λ_s, citizen at λ_C and β-weighted |
| `STAGE_*` | Stage mix at diagnosis at ages 60–69 (screen-detected 43.0, 22.6, 26.8, 7.6%; other 18.6, 25.0, 29.1, 27.3%) and 5-yr overall survival by stage of non-screen-detected CRC (I–IV 86.7, 79.2, 66.2, 13.9%), Cardoso et al. 2022, Lancet Reg Health Eur 21:100458. Stage mortality m_k = 1 − OS5 / 0.95; 0.95 = 5-yr other-cause survival at 60–69 (INE: annual death risk ~0.52% at 60, conditional survival 60→70 of 0.928; 0.96 would move q_screen to 0.229, dq to 0.156). Derived means q_screen 0.221, q_clinical 0.379, dq 0.158. Both mixes Dirichlet with α₀ = 125, which gives sd(dq) ≈ 0.032, the sd of the former LogNormal(mean 0.16, log-sd 0.20) that spanned country ranges of stage share; mortality by stage fixed. Applied to all ages |
| `T_DEATH` | 2.5 yr from diagnosis to death from the cancer: the **midpoint of the five-year window** over which q is measured, which is the only value consistent with reading q off five-year survival. Missed cancers take q_clinical |
| false alarms | no separate parameter. The episode is counted once: the citizen bears the follow-up burden κ_col·B, the PM the colonoscopy cost and L_COL. An extra decrement (Matza et al. 2024, cTTO, 0.079 QALY; Gyrd-Hansen & Søgaard 2001, DCE, no effect) would partly count the same event twice, and at 0.079 it would swamp the benefits, as false positives outnumber true positives ~18:1 |
| `L_COL_RANGE` | point value 4.34e-4 QALY: procedure-related mortality 2.9/100,000 (Reumkens et al. 2016, Am J Gastroenterol 111:1092, pooled population-based studies) × discounted quality-adjusted life expectancy at 60–69, 0.884·A(24, 0.03) = 14.97 QALY. Perforation (0.5/1,000) and bleeding (2.6/1,000) are short-lived and omitted: at any plausible short-term decrement they add under a tenth of the mortality term |
| `V_QALY` | 30,000, as in the paper; published range ~22,000–30,000, explored by the v sweep |
| `LAMBDA_C` | 0.05, common to all citizens, deliberately above λ_s = 0.03. Preserves the old lognormal's mean perceived gain at ages 60–69 to ~2% (3-point quadrature, approximate) |
| `B_THETA` | 5, stated dispersion of γ ~ Beta(5, 5) |
| `BETA_MIN` | 0.36, so E[β] = 0.68: meta-analytic β for non-monetary rewards after correcting for selective reporting, 95% CI [0.57, 0.82] (Cheung, Tymula & Wang, Management Science, doi:10.1287/mnsc.2023.04003; check the published table) |
| `F_MIN` | 0.30, stated; E[p_i / p_crc] = 0.65 |
| `_COMFORT_SCALE` (κ) | stated; only κ_col = 10 matters for a FIT-only arm. A DCE on test attributes is the right source |
| `SIGMA_C_LOG` | mean/median ratio in Jonas et al. (WTP to avoid colonoscopy) |
| `ADHERENCE` | 1.0, a simplification |

Age enters the citizen's problem only through T(age) and EQ5D(age). Under
HEALTH_MODEL 3 the age means of γ reproduced the horizon discount at λ = 0.05
almost exactly (β·E[ρ|β] at the γ means, relative to age 60: 0.16, 0.27, 0.41,
0.62, 1, against (1.05)^−(60−age): 0.14, 0.23, 0.38, 0.61, 1), so age was
counted twice.

## Unified parameter set (default)

The default since HEALTH_MODEL 9; where it differs from the provenance table above,
this table holds. `CRC_PARAMS=current` restores the previous set. Built on one reference case (Spanish NHS
perspective, 2024 EUR, 3% discounting) with each module from one source, following
standard HTA practice (López Bastida et al. 2010; Briggs et al. 2006).

| Quantity | Unified value | Source / rule |
|---|---|---|
| Price index | HICP health (CP06), Spain: 92.91 (2005), 93.31 (2012), 108.55 (2024); `CRC_PRICE_INDEX=cpi` uses all items (83.33, 99.31, 123.33). CP06 measures prices households pay for health goods and services, not the cost to the NHS of producing care (mostly wages), so it is a weak inflator here; the all-items CPI is the common alternative in Spanish evaluations. The choice moves the incentive gain by about 0.3 EUR | Eurostat `prc_hicp_aind` |
| `C_COMM` | 6.06 (2012) → 7.05: invitation, FIT kit and programme management, per invited citizen. The Basque programme mails the kit with the invitation, so its cost does not depend on uptake; where the kit is collected (pharmacy, health centre) it is per screener instead, which lowers the incentive gain by about 1 EUR | Arrospide et al. 2018, BMC Cancer 18:464 (Basque programme accounts) |
| FIT cost (per screener) | 0.99 (2012) → 1.15: analysis | Arrospide et al. 2018 |
| Follow-up colonoscopy | 78 consultation + ½(281.30 + 461.30) with/without polypectomy + 3.1‰ × 5,157 complications = 465 (2012) → 541. **The 50% polypectomy share is an assumption** | Arrospide et al. 2018; complication rate Reumkens et al. 2016 |
| Treatment tariffs | Corral et al. 2005 € × health index (×1.168), not CPI ×1.45 | Corral et al. 2016 |
| Life expectancy Ȳ(a) | INE 2024 remaining life expectancy at band midpoints: 59.5, 49.7, 40.0, 30.65, 21.9 (ages 25–65; 65 tabulated, others interpolated) | Eurostat `demo_mlexpec` |
| Other-cause survival S0 | 0.9528 = l70/l65, matching Cardoso's 60–69 patients | Eurostat `demo_mlifetable` 2024 |
| L_COL | 2.9e-5 × 0.884 × A(21.9, 0.03) = 4.07e-4 | Reumkens et al. 2016 with the life table above |
| Stage mixes | Pooled shares (from counts: screen-detected 8,380 / 4,392 / 5,221 / 1,476, other 10,531 / 14,130 / 16,460 / 15,422; ages 60–69, known stage), Dirichlet with α₀ = 125 for between-country heterogeneity, sd(dq) ≈ 0.035 (see Sensitivity scenarios). `CRC_STAGE_DIRICHLET=counts` gives Dirichlet(counts + 1), sampling uncertainty only | Cardoso et al. 2022 |
| QoL loss | Djalalov decrements as fractions of the reference utility (0.05/0.90, 0.24/0.90), multiplied by EQ5D(age) | Djalalov et al. 2014; García et al. 2016 |

Unchanged: v, λ_s, test accuracy (CRC-only; adenoma costs and benefits excluded),
citizen preferences and calibration target, reminder effect.

## Sensitivity scenarios (public scheme)

`public_incentive_scheme.py --scenario=...` overrides constants before calibrating;
outputs go to `_<scenario>` folders and `calibration.json` is not written.

| Scenario | Override | Note |
|---|---|---|
| `uptake25`, `uptake45` | calibration target 0.25, 0.45 | |
| `lambda03`, `lambda07` | `LAMBDA_C = 0.03`, `0.07` | |
| `stage_counts` | stage mixes Dirichlet(counts + 1) | sampling uncertainty only, sd(dq) ≈ 0.002 |
| `stage_a50` | stage mixes with concentration α₀ = 50 (base 125) | wider between-country heterogeneity, sd(dq) ≈ 0.05 (see below) |
| `stage_low` | screen-detected I–IV 35.7/23.7/28.1/12.5%, other 24.9/24.3/28.3/22.5% | least favourable ends of the between-country ranges; dq ≈ 0.077 |
| `stage_high` | screen-detected 52.7/19.0/22.5/5.8%, other 13.2/25.4/29.5/31.9% | most favourable ends; dq ≈ 0.22 |
| `stage_basque` | screen-detected 1,376/408/566/152 (known stage) | Basque programme, ages 50–69, 2009–2015; dq ≈ 0.18 |
| `cpi` | all-items HICP instead of health HICP (set before `cu` is imported) | Eurostat |
| information rows | `--pm_covariates=observed` (`_pmobs`), `--citizen_covariates=all` (`_citall`) | see Information structure |

Stage sources. Cardoso et al. 2022, Lancet Gastroenterol Hepatol 7:711 (nine
countries): stage I 35.7–52.7% of screen-detected vs 13.2–24.9% of other cancers,
stage IV 5.8–12.5% vs 22.5–31.9%. In `stage_low`/`stage_high` stages I and IV are set
to the ends of these ranges and II–III share the remainder pro rata to the pooled mix.
Taking the nine-country range as about 3 standard deviations, the Dirichlet variance
p(1−p)/(α₀+1) gives α₀ between roughly 75 and 200 across the stage I and IV shares, so
α₀ = 125 represents between-country heterogeneity. Portillo et al. 2017, World J
Gastroenterol 23:2731 (Basque programme): screen-detected stage I–IV 54.6/16.2/22.5/6.0%,
FIT-negative interval cancers 23.1/19.4/30.6/26.9% (n = 186), close to the pooled
non-screen-detected mix used for clinically detected and missed cancers.

The v sweep (`sensitivity_analysis.py`) re-values the base runs for the PM only,
holding uptake fixed, using `pm_decomposition.npz` (per-capita health at V_QALY and
outlays): value = (v / V_QALY) · health − outlays, exact for the risk-neutral u_PM.

## Information structure (base case)

`screening_assignment.py --risk_cells_all` writes `models/risk_cells_all.csv`: cells of
the seven observed covariates (PA, SD, Smoking, BMI, Alcohol, Sex, Age) and the
comorbidities that are parents of CRC (Diabetes, Hypertension, Hyperchol_), with
p_crc = p(CRC | all ten) and p_crc_obs = p(CRC | seven). Depression, Anxiety and SES
are d-separated from CRC given its parents, so they would not change p_crc. Every run
uses these cells.

| Agent | Risk | Flag (base) |
|---|---|---|
| Nature (outcomes) | p_crc | — |
| PM, and SP in the OBP | p_crc: comorbidities from health records | `--pm_covariates=all` |
| Citizen | p_crc_obs: does not map comorbidities to CRC risk | `--citizen_covariates=observed` |

The risk policy invites the age arm's volume in descending PM risk, splitting the
boundary cells pro rata. Uptake depends on the citizen's risk only, so it is constant
within an observed cell; p_crc_obs is the population mean of p_crc within its cell
(ratio 0.996), so the age arm and μ_B match those computed on the seven-covariate
cells. Non-base flags add `_pmobs` / `_citall` to output folders and recalibrate μ_B
without saving it. Both risks overpredict 2016 incidence equally (O/E 0.84).

## Uptake check

`plot_p_scr_K.py [patient] [--screening_policy=...]` decides invitation and the
citizen's risk as `optimal_incentive_patient.py` does (default FIT risk-based; patients
specify all ten covariates in `patients.py`), repeats the ARA estimate
of π_PM(s = 1 | I, x) 100 times with 200 type draws at I = 0, 10, 20, 30, 50 and
100 EUR, and saves the histograms to `outputs/p_scr_K/`. The centre of each histogram
traces the uptake curve; its spread is the Monte Carlo error of one estimate at 200
draws, which shrinks as 1/√N_ARA. It is not uncertainty about uptake, which does not
depend on θ.

## Measured sensitivity (HEALTH_MODEL 3, not re-measured)

Uptake ceiling (burden driven to the bracket floor, `c_mean = 1`), FIT
old-adult profile, with `L_FP` upper at 0.079:

| parameter | low → ceiling | high → ceiling |
|---|---|---|
| `L_FP` upper | 0.001 → **0.946** | 0.079 → **0.232** |
| λ median | 0.03 → **0.372** | 0.10 → **0.079** |
| `G_MEAN` | 2.7 → **0.069** | 4.0 → **0.232** |
| `F_MIN` | 0.1 → 0.180 | 0.9 → 0.388 |
| `V_QALY` | 25000 → 0.229 | 30000 → 0.232 |
| `SIGMA_C_LOG` | 0.40 → 0.233 | 1.20 → 0.231 |
| κ colonoscopy | 3 → 0.235 | 10 → 0.232 |
| `L_COL` upper | 2e-4 → 0.235 | 1e-3 → 0.232 |

## OBP scheme

Aligned with the public scheme so that the comparison isolates delegation: the same
profiles, type draws (`pis._reservations`), incentive axis, θ draws, population
replicates and status quo; the public row of the OBP table is simulated on the same
replicates. The SP has no instrument the PM lacks except its reading of ζ: one
invitation per citizen in the base case, the whole target population approached. The
approach threshold was removed.

Reminders (`--contacts=K`, K_GRID = (1..K)): assumption that the PM's communication
budget allows one invitation while the SP may send reminders at its own cost. A
reminder goes only to citizens who have not screened after the previous contact
(cost c_comm per contact actually sent) and raises their misperception floor,
f(k) = 1 − (1 − f_min)ρ^(k−1). Acceptance is nested in k, since ζ falls as f rises.
ρ is calibrated at zero incentive on the age-based calibration arm so that one
reminder multiplies uptake by `REMINDER_RR` = 1.33, the pooled RR of postal reminders
in organised CRC screening (Camilloni et al. 2013, BMC Public Health 13:464; 95% CI
1.17–1.51). Base case K = 1; K = 2 is the sensitivity analysis, with the RR at the CI
bounds as further rows.

Grids: SP budgets up to 400 EUR (the public axis stops at 150) and z4 up to 10; under
the information structure the base optimum reached z4 = 3 and a 150 EUR budget, the
earlier grid edges. `obp_scheme.py` warns when z* or the budget sits on an edge (z1 = 1
and z2 just below z1 are feasibility limits, not grid artefacts).

Comparisons (`scheme_comparison.py`) are against the age-based status quo for both
policies: the risk policy's values are shifted by its own status quo minus the age one,
state by state. OBP sensitivity (`obp_scheme.py` flags, risk policy): `--sigma_nu` 10 and
40 EUR (half and double), `--ccomm_scale` 0.5 and 2, `--ccomm_sigma` 0.7 (double), and
`--reminder_rr` 1.17 and 1.51 with `--contacts=2`; gathered in `obp_scenarios.csv`.

`SIGMA_NU` = 20 EUR: the AUC 0.871 for predicting participation in a national
cancer screening programme (Kim 2021; Chun et al. 2018, 0.82) mapped through the
model's pooled ζ distribution, for the event ζ ≤ 0 (screening unpaid). Profile alone
gives AUC 0.675, so the signal carries the increment. Derived under an earlier
calibration; re-derive if the within-profile spread of ζ moves materially.

With one contact, `c_comm` enters the SP's value as a constant c_comm·N, so it shifts
acceptance of the contract, not the budget chosen; with reminders it also shifts the
number of contacts, as their cost depends on how many citizens remain unscreened.

## Secondary code paths

Scripts no longer used by the pipeline, and functions with no callers
(`expected_utilities_cit`, `draw_block`, `score_block`, `summarise`,
`u_pm_patient`), are in `stale/old/`; they are kept for reference and may not run.
`plot_p_scr_K.py` stays in `src/` as the uptake check above.

Population replicates are drawn once (`draw_population`) and shared by every
arm (`simulate_arm`), so arms are paired citizen by citizen: each cancer citizen
keeps their acceptance, test-result and death uniforms.

## HEALTH_MODEL versions

Guards a stored calibration against a change in the health valuation's shape.

1. Losses discounted at a horizon-average factor.
2. `health = eq5d * v * A(T - losses, λ)`; `L_FP` a flat `-v * L_FP` inside the
   β-weighted block.
3. `L_FP` moved to the present period (not β-weighted); `L_COL` added, charged
   per colonoscopy and β-weighted; the citizen integrates the marginal θ by
   quadrature instead of plugging in `E_THETA`.
4. Citizen type (γ, B) independent of age; β = β_min + (1 − β_min)γ; perceived
   risk deterministic in γ; common λ_C; L_TP and L_COL point values; no citizen
   multipliers; L_FP = 0.
5. L_COL charged to the citizen at the procedure (present period, not
   β-weighted), as the PM already does.
6. Cancer losses as a death lottery: death at T_DEATH with q = X / (T_REF −
   T_DEATH), survival to T(age) otherwise; replaces horizon truncation and the
   citizen multipliers. Missed cancers treated at T_TREAT_FN, like unscreened.
7. Death probabilities set directly: q_screen for screen-detected cancers,
   q_screen + dq for clinically detected and missed ones.
8. L_FP removed from θ and from both agents' values; the false-alarm episode is
   counted once, through κ_col·B (citizen) and the colonoscopy cost plus L_COL
   (PM). θ now consumes one fewer variate per state of nature, so every analysis
   must be rerun together.

θ redesign (not a HEALTH_MODEL change, as w_C keeps its shape): θ's primitives
are the two stage mixes, the tariffs and L_COL; q_screen, dq, early_screen and
early_clinical are derived from them. Treatment tariffs follow the stage mix
instead of assuming screen-detected = stage I and clinical = stage III. Q_SCREEN
and DQ_MEAN moved from 0.22/0.16 to 0.221/0.158, so the stored calibration is
stale, and θ draws changed, so every analysis must be rerun together.

9. Quality-of-life loss from cancer treatment by stage (`STAGE_DISUTILITY`), in
   the future period for the citizen. Tariffs for all four stages replace the
   stage I / stage III pair, and L_COL is sourced. θ now carries
   (tau_screen, tau_clinical, qloss_screen, qloss_clinical) instead of the early
   shares; draws changed again, so every analysis must be rerun together.
