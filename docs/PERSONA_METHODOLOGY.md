# Persona Pack Methodology

*How sidira-personas builds a faithful, company-specific behavioral model of a real userbase — and proves it.*

**Status:** methodology of record. This document supersedes the ad-hoc "seven feature
groups" framing in the README and the demo personas in `samples/`. It is written from
first principles: every company emits a different schema, so the methodology — not the
schema — is what generalizes. Where the current pipeline (`adapter → encoder → cluster
→ calibrate → compile`) already implements a recommendation, that is noted. Where it
does not yet, the gap is called out as a roadmap item (§11). Claims are cited inline to
the primary literature.

---

## 0. The thesis in one paragraph

A Persona Pack is not a slide of fictional archetypes and it is not a clustering report.
It is **the richest faithfully-conditioning representation of a real userbase that the
company's own data can support, accompanied by a falsifiable measurement that it works.**
The product's entire defensibility rests on two facts established by the independent
literature. First, ungrounded synthetic users are a parlor trick: reliable mainly "when
the answer is already known" — an independent review of 12 studies tallied 9 encouraging vs.
14 discouraging findings
([MeasuringU](https://measuringu.com/review-of-experiments-with-synthetic-users/)) — and
standalone synthetic respondents are rated "great" by only ~31% of buyers (Columbia Business
School, via [Solomon Partners](https://solomonpartners.com/insights/reports/synthetic-data-is-transforming-market-research/)).
Second, the *same* technique grounded on a company's own real data crosses into
usefulness — double-blind tests reach ~95% correlation with ground-truth surveys when
the synthetic respondents are trained on the company's real data
([Solomon Partners / EY](https://solomonpartners.com/insights/reports/synthetic-data-is-transforming-market-research/)),
and interview-grounded agents replicate real people's survey answers 85% as well as those
people replicate themselves two weeks later
([Park et al., *Generative Agent Simulations of 1,000 People*, 2024](https://arxiv.org/abs/2411.10109)).
Grounding is the line between the two. Everything below is in service of building on the
grounded side of that line and **measuring** which side we landed on.

The corollary, which reorganizes the rest of this document: the objective function is
**conditioning fidelity, not cluster purity.** A pipeline can maximize BIC and silhouette
and still produce personas that condition an LLM agent into behaving nothing like the real
userbase. We optimize, and gate on, the thing we actually sell.

---

## 1. The two-layer feature ontology (replaces the fixed "seven groups")

### 1.1 Why a fixed schema is the wrong abstraction

The README's seven groups (Demographic, Behavioral, Motivations, Pain Points, Value
Sensitivity, Channel/Tech, Loyalty) are a *content checklist*, not a methodology. They were
chosen by intuition, and the central engineering reality defeats them: a streaming service
logs `Video Played`, a bank logs `Wire Transfer Initiated`, a B2B SaaS logs `Report
Exported`. No fixed column list survives contact with a second customer. Worse, the
demographic group leads with the *least* predictive base: large-sample marketing research
since Haley (1968) finds demographics are "poor predictors of behavior," with "little
support for the existing practice of demographic-based segmentation" and a tendency to
*lower* out-of-sample accuracy
([demographics as predictors, Penn](https://learninganalytics.upenn.edu/ryanbaker/demographic-predictors.pdf)).
Behavioral features substantially out-predict static demographics for behavioral outcomes
(one practitioner estimate puts the lift at ~2.5× —
[Zigpoll](https://www.zigpoll.com/content/how-can-a-data-scientist-help-improve-the-accuracy-of-our-customer-segmentation-models-using-behavioral-data)).

### 1.2 The fix: bind per company, compute company-agnostically

We split the feature system into two layers, mirroring the feature-store / semantic-layer
discipline that Netflix, Uber, and Airbnb built feature stores to enforce
([feature store, Wikipedia](https://en.wikipedia.org/wiki/Feature_store);
[point-in-time correctness](https://towardsdatascience.com/point-in-time-correctness-in-real-time-machine-learning-32770f322fb1/)).

**Layer A — Company-specific binding (the only per-customer artifact).**
1. **Entity resolution** to a canonical `User` (B2C) or `Account` (B2B) grain, exactly as a
   CDP resolves fragmented records via deterministic + probabilistic matching.
2. **Semantic event typing.** Each company event is classified into a small, fixed ontology
   of **semantic verbs** — `value_transaction` · `core_action` (the product's primary verb:
   play, send, query) · `consumption` · `social_action` · `friction_event` ·
   `lifecycle_event`. The industry-standard **Object–Action** event grammar
   (`Order Completed`, `Article Bookmarked` — [Segment naming conventions](https://segment.com/academy/collecting-data/naming-conventions-for-clean-data/))
   makes this tractable: the action token usually maps cleanly to one semantic verb. This
   binding is a YAML mapping config, not new code — the same role `config/*.yaml` plays today.

**Layer B — Company-agnostic semantic feature templates (fixed; the methodology).**
Ten behavioral feature *families* with documented predictive validity for retention, LTV,
and churn, each a parameterized template that instantiates against whatever events carry
the relevant semantic verb. "Frequency of `core_action` over trailing 30d" computes
identically whether `core_action` is `Video Played` or `Wire Transferred`.

| # | Family | Status | Note |
|---|--------|--------|------|
| 1 | **Recency / Frequency / Monetary (RFM)** | **Core** | The most-validated behavioral framework in database marketing; recency carries the highest predictive weight ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1319157818304178)). "Monetary" generalizes to any value proxy (spend, GMV, seats, consumption units). |
| 2 | **Engagement intensity (stickiness)** | **Core** | DAU/MAU; benchmarks vary 3–5× by vertical ([Mixpanel](https://mixpanel.com/blog/mau/)) — itself the argument for within-company normalization (§3). |
| 3 | **Breadth / diversity of usage** | **Core** | Distinct objects touched, entropy of action distribution; separates power users. |
| 4 | **Lifecycle / tenure stage** | **Core** | Days since signup, onboarding completion, cohort — the time-invariant anchor. |
| 5 | Adoption / activation funnel | Modifier | Time-to-first-value, depth into the product graph. |
| 6 | Habit / temporal rhythm | Modifier | Inter-event times, time-of-day/day-of-week regularity; repetition predicts retention better than time spent ([Mixpanel](https://mixpanel.com/blog/mau/)). |
| 7 | Social / network | Modifier | Invites, shares, collaboration edges, referrals. |
| 8 | Churn / health signals | Modifier | Declining-recency trend, downgrades, dormancy, negative deltas. |
| 9 | Value / price sensitivity | Modifier | Plan tier, discount responsiveness, payment friction, refund/dispute rate. |
| 10 | Friction / support | Modifier | Ticket volume, error events, rage-clicks, CSAT. |

The opinionated claim: **RFM + engagement-intensity + breadth + lifecycle is the
irreducible core**; the other six are high-value modifiers, present only when the schema
supports them. Demographics are demoted to *descriptive labels for the resulting personas,
never predictive inputs.*

3. **Coverage manifest.** Layer A records which families could be instantiated for this
   company and which were null — *structurally* missing (the company doesn't track it) vs.
   *behaviorally* missing (the user never did it). This manifest drives the cold-start
   ladder (§6) and the honesty gate (§7, §9): a persona's stated confidence is bounded by
   what the underlying data could support.

**The payoff:** the methodology is constant; only the schema-to-semantics binding varies.
One pipeline builds a credible pack for a streaming service and a bank alike, and a new
vertical needs a mapping config — not a new pipeline.

---

## 2. What data to collect (the intake checklist)

Behavioral signal lives in a small number of recurring systems of record, regardless of
vertical: **event/clickstream** (Amplitude/Mixpanel/Segment — the richest source),
**transactional/billing**, **CRM** (firmographics, deal stage), **support/success**
(tickets, CSAT, NPS), and the **CDP / warehouse / feature store** that unifies them. The
single largest determinant of pack quality is whether the source company tracks events
sanely (Object–Action grammar, a tracking plan, minimal core events with rich properties —
[Segment best practices](https://segment.com/docs/protocols/tracking-plan/best-practices/)).

Per-company onboarding runs this checklist:

1. **Sources present?** Enumerate event, transactional, CRM, support; is there already a
   CDP/warehouse unifying them?
2. **Event taxonomy quality.** Object–Action naming? A tracking plan? Map raw events →
   semantic verbs; flag dynamic/property-encoded event names needing cleanup.
3. **Identity & grain.** Can users be resolved across sources? User- or account-level?
4. **Coverage manifest.** Which of the ten families can be instantiated? Record nulls.
5. **Temporal extent & volume.** Days of history, active-user count → full-fit vs.
   cold-start ladder (§6).
6. **Snapshot definition.** Fix the snapshot date; enforce **strictly backward-looking
   windows**; version it (§3).
7. **Normalization baselines.** Compute the company's own population distributions for
   percentile/z-score normalization.
8. **Representation decision.** Aggregates-only vs. aggregates + sequence embedding (§4).
9. **Confidence / honesty output.** Derive per-persona confidence from coverage + volume;
   degrade to coarser archetypes when thin.

---

## 3. Feature engineering for a point-in-time simulation snapshot

A Persona Pack is a **point-in-time snapshot** used to condition agents. Its temporal
hygiene is non-negotiable.

- **Leakage / point-in-time correctness.** Failing to account for the temporal dimension
  causes *time travel* — information unavailable at the snapshot leaking into features
  ([PIT correctness](https://towardsdatascience.com/point-in-time-correctness-in-real-time-machine-learning-32770f322fb1/)).
  Every feature is computed with strictly backward-looking windows as of the snapshot date.
  This is the entire reason feature stores exist.
- **Normalize within the company's own population**, never against a cross-company constant.
  A "heavy" bank user and a "heavy" social user are incomparable in raw units (DAU/MAU norms
  differ 3–5× — [Mixpanel](https://mixpanel.com/blog/mau/)); percentile/z-score against the
  company's own distribution preserves each persona's *relative* position, which is what the
  agent must reproduce. (The current encoder's per-group PCA already standardizes within
  the population; this formalizes the requirement.)
- **Missingness** is split structural vs. behavioral via the coverage manifest (§1):
  structural absence → excluded from the persona's confidence; behavioral absence → a
  meaningful zero.
- **Stationarity.** A snapshot freezes a non-stationary process. Carry *trend* features
  (deltas/slopes over trailing windows) so the agent reflects trajectory, and version
  snapshots so persona drift is observable.

---

## 4. Representation: aggregates by default, sequences when the product demands it

Two regimes.

**Aggregated per-user feature vectors** (the §1 families collapsed to one row per user) are
the right default: interpretable, cheap, leakage-auditable, and — critically for LLM
conditioning — directly *verbalizable* into a persona prompt. For most companies this
suffices, and it is what the pipeline emits today.

**Sequence / embedding representations** become necessary only when the *order and timing*
of behavior carry the signal the agent must reproduce (recommendation, feed, transactional
flows). The state of the art is transformer-based user-behavior sequence modeling
([BERT4Rec, arXiv 1904.06690](https://arxiv.org/abs/1904.06690)), and for tabular per-user
features specifically, **TabPFN** — a foundation model doing in-context Bayesian
posterior-predictive inference, strong on small data
([Nature, TabPFN](https://www.nature.com/articles/s41586-024-08328-6)) — is worth tracking
for the cold-start regime (§6).

**Decision rule:** ship the aggregated semantic-feature vector as the pack core; layer a
sequence embedding only when (a) the product is sequence-defining and (b) the agent must
reproduce *trajectories*, not just dispositions. Standing caveat: LLM agents conditioned on
persona+session data "often fail to reproduce human-like distributional variance" and are
"highly sensitive to prompt design and temporal drift"
([arXiv 2503.20749](https://arxiv.org/pdf/2503.20749)) — so the *distribution* of features
across the persona population matters as much as any single persona's point estimate.

---

## 5. Segmentation: the right family, the wrong default

### 5.1 Where the current pipeline sits

The pipeline fits a **Gaussian mixture model** (GMM, BIC-selected k) on a concatenated
PCA embedding. GMM is the continuous-Gaussian special case of a **finite mixture model
(FMM)** — the probabilistic, state-of-the-art replacement for ad-hoc cluster analysis in
segmentation, giving within-cluster joint distributions, principled assignment, and a basis
for choosing k ([FlexMix](https://cran.r-project.org/web/packages/flexmix/vignettes/flexmix-intro.pdf)).
So we are in the right *family*. But GMM is the family's weakest member for behavioral data:
it assumes continuous, elliptical Gaussian components (even with `covariance_type="full"`,
which frees each component's shape and orientation, the assumption is still
*Gaussian-elliptical*), while event features are counts, zero-inflated, and heavy-tailed;
and its centroids are *average users who may not exist* — bland, weak conditioning targets.

### 5.2 Three disciplines to add

**(a) Test whether segments even exist — per customer.** Dolnicar, Grün & Leisch
distinguish three data realities: **natural** segments (true clusters exist),
**reproducible** structure (no true clusters but stable structure recoverable), and
**constructive** segmentation (no structure — the analyst constructs a useful-but-artificial
partition) ([Market Segmentation Analysis, ch.2](https://link.springer.com/chapter/10.1007/978-981-10-8818-6_2)).
The test is **bootstrap stability**: re-cluster many resamples and measure agreement with
the **Adjusted Rand Index** (global, chance-corrected) and **per-cluster Jaccard** (Hennig's
`clusterboot`). Acceptance gates from the literature: **mean Jaccard < 0.6 = do not trust;
≥ 0.75 = valid/stable; ≥ 0.85 = highly stable**
([Hennig 2007](https://www.homepages.ucl.ac.uk/~ucakche/papers/clusta.pdf)). Every customer
is classified natural / reproducible / constructive and told **honestly** which they are;
for constructive cases, *k* is a design knob, not a discovery. *(This is open issue #7 —
make it the release gate, not merely a reported metric.)*

**(b) Choose k by BIC tempered by interpretability and stability — never BIC alone**
(it routinely over-splits), and respect **identifiability**: too few informative features
relative to k means the personas are an artifact of EM initialization, not the data
([arXiv 1807.05444](https://arxiv.org/pdf/1807.05444)). Refuse k beyond what the feature
dimensionality supports.

**(c) Move toward mixture-of-memberships representations.** Behavioral users are
multi-motive; hard centroids misrepresent them. Two better matches:
- **Archetypal analysis** finds *extremes on the convex hull* and represents each user as a
  soft mixture of archetypes — vivid, distinctive behavioral poles instead of bland averages,
  which are far better to *simulate* ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231211006060)).
- **LDA-style latent behavioral topics** treat each user as a mixture over latent behaviors
  derived directly from the event stream — schema-agnostic and multi-membership, matching
  transactional data's true generative structure.

Keep GMM/FMM as the baseline; the soft-posterior the pipeline already computes
(`posterior_mean` in the v1.1 pack) is the first step toward soft membership. **Never lead
with demographics** as a clustering base.

### 5.3 Behavioral spine + needs overlay

A behavioral base predicts best but is descriptive — the "what." Add an inferred
**Jobs-To-Be-Done / needs/outcome overlay** so the agent has a *motivation* to rationalize
its next action (Christensen's milkshake; Ulwick's data-driven Outcome-Driven Innovation
makes it reproducible rather than intuition-led —
[Ulwick](https://www.marketingjournal.org/the-limits-of-milkshake-marketing-anthony-ulwick/)).
Behavior alone underdetermines what an agent will do next; the "why" closes the gap.

---

## 6. Synthetic individuals: within-cluster generation with covariance preserved

Step-2 of the consumer (sidira-simulator) draws a *per-agent individual row* so agents vary
*within* a persona. The v1.1 pack ships covariance-aware synthetic rows
(`sample_rows`, MVN + ridge + clip) for this. This section is the rigorous version of that
mechanism and its failure-mode ladder.

**The exact mechanism.** Any *positive-definite* covariance Σ factors as Σ = LLᵀ (Cholesky);
to draw x ~ N(μ, Σ), sample z ~ N(0, I) and set **x = μ + Lz** — covariance preserved exactly
([Gentle, *Computational Statistics*](https://juanitorduz.github.io/multivariate_normal/)).
Cluster covariances are often only positive *semi*-definite or singular (small n, collinear
features), where ordinary Cholesky fails; add diagonal jitter (εI) before factoring — the
bridge to the shrinkage estimators below.

**Three problems the naive MVN draw must address:**

1. **Small-n, ill-conditioned Σ → covariance shrinkage.** The cluster sample covariance is
   untrustworthy when n is small relative to dimension. **Ledoit–Wolf** shrinks S toward a
   structured target with a closed-form optimal intensity, always invertible and
   well-conditioned ([Ledoit & Wolf 2004](https://www.sciencedirect.com/science/article/pii/S0047259X03000964));
   **Schäfer–Strimmer** adapts it to n ≪ p
   ([2005](https://strimmerlab.github.io/publications/journals/shrinkcov2005.pdf)); **OAS**
   lowers MSE further for small Gaussian samples
   ([Chen et al. 2010](https://arxiv.org/abs/0907.4698)). scikit-learn ships `LedoitWolf`
   and `OAS`. The current `ridge·I` is the crudest member of this family — upgrade to
   Ledoit–Wolf/OAS.

2. **Non-Gaussian / mixed-type marginals → Gaussian copula (Sklar).** Behavioral features
   are counts, ratios, flags — not Gaussian. Sklar's theorem factors any joint into marginals
   + a copula, so we model dependence and marginals separately: transform each variable to a
   uniform via its empirical CDF, then to normal via Φ⁻¹, estimate the latent correlation,
   sample N(0,R), invert back through the marginals
   ([Sklar 1959; SDV, Patki 2016](https://www.semanticscholar.org/paper/The-Synthetic-Data-Vault-Patki-Wedge/64ad643e8084486ca7d3312ed491a814d3fe440c)).
   Rank correlations are preserved in expectation (exactly in the population limit; only
   approximately in finite samples, and bounded for tied/discrete marginals — see the caveat
   below). For explicitly mixed continuous/count/categorical
   under a covariance constraint, **NORTA** (Cario–Nelson) and **Hoff's extended rank
   likelihood** ([2007](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-1/issue-1/Extending-the-rank-likelihood-for-semiparametric-copula-estimation/10.1214/07-AOAS107.full))
   are the canonical tools. *Caveat:* with discrete marginals the copula is non-unique and
   attainable correlations are Fréchet-bounded — not every target Σ is exactly reproducible.

3. **Bounds → truncated MVN, not post-hoc clipping.** Naive clipping (winsorizing draws to
   plausible ranges, as v1.1 does) distorts covariance and piles mass at the bounds. The
   principled tool is **truncated MVN sampling** via Botev's minimax exponential tilting,
   which yields i.i.d. bound-respecting draws even in extreme-truncation regimes
   ([Botev 2017](https://academic.oup.com/jrsssb/article/79/1/125/7041924)). Clipping is an
   acceptable v1 shortcut; TMVN is the correct destination when bounds bind hard.

**The small-cluster fallback ladder** (the most important operational rule):
- **Large cluster** (n ≫ dimension) → shrunk cluster covariance (Ledoit–Wolf/OAS).
- **Small cluster** → shrink the cluster covariance *toward the pooled/global covariance*,
  not toward the identity — Ledoit–Wolf with the global Σ as target, generalizing to a
  hierarchical/partial-pooling model ([Hoff 2009](https://arxiv.org/pdf/0804.0031);
  [Hoff, McCormack & Zhang 2023](https://arxiv.org/abs/2302.09211)).
- **Tiny / degenerate cluster** → fall back to marginal-only (independence) or the fully
  pooled covariance, and lower the persona's confidence accordingly (§7).

**How many synthetic rows?** Monte Carlo standard error scales as 1/√m, independent of
dimension, so size m to a target precision — synthetic m should be driven by the variance
wanted in *downstream* estimates and generally should *exceed* real n
([Morris, White & Crowther 2019](https://arxiv.org/pdf/1712.03198)). The v1.1 default of 64
rows is fine for per-agent seeding; raise it when the pack is used for distributional
estimation.

**Why synthetic over real anonymized rows:** MVN/copula draws emit no verbatim real row,
which makes the pack privacy-*friendlier* by construction than shipping real rows (real
anonymized rows stay reserved for benchmarking). This is a reduction of risk, **not** a
guarantee — the field consensus is explicit that synthetic data does *not*
automatically confer privacy ([Stadler et al., "Anonymisation Groundhog Day," USENIX
2022](https://www.usenix.org/system/files/sec22-stadler.pdf)), and cheap heuristics like
Distance-to-Closest-Record are inadequate ([DCR Delusion, ESORICS
2025](https://arxiv.org/abs/2505.01524)); outliers/singletons are the highest-risk records
([Carlini, "Secret Sharer"](https://www.usenix.org/conference/usenixsecurity19/presentation/carlini)).
The honest posture: covariance-aware synthesis is privacy-*friendly*, not privacy-*proven* —
where a formal guarantee is required, use a DP mechanism (PATE-GAN, PrivBayes) and disclose
the ε.

---

## 7. What a Persona Pack must actually carry

Synthesizing the segmentation and LLM-conditioning literature, a persona that *conditions
behavior faithfully* must carry more than a centroid and a variance. Park (2024) showed
narrative, interview-grounded conditioning beats categorical/demographic conditioning *and*
reduces subgroup accuracy bias ([arXiv 2411.10109](https://arxiv.org/abs/2411.10109)). So
each persona ships:

1. **Distributional, not point, description** — the joint distribution *with covariance*
   (for "pattern correspondence," below), not just per-feature variance. *(v1.1: `centroid_full`,
   `feature_scale`, `sample_rows`, `posterior_mean` — extend toward full covariance.)*
2. **Narrative + behavioral exemplars** — representative real (anonymized) event sequences
   and a generated behavioral narrative, because narrative grounding moved fidelity most.
3. **The "why"** — an inferred JTBD/needs/outcome overlay (§5.3).
4. **A confidence / honesty block** — the stability classification (natural/reproducible/
   constructive), per-cluster Jaccard, coverage-manifest completeness, and the synthesis
   tier used (§6 ladder). Clean centroids "inspire non-critical acceptance"
   ([arXiv 2306.14551](https://arxiv.org/pdf/2306.14551)); visible rigor is the antidote and
   a durable differentiator.

The direction of travel: from *"discover clean clusters and describe them"* to *"build the
richest faithfully-conditioning representation the data supports, and prove it."*

---

## 8. Conditioning the LLM agent: algorithmic fidelity

The downstream consumer is an LLM agent, so the governing construct is **algorithmic
fidelity** — the degree to which an LLM, *adequately conditioned*, reproduces the patterns
of a specific human sub-population ([Argyle et al.,
PLOS](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0300024)). Its
four tests are the spec for the pack: a social-science Turing test, **backward continuity**
(outputs consistent with the conditioning input), **forward continuity** (outputs flow
naturally from context), and **pattern correspondence** (the agent reproduces the
*relationships among variables*, not just marginals).

Two hard, well-documented warnings shape what we promise:
- **Believability ≠ accuracy.** On 31,000+ real shopping sessions, prompt-based frontier
  LLMs hit only ~11.9% step-action accuracy; fine-tuning on real click data roughly doubled
  it, and prior evaluations had measured believability to human raters, not match to real
  behavior ([Lu et al., arXiv 2503.20749](https://arxiv.org/pdf/2503.20749)).
- **Base models reflect no specific population** ([Santurkar, "Whose Opinions Do Language
  Models Reflect?"](https://arxiv.org/abs/2303.17548)), and synthetic respondents can match
  overall means while producing wrong subgroup means and shrunken variance ([*Political
  Analysis*, "Perils of LLMs"](https://www.cambridge.org/core/journals/political-analysis/article/synthetic-replacements-for-human-survey-data-the-perils-of-large-language-models/B92267DC26195C7F36E63EA04A47D2FE)).

Implication: we **sell decisions we can validate** — directional pre-screening of product
experiments — and refuse to over-claim point estimates or tail/subgroup behavior. The
honesty *is* the credibility.

---

## 9. The validation protocol (the falsifiable proof)

The pack does not ship on faith. Eight gated stages, publishable as evidence:

| Stage | Test | Metric | Acceptance gate |
|---|---|---|---|
| **0. Data sufficiency** | Sample-size / coverage check | Effective N per segment; coverage manifest | Boost/ship only on robust inputs (weak inputs amplify noise) |
| **1. Segmentation stability** | Bootstrap Jaccard + ARI vs. null | mean Jaccard, ARI, PAC vs. permutation | Jaccard ≥ 0.75 **and** beats a null/permutation model |
| **2. Predictive validity** | Segment → held-out churn/conversion | AUC lift vs. demographic baseline | Significant lift on a held-out split |
| **3. Individual fidelity** | Held-out behavior prediction | **Normalized accuracy** (vs. human test–retest) | Report vs. the human ceiling, never raw % |
| **4. Distributional match** | Sim vs. real behavior logs | KS / Wasserstein / JSD, per-marginal + joint | Within band on marginals *and* correlations |
| **5. Treatment-effect recovery** | Backtest the client's own past A/B tests | ATE correlation (Stanford hit 0.98) | Recover sign **and** magnitude |
| **6. Utility (TSTR)** | Train-on-synthetic, test-on-real | Downstream model generalizes | Real-test performance ≥ threshold |
| **7. Discriminator** | Agent-vs-real classifier | AUC | ≈ 0.5 (indistinguishable) |
| **8. Temporal** | Re-validate next quarter | Segment migration + structure persistence | Structure holds; re-run gates 1–5 |

Two non-negotiables separate this from the field. (1) **Normalized accuracy** against the
human test–retest ceiling as the headline fidelity number — never a vanity raw percentage
(the Stanford ceiling; corroborated by **Twin-2K-500**: 71.7% holdout = 87.7% of the human
ceiling, honestly failing on cognitive biases — [*Marketing Science*
2025](https://pubsonline.informs.org/doi/10.1287/mksc.2025.0262)). (2) **Treatment-effect
recovery backtested against each client's own historical A/B tests** — a falsifiable claim
no generic-persona vendor can make, because they have no client ground truth. Always
benchmark stability against a null model — consensus clustering reports apparent stability
*even on data with no real clusters* ([Şenbabaoğlu 2014](https://www.nature.com/articles/srep06207)).
TSTR caveat: passing every distributional test still does not guarantee downstream validity
— correlations and tails can be wrong
([TDS](https://towardsdatascience.com/your-synthetic-data-passed-every-test-and-still-broke-your-model/)).

The headline claim this licenses: *"On N client deployments, Persona Packs recovered the
direction and magnitude of past A/B treatment effects at correlation X, achieving Y% of
human test–retest accuracy on held-out behavior."*

---

## 10. Fundability: why this is a moat, not a feature

The "synthetic customer" category is hot and unproven — Aaru raised at a ~$1B headline on
<$10M ARR ([Yahoo/TechCrunch](https://finance.yahoo.com/news/sources-ai-synthetic-research-startup-233859223.html)),
while two of the best-known pure-synthetic plays signaled the category's fragility:
**Roundtable pivoted away** from "synthetic humans" to survey-fraud detection
([YC](https://www.ycombinator.com/launches/KKY-roundtable-survey-fraud-and-bot-detection-api)),
and **Viable shut down** in 2025 ([dang.ai](https://dang.ai/tool/ai-qualitative-data-analysis-tool-viable)).
The survivors that win enterprise trust (EY's 95%-correlation double-blind; Fairgen's
statistical augmentation) all **ground on real data** — which is the entire reputational
tailwind for a grounding-first, validation-first entrant.

The moat is **proprietary per-company behavioral grounding plus a closed validation loop**:
every shipped A/B test becomes new backtest ground truth, so accuracy and defensibility
*compound per deployment* — a data-and-validation flywheel generic vendors structurally
cannot copy. The five fundraise claims, each independently sourced:
1. **Grounding closes the gap, measured:** grounded agents beat demographic personas by
   +11–15 pts normalized accuracy (Stanford); standalone synthetic ≈ 31% "great," grounded
   ≈ 95% correlation (EY).
2. **We backtest, they can't:** recover the sign and magnitude of clients' *past* A/B
   results (target ATE correlation ≈ 0.9; Stanford reached 0.98).
3. **Compounding flywheel:** each experiment becomes new ground truth.
4. **Behavioral > demographic for prediction** — the scientific basis of the pack
   (large-sample marketing research; practitioner lift estimates ~2.5×).
5. **Model-agnostic & distribution-aligned:** rides Bedrock/Azure/Anthropic and the AWS
   Agent Marketplace / Agent Skills standard rather than competing with the platforms.

The risk we name rather than hide: distributional realism degrades at tails and subgroups
even when means match — so we sell validated *directional* decisions and report subgroup/tail
behavior honestly.

---

## 11. Roadmap: from today's pipeline to this methodology

| Area | Today | This methodology |
|---|---|---|
| Feature schema | Fixed 7 groups (intuition) | Two-layer ontology: per-company semantic-verb binding + 10 agnostic templates + coverage manifest (§1) |
| Demographics | A feature group | Descriptive labels only; never a predictive base (§1.1) |
| Point-in-time | Implicit | Enforced backward-looking windows; versioned snapshots (§3) |
| Segmentation | GMM + overall ARI | FMM baseline + archetypal/LDA soft membership; BIC tempered by stability + identifiability (§5) |
| Stability | Reported single ARI | **Release gate**: per-cluster Jaccard ≥ 0.75 vs. null; natural/reproducible/constructive label (§5.2, #7) |
| Needs/why | Absent | JTBD/needs overlay per persona (§5.3) |
| Within-cluster draw | MVN + ridge + clip (v1.1) | Ledoit–Wolf/OAS shrinkage → pooled-target small-cluster ladder; copula for marginals; TMVN for bounds (§6) |
| Pack contents | centroid_full, feature_scale, sample_rows, posterior_mean | + covariance, narrative, exemplars, needs, confidence/honesty block (§7) |
| Validation | — | 8-stage gated protocol; normalized accuracy + A/B backtest as headlines (§9) |

**Immediate next steps** (smallest leverage-weighted): (1) promote per-cluster Jaccard +
holdout to a release gate (#7); (2) upgrade the v1.1 ridge to Ledoit–Wolf/OAS with the
pooled-target small-cluster ladder; (3) add the coverage manifest and per-persona confidence
block; (4) stand up stages 1–3 of the validation harness against the first real client whose
A/B history we can backtest.

---

*Every claim above is linked to a source, primary literature wherever one exists (arXiv,
JMR/JRSS-B/Annals/Marketing Science, Nature, USENIX, Stanford HAI, Hennig/fpc); a few market
and landscape figures cite secondary sources (industry docs, analyst notes, press), flagged
as such inline.
The two-repo split is by design: this document is the producer's (`sidira-personas`)
methodology of record; the consumer (`sidira-simulator`) implements §6–§8 against the packs
it emits.*
