"""Pure library entry point for the persona pipeline.

``run_pipeline`` is the in-process API the rest of the world (the CLI, the
simulator backend's persona-build worker) drives. It owns the full
Adapter → Encoder → Cluster → Calibrate → Compile flow and returns the
artifacts in memory; it does NO source reading and NO file writing, so callers
control both ends (the CLI reads Snowflake/local + writes a run dir; the
simulator hands in a DataFrame from its own query + writes the pack to R2).

Behaviour is identical to the original ``cli.run`` body — that monolith was
split here and the CLI now reads the source, calls this, and writes the run
dir. The end-to-end smoke test (``tests/test_pipeline_smoke.py``) drives the
CLI and is the behaviour-preservation guarantee for that split.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
import datetime as dt

import pandas as pd

from .adapter import adapt_all
from .calibrate import (
    bootstrap_stability,
    compute_weights,
    holdout_evaluation,
    jaccard_verdict,
    per_cluster_jaccard,
    posterior_entropy,
    profile_clusters,
    top_distinctive_features,
)
from .cluster import assign, fit_gmm_bic
from .compile_prompts import compile_prompt, generic_fallback_name, llm_name, llm_provider_in_use
from .encoder import encode_all
from .sampling import cluster_centroid_and_scale, population_bounds, sample_synthetic_rows
from .schema import PERSONA_PACK_SCHEMA_VERSION, FeatureDelta, Persona, PersonaPack, PersonaWeights


def _noop(_: str) -> None:  # default log sink
    return None


@dataclass
class PipelineResult:
    """In-memory output of :func:`run_pipeline`.

    ``pack`` is the full :class:`~sidira_personas.schema.PersonaPack` (serialize
    with ``dataclasses.asdict``). ``prompts`` maps persona id → compiled prompt
    markdown. ``assignments`` is the per-user persona assignment + soft
    posteriors. ``baseline_crosstab`` is the optional baseline-segment ×
    persona crosstab (``None`` when no ``baseline_segment_column`` is configured
    or the column is absent).
    """

    pack: PersonaPack
    prompts: dict[str, str]
    assignments: pd.DataFrame
    baseline_crosstab: pd.DataFrame | None = field(default=None)


def run_pipeline(
    df: pd.DataFrame,
    cfg: dict,
    *,
    run_id: str,
    no_llm: bool = False,
    provider: str | None = None,
    log: Callable[[str], None] = _noop,
) -> PipelineResult:
    """Run the persona pipeline over an already-loaded ``df``.

    ``df`` is the raw per-user source table (the caller does the reading/limit).
    ``cfg`` is the parsed pipeline config (same shape as ``config/*.yaml``).
    ``run_id`` is stamped into the pack (the caller owns its naming/uniqueness).
    ``no_llm`` forces generic ``Persona##`` names. ``provider`` forces a specific
    naming backend (``"groq"`` / ``"anthropic"``); ``None`` means
    first-available-wins. ``log`` receives progress lines (the CLI passes
    ``click.echo``; the default is a no-op).
    """
    pre = cfg.get("prefilter")
    excluded_df = None
    if pre:
        col, vals = pre["column"], pre["values"]
        if col in df.columns:
            mask = df[col].isin(vals)
            excluded_df = df.loc[~mask].copy()
            df = df.loc[mask].copy().reset_index(drop=True)
            log(f"      prefilter {col} ∈ {vals}: kept {len(df):,}, set aside {len(excluded_df):,}")

    log("[2/6] Layer A — adapting per group …")
    groups = {g: specs for g, specs in cfg["feature_groups"].items()}
    snapshot_col = cfg.get("snapshot_date_column")
    snapshot = None
    if snapshot_col and snapshot_col in df.columns:
        snapshot = pd.to_datetime(df[snapshot_col]).max()
    adapted = adapt_all(df, groups, snapshot=snapshot)
    for g, feats in adapted.items():
        log(f"      {g}: {feats.shape[1]} features")

    log("[3/6] Layer B — encoding per group …")
    dim = cfg["encoder"]["embedding_dim_per_group"]
    excluded_groups = set(cfg.get("clustering_exclude_groups") or [])
    adapted_for_cluster = {g: f for g, f in adapted.items() if g not in excluded_groups}
    if excluded_groups:
        log(f"      excluding from embedding: {sorted(excluded_groups)}")
    X, _models = encode_all(adapted_for_cluster, dim)
    log(f"      embedding shape: {X.shape}")

    log("[4/6] Layer C — clustering (GMM + BIC) …")
    k_range = tuple(cfg["cluster"]["k_range"])
    gmm, k = fit_gmm_bic(X, k_range, cfg["cluster"]["random_state"])
    labels, confidence, probs = assign(gmm, X)
    log(f"      selected k = {k}")

    # v1.1 sampling config (defaults work with no config edits): N synthetic rows per persona +
    # the RNG seed for reproducible draws. See sampling.py / schema.py for the §2 step-2 contract.
    sampling_cfg = cfg.get("sampling") or {}
    n_synth = int(sampling_cfg.get("n_synthetic_rows", 64))
    synth_seed = int(sampling_cfg.get("random_state", cfg["cluster"]["random_state"]))

    log("[5/6] Layer D — calibrating (weights + bootstrap stability) …")
    weights = compute_weights(
        df, labels,
        cfg["weights"]["schemes"],
        cfg["weights"].get("orders_column"),
        cfg["weights"].get("revenue_column"),
    )
    overall_stability = bootstrap_stability(
        X, k, iters=cfg["cluster"]["bootstrap_iters"],
        random_state=cfg["cluster"]["random_state"],
    )
    log(f"      overall stability (mean ARI, coarse): {overall_stability:.3f}")
    if overall_stability < cfg["cluster"]["stability_threshold"]:
        log(f"      WARNING: overall stability below threshold {cfg['cluster']['stability_threshold']}")

    # #7: per-cluster bootstrap Jaccard (clusterboot-style) replaces the global-ARI-only signal —
    # it tells which clusters are solid (≥0.85) vs artifacts (≤0.5), which the mean hides.
    jaccard = per_cluster_jaccard(
        X, labels, k, iters=cfg["cluster"]["bootstrap_iters"],
        random_state=cfg["cluster"]["random_state"],
    )
    log("      per-cluster Jaccard stability (bootstrap, clusterboot-style):")
    for cid in sorted(jaccard):
        v = jaccard[cid]
        verdict = jaccard_verdict(v)
        log(f"        P{cid+1:02d}: {v:.3f}  [{verdict}]")

    # #7: train/test holdout — evidence the clustering generalizes (diagnostic; the shipped
    # pack still fits on all rows). test_frac defaults to 0.2; no config edit required.
    holdout = holdout_evaluation(
        X, k, test_frac=float(cfg["cluster"].get("holdout_frac", 0.2)),
        random_state=cfg["cluster"]["random_state"],
    )
    log(f"      holdout: train/test avg log-lik {holdout['train_avg_loglik']:.2f}/"
        f"{holdout['test_avg_loglik']:.2f} (gap {holdout['loglik_gap']:.2f}), "
        f"proportion drift {holdout['proportion_tv_distance']:.3f}")

    log("      computing cluster profiles in original feature units …")
    profile = profile_clusters(df, groups, labels)
    baseline = profile["baseline"]

    if no_llm:
        log("      --no-llm flag set → all personas will use generic Persona## names")
    else:
        provider_label = llm_provider_in_use(provider)
        if provider_label:
            log(f"      naming via {provider_label} (fallback to Persona## on failure)")
        else:
            log("      no LLM key (GROQ_API_KEY / ANTHROPIC_API_KEY) → using Persona## names")

    personas: list[Persona] = []
    persona_value_maps: dict[str, dict] = {}
    # Population bounds (full adapted-feature space) clamp every synthetic draw in-distribution.
    pop_bounds = population_bounds(adapted)
    for cid in sorted(set(labels.tolist())):
        feats = top_distinctive_features(
            adapted, labels, cid,
            top_n=cfg["prompt_compiler"]["top_n_distinctive_features_per_group"],
            threshold_sigma=cfg["prompt_compiler"]["threshold_sigma"],
        )
        pid = f"P{cid+1:02d}"
        pvals = profile["clusters"][int(cid)]

        # Naming: LLM is primary. Falls back to generic Persona## if LLM
        # is unavailable (no key, network error, etc.) or --no-llm is set.
        named = None if no_llm else llm_name(cfg["company_id"], pvals, feats, provider)
        if named is None:
            named = generic_fallback_name(pid, feats)
        name, desc, jtbd = named

        # v1.1: full centroid + within-cluster scale + joint synthetic rows (adapted-feature space),
        # plus the mean soft posterior over this cluster's members (component idx j → P{j+1:02d}).
        centroid_full, feature_scale = cluster_centroid_and_scale(adapted, labels, cid)
        sample_rows = sample_synthetic_rows(
            adapted, labels, cid, n_rows=n_synth, random_state=synth_seed + int(cid),
            bounds=pop_bounds,
        )
        post = probs[labels == cid].mean(axis=0)
        posterior_mean = {f"P{j+1:02d}": float(post[j]) for j in range(post.shape[0])}

        jac = jaccard.get(int(cid), 0.0)
        persona = Persona(
            id=pid,
            name=name,
            description=desc,
            jtbd=jtbd,
            size=int((labels == cid).sum()),
            weights=PersonaWeights(**weights[cid]),
            stability=jac,  # #7: per-cluster Jaccard, not the global ARI
            top_features=[FeatureDelta(**f) for f in feats],
            centroid={f["feature"]: f["centroid_value"] for f in feats},
            centroid_full=centroid_full,
            feature_scale=feature_scale,
            sample_rows=sample_rows,
            posterior_mean=posterior_mean,
            stability_verdict=jaccard_verdict(jac),
            posterior_entropy=posterior_entropy(posterior_mean),
        )
        personas.append(persona)
        persona_value_maps[pid] = pvals

    if excluded_df is not None and len(excluded_df) > 0:
        log(f"      adding {excluded_df[pre['column']].nunique()} fixed personas from prefilter")
        # Build a fake-labels array over excluded_df so we can reuse profile_clusters
        fixed_groups = list(excluded_df[pre["column"]].unique())
        fixed_label_map = {v: i for i, v in enumerate(fixed_groups)}
        fixed_labels = excluded_df[pre["column"]].map(fixed_label_map).to_numpy()
        excluded_reset = excluded_df.reset_index(drop=True)
        fixed_profile = profile_clusters(excluded_reset, groups, fixed_labels)
        # Adapt the excluded population so FIXED_* personas carry the same v1.1 fields as discovered
        # ones (computed within their own distribution; they never went through the GMM, so no posterior).
        adapted_excluded = adapt_all(excluded_reset, groups, snapshot=snapshot)
        fixed_bounds = population_bounds(adapted_excluded)
        # Compute distinctive features by comparing fixed-persona means to the CLUSTERING baseline
        for fixed_val in fixed_groups:
            cid = fixed_label_map[fixed_val]
            sub = excluded_df[excluded_df[pre["column"]] == fixed_val]
            pvals = fixed_profile["clusters"][cid]
            # Build top-features by computing pseudo-z vs clustering baseline (using baseline distribution std unknown → use rank-order)
            tf_candidates: list[dict] = []
            for feat, info in pvals.items():
                if feat not in baseline:
                    continue
                if info.get("kind") in ("count", "score", "ratio", "tenure_days"):
                    pop_v = baseline[feat]["value"]
                    delta = info["value"] - pop_v
                    denom = abs(pop_v) if pop_v else 1.0
                    z = delta / denom
                    if abs(z) >= 0.5:
                        tf_candidates.append({
                            "feature": feat, "group": info["group"],
                            "centroid_value": info["value"],
                            "population_mean": pop_v,
                            "population_std": denom,
                            "z_delta": float(z),
                        })
            tf_candidates.sort(key=lambda x: abs(x["z_delta"]), reverse=True)
            tf_candidates = tf_candidates[:15]

            cid_label = f"FIXED_{str(fixed_val).upper().replace(' ', '_')}"
            w: dict[str, float] = {"by_headcount": float(len(sub) / (len(df) + len(excluded_df)))}
            orders_col = cfg["weights"].get("orders_column")
            if orders_col and orders_col in sub.columns:
                tot_all = float(pd.concat([df, excluded_df])[orders_col].sum()) or 1.0
                w["by_orders"] = float(sub[orders_col].sum() / tot_all)
            persona_id = cid_label[:24]
            fixed_named = None if no_llm else llm_name(cfg["company_id"], pvals, tf_candidates, provider)
            if fixed_named is None:
                fixed_named = (
                    f"Fixed — {fixed_val}",
                    f"Pre-defined segment from prefilter column {pre['column']} = {fixed_val}. NOT discovered via clustering.",
                    "(not generated — LLM unavailable)",
                )
            f_name, f_desc, f_jtbd = fixed_named
            f_centroid_full, f_feature_scale = cluster_centroid_and_scale(
                adapted_excluded, fixed_labels, cid)
            f_sample_rows = sample_synthetic_rows(
                adapted_excluded, fixed_labels, cid, n_rows=n_synth,
                random_state=synth_seed + 1000 + int(cid), bounds=fixed_bounds,
            )
            personas.append(Persona(
                id=persona_id,
                name=f_name,
                description=f_desc,
                jtbd=f_jtbd,
                size=int(len(sub)),
                weights=PersonaWeights(**w),
                stability=None,  # #7: hard-defined segment — no GMM bootstrap applies (was a fake 1.0)
                top_features=[FeatureDelta(**f) for f in tf_candidates],
                centroid={f["feature"]: f["centroid_value"] for f in tf_candidates},
                centroid_full=f_centroid_full,
                feature_scale=f_feature_scale,
                sample_rows=f_sample_rows,
                posterior_mean=None,  # hard-defined segment, no GMM posterior
                stability_verdict="n/a (hard-defined)",
                posterior_entropy=None,
            ))
            persona_value_maps[persona_id] = pvals

    log("[6/6] Layer E — compiling prompts + assembling pack …")
    pack = PersonaPack(
        schema_version=PERSONA_PACK_SCHEMA_VERSION,
        company_id=cfg["company_id"],
        run_id=run_id,
        created_at=dt.datetime.utcnow().isoformat(),
        n_users=int(len(df)),
        feature_groups=list(groups.keys()),
        personas=personas,
        cluster_method=cfg["cluster"]["method"],
        k_selected=k,
        encoder_method=cfg["encoder"]["method"],
        embedding_dim_per_group=dim,
        overall_stability=overall_stability,
        notes={
            "stability_method": (
                "per_cluster_jaccard (bootstrap Jaccard, clusterboot-style classify-all variant; "
                "Hennig interpretation bands): per-persona stability lives on Persona.stability; "
                "bands ≥0.85 highly stable, ≥0.75 valid, ≤0.5 dissolved. overall_stability is the "
                "coarse mean-ARI, retained for back-compat."
            ),
            "per_cluster_jaccard": {f"P{c+1:02d}": jaccard[c] for c in sorted(jaccard)},
            "holdout": holdout,
        },
    )

    # Build an explicit scope label so the LLM knows what "baseline" actually means
    src = cfg["source"]
    src_ref = (f"{src.get('database','?')}.{src.get('schema','?')}.{src.get('table','?')}"
               if src["type"] == "snowflake" else src.get("path", "local file"))
    scope_parts = [f"Snapshot from {src_ref}", f"{int(len(df)+len(excluded_df) if excluded_df is not None else len(df)):,} users in source"]
    if pre:
        scope_parts.append(
            f"Baseline = the {len(df):,} users where {pre['column']} ∈ {pre['values']} "
            f"(prefiltered before clustering)"
        )
    else:
        scope_parts.append(f"Baseline = all {len(df):,} users in the sample")
    scope_parts.append(f"This is a sample, not the full {cfg['company_id']} population.")
    baseline_scope = " | ".join(scope_parts)
    baseline_n = len(df)

    prompts: dict[str, str] = {}
    for p in personas:
        pvals = persona_value_maps.get(p.id, {})
        prompts[p.id] = compile_prompt(
            cfg["company_id"],
            {"name": p.name, "description": p.description, "jtbd": p.jtbd},
            pvals,
            baseline,
            [asdict(f) for f in p.top_features],
            baseline_scope=baseline_scope,
            baseline_n=baseline_n,
        )

    user_id_col = src.get("user_id_column", "user_id")
    assignments = pd.DataFrame({
        user_id_col: df[user_id_col].values if user_id_col in df.columns else range(len(df)),
        "persona_id": [f"P{c+1:02d}" for c in labels],
        "confidence": confidence,
    })
    # v1.1: carry the full soft posterior through to disk (one post_P* column per GMM component),
    # not just the max-confidence — the #7 "persist the posterior end-to-end" deliverable.
    for j in range(probs.shape[1]):
        assignments[f"post_P{j+1:02d}"] = probs[:, j]

    baseline_crosstab = None
    baseline_col = cfg.get("baseline_segment_column")
    if baseline_col and baseline_col in df.columns:
        baseline_crosstab = pd.crosstab(
            df[baseline_col].fillna("UNK"),
            assignments["persona_id"].values,
            margins=True, margins_name="TOTAL",
        )

    return PipelineResult(
        pack=pack,
        prompts=prompts,
        assignments=assignments,
        baseline_crosstab=baseline_crosstab,
    )
