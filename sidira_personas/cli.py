"""CLI entry point: sidira-personas run --config config/<company>.yaml"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import asdict
from pathlib import Path

import click
import pandas as pd
import yaml

from .adapter import adapt_all
from .calibrate import bootstrap_stability, compute_weights, profile_clusters, top_distinctive_features
from .cluster import assign, fit_gmm_bic
from .compile_prompts import compile_prompt, generic_fallback_name, llm_name, llm_provider_in_use
from .encoder import encode_all
from .io import read_local, read_snowflake
from .schema import PERSONA_PACK_SCHEMA_VERSION, FeatureDelta, Persona, PersonaPack, PersonaWeights


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--out", "out_dir", default="outputs", type=click.Path())
@click.option("--limit", default=None, type=int, help="Optional row cap for fast iteration.")
@click.option("--no-llm", is_flag=True, default=False,
              help="Skip LLM naming entirely; use generic Persona01/02… fallback for every persona.")
def run(config_path: str, out_dir: str, limit: int | None, no_llm: bool) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    run_id = f"{cfg.get('run_id_prefix', 'run')}-{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    out = Path(out_dir) / run_id
    (out / "prompts").mkdir(parents=True, exist_ok=True)

    click.echo(f"[1/6] Reading source ({cfg['source']['type']}) …")
    src = cfg["source"]
    if src["type"] == "snowflake":
        df = read_snowflake(src["database"], src["schema"], src["table"])
    else:
        df = read_local(src["path"])
    if limit:
        df = df.head(limit)
    click.echo(f"      {len(df):,} rows, {df.shape[1]} columns")

    pre = cfg.get("prefilter")
    excluded_df = None
    if pre:
        col, vals = pre["column"], pre["values"]
        if col in df.columns:
            mask = df[col].isin(vals)
            excluded_df = df.loc[~mask].copy()
            df = df.loc[mask].copy().reset_index(drop=True)
            click.echo(f"      prefilter {col} ∈ {vals}: kept {len(df):,}, set aside {len(excluded_df):,}")

    click.echo("[2/6] Layer A — adapting per group …")
    groups = {g: specs for g, specs in cfg["feature_groups"].items()}
    snapshot_col = cfg.get("snapshot_date_column")
    snapshot = None
    if snapshot_col and snapshot_col in df.columns:
        snapshot = pd.to_datetime(df[snapshot_col]).max()
    adapted = adapt_all(df, groups, snapshot=snapshot)
    for g, feats in adapted.items():
        click.echo(f"      {g}: {feats.shape[1]} features")

    click.echo("[3/6] Layer B — encoding per group …")
    dim = cfg["encoder"]["embedding_dim_per_group"]
    excluded_groups = set(cfg.get("clustering_exclude_groups") or [])
    adapted_for_cluster = {g: f for g, f in adapted.items() if g not in excluded_groups}
    if excluded_groups:
        click.echo(f"      excluding from embedding: {sorted(excluded_groups)}")
    X, _models = encode_all(adapted_for_cluster, dim)
    click.echo(f"      embedding shape: {X.shape}")

    click.echo("[4/6] Layer C — clustering (GMM + BIC) …")
    k_range = tuple(cfg["cluster"]["k_range"])
    gmm, k = fit_gmm_bic(X, k_range, cfg["cluster"]["random_state"])
    labels, confidence = assign(gmm, X)
    click.echo(f"      selected k = {k}")

    click.echo("[5/6] Layer D — calibrating (weights + bootstrap stability) …")
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
    click.echo(f"      overall stability (mean ARI): {overall_stability:.3f}")
    if overall_stability < cfg["cluster"]["stability_threshold"]:
        click.secho(f"      WARNING: stability below threshold {cfg['cluster']['stability_threshold']}",
                    fg="yellow")

    click.echo("      computing cluster profiles in original feature units …")
    profile = profile_clusters(df, groups, labels)
    baseline = profile["baseline"]

    if no_llm:
        click.echo("      --no-llm flag set → all personas will use generic Persona## names")
    else:
        provider = llm_provider_in_use()
        if provider:
            click.echo(f"      naming via {provider} (fallback to Persona## on failure)")
        else:
            click.secho("      no LLM key (GROQ_API_KEY / ANTHROPIC_API_KEY) → using Persona## names",
                        fg="yellow")

    personas: list[Persona] = []
    persona_value_maps: dict[str, dict] = {}
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
        named = None if no_llm else llm_name(cfg["company_id"], pvals, feats)
        if named is None:
            named = generic_fallback_name(pid, feats)
        name, desc, jtbd = named

        persona = Persona(
            id=pid,
            name=name,
            description=desc,
            jtbd=jtbd,
            size=int((labels == cid).sum()),
            weights=PersonaWeights(**weights[cid]),
            stability=overall_stability,
            top_features=[FeatureDelta(**f) for f in feats],
            centroid={f["feature"]: f["centroid_value"] for f in feats},
        )
        personas.append(persona)
        persona_value_maps[pid] = pvals

    if excluded_df is not None and len(excluded_df) > 0:
        click.echo(f"      adding {excluded_df[pre['column']].nunique()} fixed personas from prefilter")
        # Build a fake-labels array over excluded_df so we can reuse profile_clusters
        fixed_groups = list(excluded_df[pre["column"]].unique())
        fixed_label_map = {v: i for i, v in enumerate(fixed_groups)}
        fixed_labels = excluded_df[pre["column"]].map(fixed_label_map).to_numpy()
        fixed_profile = profile_clusters(excluded_df.reset_index(drop=True), groups, fixed_labels)
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
            fixed_named = None if no_llm else llm_name(cfg["company_id"], pvals, tf_candidates)
            if fixed_named is None:
                fixed_named = (
                    f"Fixed — {fixed_val}",
                    f"Pre-defined segment from prefilter column {pre['column']} = {fixed_val}. NOT discovered via clustering.",
                    "(not generated — LLM unavailable)",
                )
            f_name, f_desc, f_jtbd = fixed_named
            personas.append(Persona(
                id=persona_id,
                name=f_name,
                description=f_desc,
                jtbd=f_jtbd,
                size=int(len(sub)),
                weights=PersonaWeights(**w),
                stability=1.0,
                top_features=[FeatureDelta(**f) for f in tf_candidates],
                centroid={f["feature"]: f["centroid_value"] for f in tf_candidates},
            ))
            persona_value_maps[persona_id] = pvals

    click.echo("[6/6] Layer E — compiling prompts + writing artifacts …")
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
    )
    pack.to_json(str(out / "persona_pack.json"))

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
    scope_parts.append("This is a sample, not the full DoorDash population.")
    baseline_scope = " | ".join(scope_parts)
    baseline_n = len(df)

    for p in personas:
        pvals = persona_value_maps.get(p.id, {})
        prompt = compile_prompt(
            cfg["company_id"],
            {"name": p.name, "description": p.description, "jtbd": p.jtbd},
            pvals,
            baseline,
            [asdict(f) for f in p.top_features],
            baseline_scope=baseline_scope,
            baseline_n=baseline_n,
        )
        (out / "prompts" / f"{p.id}.md").write_text(prompt)

    user_id_col = src.get("user_id_column", "consumer_id")
    assignments = pd.DataFrame({
        user_id_col: df[user_id_col].values if user_id_col in df.columns else range(len(df)),
        "persona_id": [f"P{c+1:02d}" for c in labels],
        "confidence": confidence,
    })
    assignments.to_parquet(out / "assignments.parquet", index=False)

    baseline_col = cfg.get("baseline_segment_column")
    if baseline_col and baseline_col in df.columns:
        click.echo("\nBaseline segment vs. discovered persona — crosstab:")
        ct = pd.crosstab(df[baseline_col].fillna("UNK"),
                         assignments["persona_id"].values,
                         margins=True, margins_name="TOTAL")
        click.echo(ct.to_string())
        ct.to_csv(out / "baseline_crosstab.csv")

    click.echo("\nPersona sizes & weights:")
    for p in personas:
        w = p.weights
        click.echo(f"  {p.id}: n={p.size:>5d}  "
                   f"headcount={w.by_headcount:.3f}  "
                   f"orders={(w.by_orders or 0):.3f}")

    click.secho(f"\n✓ Persona Pack written to {out}", fg="green")


if __name__ == "__main__":
    cli()
