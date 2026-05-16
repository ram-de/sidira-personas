"""CLI entry point: pythia-personas run --config config/<company>.yaml"""
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
from .calibrate import bootstrap_stability, compute_weights, top_distinctive_features
from .cluster import assign, fit_gmm_bic
from .compile_prompts import compile_prompt
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
def run(config_path: str, out_dir: str, limit: int | None) -> None:
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

    click.echo("[2/6] Layer A — adapting per group …")
    groups = {g: specs for g, specs in cfg["feature_groups"].items()}
    adapted = adapt_all(df, groups)
    for g, feats in adapted.items():
        click.echo(f"      {g}: {feats.shape[1]} features")

    click.echo("[3/6] Layer B — encoding per group …")
    dim = cfg["encoder"]["embedding_dim_per_group"]
    X, _models = encode_all(adapted, dim)
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

    personas: list[Persona] = []
    for cid in sorted(set(labels.tolist())):
        feats = top_distinctive_features(
            adapted, labels, cid,
            top_n=cfg["prompt_compiler"]["top_n_distinctive_features_per_group"],
            threshold_sigma=cfg["prompt_compiler"]["threshold_sigma"],
        )
        pid = f"P{cid+1:02d}"
        persona = Persona(
            id=pid,
            name=f"Persona {pid}",          # LLM naming hook lives in Layer E
            description="(to be generated)",
            jtbd="(to be generated)",
            size=int((labels == cid).sum()),
            weights=PersonaWeights(**weights[cid]),
            stability=overall_stability,    # per-cluster stability is a follow-up
            top_features=[FeatureDelta(**f) for f in feats],
            centroid={f["feature"]: f["centroid_value"] for f in feats},
        )
        personas.append(persona)

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

    for p in personas:
        prompt = compile_prompt(
            cfg["company_id"],
            {"name": p.name, "description": p.description, "jtbd": p.jtbd},
            [asdict(f) for f in p.top_features],
        )
        (out / "prompts" / f"{p.id}.md").write_text(prompt)

    user_id_col = src.get("user_id_column", "consumer_id")
    assignments = pd.DataFrame({
        user_id_col: df[user_id_col].values if user_id_col in df.columns else range(len(df)),
        "persona_id": [f"P{c+1:02d}" for c in labels],
        "confidence": confidence,
    })
    assignments.to_parquet(out / "assignments.parquet", index=False)

    click.secho(f"✓ Persona Pack written to {out}", fg="green")


if __name__ == "__main__":
    cli()
