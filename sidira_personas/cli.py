"""CLI entry point: sidira-personas run --config config/<company>.yaml

Thin wrapper over :func:`sidira_personas.pipeline.run_pipeline`: this layer owns
source reading (Snowflake/local + ``--limit``) and artifact writing (run dir,
prompts, parquet, crosstab); all the modelling lives in ``run_pipeline``.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import click
import yaml

from .io import read_local, read_snowflake
from .pipeline import run_pipeline


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--out", "out_dir", default="outputs", type=click.Path())
@click.option("--limit", default=None, type=int, help="Optional row cap for fast iteration.")
@click.option("--no-llm", is_flag=True, default=False,
              help="Skip LLM naming entirely; use generic Persona01/02… fallback for every persona.")
@click.option("--require-llm", is_flag=True, default=False,
              help="Production posture: fail the build if any persona would ship with a "
                   "generic-fallback (jargon) description instead of an LLM name. Use for "
                   "real-customer packs. Mutually exclusive with --no-llm.")
def run(config_path: str, out_dir: str, limit: int | None, no_llm: bool,
        require_llm: bool) -> None:
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

    result = run_pipeline(df, cfg, run_id=run_id, no_llm=no_llm,
                          require_llm=require_llm, log=click.echo)

    result.pack.to_json(str(out / "persona_pack.json"))
    for pid, prompt in result.prompts.items():
        (out / "prompts" / f"{pid}.md").write_text(prompt)
    result.assignments.to_parquet(out / "assignments.parquet", index=False)

    if result.baseline_crosstab is not None:
        click.echo("\nBaseline segment vs. discovered persona — crosstab:")
        click.echo(result.baseline_crosstab.to_string())
        result.baseline_crosstab.to_csv(out / "baseline_crosstab.csv")

    click.echo("\nPersona sizes & weights:")
    for p in result.pack.personas:
        w = p.weights
        click.echo(f"  {p.id}: n={p.size:>5d}  "
                   f"headcount={w.by_headcount:.3f}  "
                   f"orders={(w.by_orders or 0):.3f}")

    click.secho(f"\n✓ Persona Pack written to {out}", fg="green")


if __name__ == "__main__":
    cli()
