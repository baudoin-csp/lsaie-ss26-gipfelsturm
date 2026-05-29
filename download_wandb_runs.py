import wandb
import os

api = wandb.Api()

run_ids = {
    "baseline": "ar2wlr1u",
    "flash_attn": "iwp9kyas",
    "liger": "4bffxrsh",
    "fa_liger": "gj9wa0oz",
    "cce": "jwa5kvnw"
}

os.makedirs("wandb_runs", exist_ok=True)

for name, run_id in run_ids.items():
    run = api.run(f"lsai-project-26/gipfelsturm/{run_id}")
    df = run.history()
    df.to_csv(f"wandb_runs/{name}_{run_id}.csv", index=False)
    print(f"Saved {name} ({run_id}) — {len(df)} rows")

print("Done.")
