"""E5: algorithm-only runtime/error comparison and directed forest cohort."""
from __future__ import annotations
import numpy as np
import pandas as pd
try:
    from .common import *
except ImportError:  # pragma: no cover
    from common import *

def run(args):
    cfg=config(args); all_nets=load_networks(args.stage2_results); nets=choose_secondary(match_networks(all_nets),cfg["quick"])
    if cfg["network_limit"]: nets=nets[:cfg["network_limit"]]
    run_manifest(args,cfg,all_nets)
    jobs=[]
    for net in nets:
      for sid,sigma in source_specs(net,False):
       for alpha in cfg["alphas"]:
        for T in cfg["walk_budgets"]:
         # Same keys as E2's walk trials, so these are reused from the cache rather than rerun.
         for method in WALK_METHODS: jobs.append(dict(experiment="walks",cohort="secondary",network=net,source_id=sid,sigma=sigma,alpha=alpha,method=method,T=T,K=0 if method=="mcw" else (cfg["pw_K"] if method=="pw" else cfg["ppw_K"]),B=cfg["ppw_B"] if method=="ppw" else 1,residuals=method=="ppw"))
        for F in cfg["forest_budgets"]:
         for method in FOREST_METHODS: jobs.append(dict(experiment="runtime_forests",cohort="secondary",network=net,source_id=sid,sigma=sigma,alpha=alpha,method=method,F=F,K=0 if method=="mcf" else (cfg["pw_K"] if method=="pf" else cfg["ppw_K"]),B=cfg["ppw_B"] if method=="ppf" else 1))
    for job in track("E5 runtime/forests",jobs,args.output,describe_trial_job): execute_trials(args,cfg,**job)
    sort_trials(args.output); summary=summarize(args.output,args.stage2_results,{n.network_id:n for n in all_nets},read_trials(args.output))
    ids={n.network_id for n in nets}; walks=summary["experiment"].eq("walks")&summary["network_id"].isin(ids)&summary["source_id"].eq("uniform")
    d=summary[summary["experiment"].eq("runtime_forests")|walks]; rows=[]
    for keys,x in d.groupby(["network_id","source_id","alpha","method"]):
      for target in (1e-2,1e-3):
       hit=x[x["l1"]<=target].sort_values(["T","F"])
       budget = int(hit.iloc[0]["T"] or hit.iloc[0]["F"]) if len(hit) else np.nan
       rows.append(dict(network_id=keys[0],source_id=keys[1],alpha=keys[2],method=keys[3],target_l1=target,tested_budget=budget,success_fraction=float((x["l1"]<=target).mean()),status="reached" if len(hit) else "not_reached"))
    pd.DataFrame(rows).to_csv(output_paths(args.output)["tables"] / "sample_to_target.csv",index=False)
    plt=mpl(); fig,ax=plt.subplots(figsize=(6,4))
    for method,x in d.groupby("method"):
      ax.loglog(x["seconds"],x["l1"],linestyle="none",marker=METHOD_STYLE[method]["marker"],color=METHOD_STYLE[method]["color"],label=method)
    ax.set(xlabel="algorithm-only seconds",ylabel="mean L1 error",title="Stage 3 E5 · runtime/error trade-off"); ax.legend(); fig.tight_layout(); save_figure(fig,args.output,"e5_runtime_forests")
    write_report(args.output,summary); return summary
if __name__ == "__main__": run(parse_args(__doc__))
