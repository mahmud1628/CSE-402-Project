"""E4: PW propagation K and PPW batch sensitivity on the deterministic secondary cohort."""
from __future__ import annotations
try:
    from .common import *
except ImportError:  # pragma: no cover
    from common import *

def run(args):
    cfg=config(args); all_nets=load_networks(args.stage2_results); nets=choose_secondary(match_networks(all_nets),cfg["quick"])
    if cfg["network_limit"]: nets=nets[:cfg["network_limit"]]
    run_manifest(args,cfg,all_nets); T=max(cfg["walk_budgets"])
    jobs=[]
    for net in nets:
      for sid,sigma in source_specs(net,False):
       for alpha in cfg["alphas"]:
        for K in cfg["K_grid"]: jobs.append(dict(experiment="parameters",cohort="secondary",network=net,source_id=sid,sigma=sigma,alpha=alpha,method="pw",T=T,K=K))
        # Fixed total propagation of 10 (actual K*B is recorded) rather than silently rounding a per-batch setting.
        for B in cfg["B_grid"]: jobs.append(dict(experiment="parameters",cohort="secondary",network=net,source_id=sid,sigma=sigma,alpha=alpha,method="ppw",T=T,K=max(1,10//B),B=B,residuals=True))
    for job in track("E4 parameters",jobs,args.output,describe_trial_job): execute_trials(args,cfg,**job)
    sort_trials(args.output); summary=summarize(args.output,args.stage2_results,{n.network_id:n for n in all_nets},read_trials(args.output)); d=summary[summary["experiment"].eq("parameters")]; plt=mpl(); fig,ax=plt.subplots(figsize=(6,4))
    for method,x in d.groupby("method"):
        x=x.groupby(["method","K"],as_index=False).l1.mean().sort_values("K"); ax.semilogy(x["K"],x["l1"],marker=METHOD_STYLE[method]["marker"],color=METHOD_STYLE[method]["color"],label=method)
    ax.set(xlabel="K per batch",ylabel="mean L1 error",title="Stage 3 E4 · propagation sensitivity"); ax.legend(); fig.tight_layout(); save_figure(fig,args.output,"e4_parameters"); return summary
if __name__ == "__main__": run(parse_args(__doc__))
