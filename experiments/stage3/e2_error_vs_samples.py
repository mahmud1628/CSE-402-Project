"""E2: repeated MCW/PW/PPW errors and ranking reliability versus walk budget."""
from __future__ import annotations
try:
    from .common import *
except ImportError:  # pragma: no cover
    from common import *

def run(args):
    cfg=config(args); all_nets=load_networks(args.stage2_results); primary=match_networks(all_nets); secondary=choose_secondary(primary, cfg["quick"])
    if cfg["quick"]: primary=secondary
    if cfg["network_limit"]: primary=primary[:cfg["network_limit"]]; secondary=secondary[:cfg["network_limit"]]
    run_manifest(args,cfg,all_nets)
    jobs=[]
    for cohort,nets in (("primary",primary),("secondary",secondary)):
        for net in nets:
            specs=source_specs(net, cohort=="secondary")
            for sid,sigma in specs:
                if cohort=="primary" and sid != "uniform": continue
                for alpha in cfg["alphas"]:
                    for T in cfg["walk_budgets"]:
                        for method in WALK_METHODS:
                            K=0 if method=="mcw" else (cfg["pw_K"] if method=="pw" else cfg["ppw_K"]); B=cfg["ppw_B"] if method=="ppw" else 1
                            jobs.append(dict(experiment="walks",cohort=cohort,network=net,source_id=sid,sigma=sigma,alpha=alpha,method=method,T=T,K=K,B=B,residuals=method=="ppw"))
    for job in track("E2 walks",jobs,args.output,describe_trial_job): execute_trials(args,cfg,**job)
    sort_trials(args.output); trials=read_trials(args.output); summary=summarize(args.output,args.stage2_results,{n.network_id:n for n in all_nets},trials)
    d=summary[summary["experiment"].eq("walks")]; plt=mpl(); fig,ax=plt.subplots(figsize=(6,4))
    for method, x in d.groupby("method"):
        x=x.groupby(["method","T"],as_index=False).l1.mean().sort_values("T"); st=METHOD_STYLE[method]; ax.loglog(x["T"],x["l1"],marker=st["marker"],color=st["color"],label=st["label"])
    ax.set(xlabel="walks T",ylabel="mean per-network L1 error",title="Stage 3 E2 · error versus walk budget"); ax.legend(); fig.tight_layout(); save_figure(fig,args.output,"e2_error_vs_samples")
    write_report(args.output,summary); return summary
if __name__ == "__main__": run(parse_args(__doc__))
