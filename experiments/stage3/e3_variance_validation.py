"""E3: variance/MSE checks using stored E2 trials and the paper's MCW/PW formulas."""
from __future__ import annotations
import numpy as np
import pandas as pd
try:
    from .common import *
except ImportError:  # pragma: no cover
    from common import *

def run(args):
    cfg=config(args); nets=load_networks(args.stage2_results); run_manifest(args,cfg,nets); trials=read_trials(args.output)
    if trials.empty: raise RuntimeError("run e2_error_vs_samples.py before E3 so variance uses the cached trials")
    index={n.network_id:n for n in nets}; rows=[]
    for keys,d in trials[trials["experiment"].eq("walks")].groupby(["network_id","source_id","alpha","method","T","K","B"]):
        nid,sid,alpha,method,T,K,B=keys; net=index[nid]; sigma=dict(source_specs(net,True)).get(sid,pl.uniform(net.graph)); truth=truth_for(args.stage2_results,net,alpha,sid,sigma)
        X=np.stack([np.load(Path(args.output)/p) for p in d.estimate_path]); mse=float(((X-truth)**2).sum(axis=1).mean()); empirical=float(X.var(axis=0,ddof=1).sum()) if len(X)>1 else np.nan
        theory=np.nan
        if method=="mcw": theory=pl.theory.walk_variance(truth)/T
        elif method=="pw": theory=pl.theory.exact_pw_variance(net.graph,truth,alpha,K)/T
        residual=np.array([json.loads(x) for x in d.residual_l1]); rows.append(dict(network_id=nid,source_id=sid,alpha=alpha,method=method,T=T,K=K,B=B,n_trials=len(d),mse=mse,total_sample_variance=empirical,theory_variance=theory,bias_l1=float(np.abs(X.mean(0)-truth).sum()),mean_residual_l1=json.dumps(residual.mean(0).tolist())))
    out=pd.DataFrame(rows); out.to_csv(output_paths(args.output)["tables"] / "variance_validation.csv",index=False); return out
if __name__ == "__main__": run(parse_args(__doc__))
