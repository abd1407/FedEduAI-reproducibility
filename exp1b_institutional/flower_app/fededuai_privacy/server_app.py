import json,os,time
from pathlib import Path
import numpy as np
from flwr.common import ndarrays_to_parameters,parameters_to_ndarrays
from flwr.server import Grid,LegacyContext,ServerApp,ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow,SecAggPlusWorkflow
from .task import make_net,weights

app=ServerApp()

@app.main()
def main(grid:Grid,context):
    rc=context.run_config;n=int(rc['num-clients']);rounds=int(rc['num-server-rounds'])
    if n<3:raise ValueError('At least three clients required for the configured secure aggregation experiment')
    root=Path(os.environ['FEDEDUAI_SERVER_OUTPUT']);root.mkdir(parents=True,exist_ok=True)
    tag=str(rc['case-tag'])
    if not tag.replace('_','').replace('-','').isalnum():raise ValueError('Invalid case tag')
    target=root/(tag+'.json')
    if target.exists():raise FileExistsError('Use a fresh case tag/output folder to avoid mixing runs')
    t0=time.perf_counter();records=[];pending={}
    def fit_metrics(items):
        pending.update(max_client_epsilon=max([m.get('epsilon',0.) for _,m in items],default=0.),
                       client_training_seconds=sum(m.get('seconds',0.) for _,m in items))
        return dict(pending)
    def evaluate(server_round,parameters,config):
        rec={'round':server_round,**pending,'seconds_since_start':time.perf_counter()-t0}
        records.append(rec)
        # Global model is saved in a private output directory, not the publication archive.
        np.savez(root/(tag+'_model.npz'),**{f'p{i}':p for i,p in enumerate(parameters)})
        # Per-round snapshots allow validation learning curves to be computed outside the server.
        np.savez(root/(tag+f'_round{server_round}_model.npz'),**{f'p{i}':p for i,p in enumerate(parameters)})
        result={'status':'complete' if server_round==rounds else 'in_progress',
                'mode':os.environ.get('FEDEDUAI_EXECUTION_MODE','unspecified'),
                'secure_aggregation':bool(rc['secagg']),'dp':bool(rc['dp']),'clip_only':bool(rc.get('clip-only',False)),'poisson_only':bool(rc.get('poisson-only',False)),
                'rounds':records,'config':dict(rc),
                'warning':'Model-training accounting only; research metrics/baseline models are not DP releases.'}
        tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(target)
        return 0.,{}
    strategy=FedAvg(fraction_fit=1.,fraction_evaluate=0.,min_fit_clients=n,min_available_clients=n,
        accept_failures=False,initial_parameters=ndarrays_to_parameters(weights(make_net(int(rc['input-dim']),int(rc['seed'])))),
        on_fit_config_fn=lambda rnd:{'server_round':rnd},fit_metrics_aggregation_fn=fit_metrics,evaluate_fn=evaluate)
    legacy=LegacyContext(context=context,config=ServerConfig(num_rounds=rounds),strategy=strategy)
    if bool(rc['secagg']):
        # Built-in protocol, not a custom masking approximation. Full participation for primary comparison.
        fit=SecAggPlusWorkflow(num_shares=n,reconstruction_threshold=n-1,
            max_weight=float(rc['max-weight']),clipping_range=float(rc['clipping-range']))
        workflow=DefaultWorkflow(fit_workflow=fit)
    else:workflow=DefaultWorkflow()
    workflow(grid,legacy)
    result=json.loads(target.read_text())
    if result['status']!='complete':raise RuntimeError('Federation did not finish all configured rounds')
