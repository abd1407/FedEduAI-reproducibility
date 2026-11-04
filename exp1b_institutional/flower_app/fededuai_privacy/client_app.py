"""Official Flower SecAgg+ mod protects arrays; ordinary metrics are not masked."""
import json
import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import ConfigRecord
from .task import load_shard,local_train,run_cfg,make_net,set_weights,probabilities

class StudentClient(NumPyClient):
    def __init__(self,context):
        self.context=context
        self.pid=int(context.node_config['partition-id'])
        self.shard=load_shard(self.pid);self.cfg=run_cfg(context)
    def fit(self,parameters,config):
        key='student-dp-accountant'
        prev=self.context.state.config_records.get(key)
        history=json.loads(prev['history']) if prev else []
        seen=int(prev['round']) if prev else 0
        rnd=int(config['server_round'])
        if rnd!=seen+1: raise RuntimeError('Unexpected round/retry: stop rather than reset DP accounting')
        self.cfg.update(partition=self.pid,round_number=rnd)
        arrays,history,stats=local_train(parameters,self.shard['x_train'],self.shard['y_train'],self.cfg,history)
        arrays=[np.clip(a,-float(self.context.run_config['clipping-range']),float(self.context.run_config['clipping-range'])) for a in arrays]
        self.context.state.config_records[key]=ConfigRecord({'history':json.dumps(history),'round':rnd})
        # No losses, labels, identifiers or individual predictions in fit metrics.
        metrics={'epsilon':stats['epsilon'] or 0.,'seconds':stats['seconds'],'steps':stats['steps']}
        return arrays,len(self.shard['y_train']),metrics
    def evaluate(self,parameters,config):
        m=make_net(self.shard['x_train'].shape[1],self.cfg['seed']);set_weights(m,parameters)
        y=self.shard['y_test'];p=probabilities(m,self.shard['x_test']);pred=p>=.5
        if not len(y):return 0.,0,{}
        return float(np.mean(-(y*np.log(np.clip(p,1e-7,1))+(1-y)*np.log(np.clip(1-p,1e-7,1))))),len(y),{
            'tp':int(((y==1)&pred).sum()),'fp':int(((y==0)&pred).sum()),
            'tn':int(((y==0)&~pred).sum()),'fn':int(((y==1)&~pred).sum())}

def client_fn(context):return StudentClient(context).to_client()
def configured_secagg_mod(message,context,call_next):
    if context.run_config['secagg']:
        return secaggplus_mod(message,context,call_next)
    return call_next(message,context)

app=ClientApp(client_fn=client_fn,mods=[configured_secagg_mod])
