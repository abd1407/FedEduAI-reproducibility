"""Single-process transport of REAL Flower SecAgg+ messages, NOT distributed deployment.
Serialization counts are protocol-payload bytes, excluding TCP/TLS/retransmissions.
Local host can access all data: security evidence here is protocol execution only.
"""
import time,uuid
from flwr.server import Grid
from flwr.common import Context,Message,RecordDict
from flwr.common.typing import Run
from flwr.app.metadata import Metadata
from flwr.common.serde import message_to_proto,message_from_proto
from .client_app import app as client_app

class LocalGrid(Grid):
    def __init__(self,config):
        self.config=config;self._run=Run.create_empty(1);self.replies={}
        self.contexts={i+1:Context(run_id=1,node_id=i+1,node_config={'partition-id':i},state=RecordDict(),run_config=config)
                       for i in range(int(config['num-clients']))}
        self.payload_bytes={'server_to_clients':0,'clients_to_server':0}
        self.exposed_parameter_arrays=0;self.message_count=0
    @property
    def run(self):return self._run
    def set_run(self,run_id):
        if run_id!=1:raise ValueError('LocalGrid supports its single run only')
    def get_node_ids(self):return list(self.contexts)
    def create_message(self,content,message_type,dst_node_id,group_id,ttl=None):
        return Message(content=content,dst_node_id=dst_node_id,message_type=message_type,group_id=group_id,ttl=ttl)
    def push_messages(self,messages):
        ids=[]
        for m in messages:
            old=m.metadata;mid=uuid.uuid4().hex
            meta=Metadata(1,mid,0,old.dst_node_id,'',old.group_id,time.time(),old.ttl,old.message_type)
            message=Message(metadata=meta,content=m.content)
            wire=message_to_proto(message);self.payload_bytes['server_to_clients']+=wire.ByteSize()
            reply=client_app(message_from_proto(wire),self.contexts[old.dst_node_id])
            if reply.has_error():raise RuntimeError(str(reply.error))
            if reply.has_content():
                exposed=sum(len(r) for r in reply.content.array_records.values())
                self.exposed_parameter_arrays+=exposed
                if self.config['secagg'] and exposed:raise RuntimeError('SecAgg client exposed unmasked parameter arrays')
            wire=message_to_proto(reply);self.payload_bytes['clients_to_server']+=wire.ByteSize()
            self.replies[mid]=message_from_proto(wire);self.message_count+=1;ids.append(mid)
        return ids
    def pull_messages(self,message_ids):return [self.replies.pop(i) for i in message_ids]
    def send_and_receive(self,messages,*,timeout=None):return self.pull_messages(self.push_messages(messages))


def simulate(config):
    from .server_app import app as server_app
    grid=LocalGrid(config)
    server_app(grid,Context(run_id=1,node_id=0,node_config={},state=RecordDict(),run_config=config))
    return {'accountant_histories':{str(k):__import__('json').loads(c.state.config_records['student-dp-accountant']['history']) for k,c in grid.contexts.items()},
        'transport':'single_process_serialized_Flower_messages',
        'payload_bytes':grid.payload_bytes,'message_count':grid.message_count,
        'individual_parameter_arrays_visible_at_server_boundary':grid.exposed_parameter_arrays,
        'network_bytes_measured':False,'host_isolation':False,
        'warning':'Protocol execution audit, not proof against a malicious host or server/client collusion'}
