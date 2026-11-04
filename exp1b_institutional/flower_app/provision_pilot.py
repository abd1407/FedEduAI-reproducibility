"""Generate fresh pilot certificates/keys OUTSIDE the distributable application."""
import argparse,csv,datetime,ipaddress
from pathlib import Path
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.asymmetric import ec,rsa
p=argparse.ArgumentParser();p.add_argument('--server-host',required=True);p.add_argument('--output',required=True);p.add_argument('--clients',type=int,default=3)
a=p.parse_args();out=Path(a.output).resolve()
if out.exists():raise SystemExit('Choose a new empty output path; existing keys are never overwritten')
out.mkdir(parents=True,mode=0o700)
def write(name,data,private=False):
 q=out/name;q.parent.mkdir(exist_ok=True);q.write_bytes(data);q.chmod(0o600 if private else 0o644)
now=datetime.datetime.now(datetime.timezone.utc);ca=rsa.generate_private_key(public_exponent=65537,key_size=3072)
name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'FedEduAI v19 pilot CA')])
cert=x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(ca.public_key()).serial_number(x509.random_serial_number()).not_valid_before(now-datetime.timedelta(minutes=5)).not_valid_after(now+datetime.timedelta(days=90)).add_extension(x509.BasicConstraints(ca=True,path_length=0),critical=True).sign(ca,hashes.SHA256())
write('ca.key',ca.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()),True)
write('ca.crt',cert.public_bytes(serialization.Encoding.PEM))
server=rsa.generate_private_key(public_exponent=65537,key_size=3072)
try:san=x509.IPAddress(ipaddress.ip_address(a.server_host))
except ValueError:san=x509.DNSName(a.server_host)
sc=x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,a.server_host)])).issuer_name(name).public_key(server.public_key()).serial_number(x509.random_serial_number()).not_valid_before(now-datetime.timedelta(minutes=5)).not_valid_after(now+datetime.timedelta(days=60)).add_extension(x509.SubjectAlternativeName([san,x509.DNSName('localhost'),x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]),critical=False).sign(ca,hashes.SHA256())
write('server.key',server.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()),True);write('server.crt',sc.public_bytes(serialization.Encoding.PEM))
keys=[]
for i in range(a.clients):
 key=ec.generate_private_key(ec.SECP384R1());pub=key.public_key().public_bytes(serialization.Encoding.OpenSSH,serialization.PublicFormat.OpenSSH)
 write(f'node{i}.key',key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.OpenSSH,serialization.NoEncryption()),True)
 write(f'node{i}.pub',pub+b'\n');keys.append(pub.decode())
with (out/'allowed_nodes.csv').open('w',newline='') as f:csv.writer(f).writerow(keys)
print('Pilot keys generated. Transfer only each node\'s own key pair and ca.crt to that node.')
