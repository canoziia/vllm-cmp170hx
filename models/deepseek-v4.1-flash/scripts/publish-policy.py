#!/usr/bin/env python3
"""Run on trusted host. Publish source atomically; never print capability token.
Python policies are trusted executable code, not sandboxed. Pretest separately.
Existing requests retain old version; disable affects NEW admissions only.
"""
import argparse,hashlib,json,os,secrets,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('source',type=Path,nargs='?');p.add_argument('--disable',action='store_true');p.add_argument('--config',default='{}');a=p.parse_args()
a.directory.mkdir(parents=True,exist_ok=True);a.directory.chmod(0o700)
control=a.directory/'control.json';old=json.loads(control.read_text()) if control.exists() else {}
if a.disable:
 data={**old,'enabled':False}
else:
 if a.source is None:p.error('source required')
 src=a.source.read_bytes();compile(src,str(a.source),'exec')
 if len(src)>131072:raise ValueError('source too large')
 digest=hashlib.sha256(src).hexdigest();versions=a.directory/'versions';versions.mkdir(exist_ok=True)
 target=versions/(digest+'.py')
 if target.exists():assert target.read_bytes()==src
 else:
  with target.open('xb') as f:f.write(src)
  target.chmod(0o600)
 data=dict(enabled=True,sha256=digest,token=old.get('token') or secrets.token_hex(32),config=json.loads(a.config))
fd,name=tempfile.mkstemp(prefix='.control-',dir=a.directory)
try:
 with os.fdopen(fd,'w') as f:json.dump(data,f);f.flush();os.fsync(f.fileno())
 os.replace(name,control)
finally:
 if os.path.exists(name):os.unlink(name)
print('disabled new admissions' if a.disable else 'published '+data['sha256'])
