#!/usr/bin/env python3
"""Run every panel query of the provisioned dashboards through Grafana's
/api/ds/query and report which return data. Read-only.

  set -a; . ./.env; set +a; python3 tools/check_dashboards.py [range_seconds]
"""
import base64
import json
import os
import sys
import time
import urllib.request

URL = f"http://127.0.0.1:{os.environ.get('GRAFANA_PORT', '13000')}"
AUTH = base64.b64encode(
    f"{os.environ.get('GRAFANA_ADMIN_USER', 'admin')}:{os.environ['GRAFANA_ADMIN_PASSWORD']}".encode()).decode()
RANGE = int(sys.argv[1]) if len(sys.argv) > 1 else 3 * 3600


def api(path, body=None):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": "Basic " + AUTH,
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def subst(expr):
    for k, v in {"$server": ".*", "$instance": ".*", "$role": ".*", "$model_name": ".*",
                 "$__rate_interval": "1m", "$__interval": "1m", "$__range": f"{RANGE}s",
                 "$agg_method": "0.5"}.items():
        expr = expr.replace(k, v)
    return expr


def main():
    now = int(time.time() * 1000)
    bad = total = 0
    for d in api("/api/search?type=dash-db&tag=vllm"):
        dash = api(f"/api/dashboards/uid/{d['uid']}")["dashboard"]
        print(f"## {dash['title']}")
        panels = []
        for p in dash.get("panels", []):
            panels += [p] + p.get("panels", [])
        for p in panels:
            for t in p.get("targets", []):
                if not t.get("expr"):
                    continue
                total += 1
                q = {"refId": "A", "datasource": {"type": "prometheus", "uid": "prometheus"},
                     "expr": subst(t["expr"]), "instant": bool(t.get("instant")),
                     "range": not t.get("instant"), "intervalMs": 60000, "maxDataPoints": 200}
                r = api("/api/ds/query", {"queries": [q], "from": str(now - RANGE * 1000), "to": str(now)})
                res = r["results"]["A"]
                n = sum(len(f["data"]["values"][0]) if f["data"]["values"] else 0 for f in res.get("frames", []))
                series = len(res.get("frames", []))
                status = "ok " if n else ("ERR" if res.get("error") else "-- ")
                bad += status != "ok "
                print(f"  {status} {series:3d} series {p.get('title', '')[:48]:48s} {t.get('legendFormat', '')[:30]}"
                      + (f" {res.get('error')}" if res.get("error") else ""))
    print(f"{total - bad}/{total} queries returned data")


if __name__ == "__main__":
    main()
