"""Experimental v13: v12 plus uncertainty-aware measured-arm comparison.
Decayed centered reward residuals estimate ratio standard error, with a1token
variance floor and one-SE gate. Heuristic, not valid iid confidence guarantee.
Fresh noisy arms cannot instantly dominate/reject based on one sample.
Historical v12: fixed3% gain band instead of5%, no optimistic tail.
Measured real feedback gate and current-arm calibration retained. Explicit
engineering tolerance; no claim of confidence or measured switching cost.
Historical v4: v2 plus explicit cold tie and steady gain thresholds.
Cold first trim prefers widest within2% of best; later requires5% improvement.
These are engineering tolerances, not confidence guarantees or measured cost.
Experimental v2 (API v1): comparable prediction scale + realized-width gate.
Current measured output/time per block calibrates ALL current predictions.
This changes v1: if the current arm has no evidence, it is pure model; if it
has evidence, predictions use its observed level and model width ratios.
Per-arm old evidence still decays into the current calibrated prediction.
Choose only after valid recurrence feedback for the currently selected width;
old in-flight or transition-only feedback updates statistics but cannot switch.
CPU-only, no IO, no timers, no hard warmup, no score hysteresis, no task labels.
Not an asserted optimal policy. Scalar calibration cannot correct all k-specific
counterfactual bias. Input timing is CPU-visible recurrence, NOT GPU-only time.
"""
API_VERSION=1
COST_MS={1:(23.3363,25.1028,26.8228,29.0576,30.0880),2:(24.3722,26.0088,27.7099,29.8630,30.9034),4:(27.7395,29.2079,30.3102,32.4898,33.6713),8:(31.1514,34.9460,39.0532,41.8467,44.5077),16:(50.8700,54.9209,58.4146,61.6853,64.1886),32:(64.8606,69.6108,74.2044,80.4401,83.9715)}

def costs(c):
    cs=sorted(COST_MS)
    if c<=cs[0]:return [v/1000 for v in COST_MS[cs[0]]]
    for lo,hi in zip(cs,cs[1:]):
        if c<=hi:
            w=(c-lo)/(hi-lo)
            return [(a*(1-w)+b*w)/1000 for a,b in zip(COST_MS[lo],COST_MS[hi])]
    return [v/1000 for v in COST_MS[cs[-1]]]

class Policy:
    def __init__(self,config):
        self.decay=float(config.get('decay',.95));self.w=float(config.get('prediction_weight',1.))
        if not 0<=self.decay<1 or not 0<self.w<=100:raise ValueError('invalid config')
        self.k=5;self.n=[0.]*5;self.y=[0.]*5;self.t=[0.]*5
        self.risk=[0.]*5;self.success=[0.]*5;self.prior=1.;self.epoch=None
        self.diagnostics={};self.latest=None;self.trimmed=False
        self.q=[0.]*5;self.y2=[0.]*5;self.t2=[0.]*5;self.yt=[0.]*5
    def observe(self,e):
        k=e['k'];a=e['accepted'];dt=e['seconds']
        if not 1<=k<=5 or not 0<=a<=k:raise ValueError('invalid feedback')
        self.latest=e.copy()
        self.prior*=self.decay
        if self.epoch!=e['load_epoch']:
            self.n=[0.]*5;self.y=[0.]*5;self.t=[0.]*5;self.epoch=e['load_epoch']
            self.q=[0.]*5;self.y2=[0.]*5;self.t2=[0.]*5;self.yt=[0.]*5
        for j in range(5):
            self.n[j]*=self.decay;self.y[j]*=self.decay;self.t[j]*=self.decay
            self.q[j]*=self.decay**2;self.y2[j]*=self.decay;self.t2[j]*=self.decay;self.yt[j]*=self.decay
            self.risk[j]*=self.decay;self.success[j]*=self.decay
            if j<k:self.risk[j]+=1;self.success[j]+=a>j
        if e['timing_valid']:
            j=k-1;y=e['output_tokens']
            self.n[j]+=1;self.y[j]+=y;self.t[j]+=dt
            self.q[j]+=1;self.y2[j]+=y*y;self.t2[j]+=dt*dt;self.yt[j]+=y*dt
    def choose(self,context):
        self.k=context['current_k'];p=[];previous=1.
        for n,s in zip(self.risk,self.success):
            previous=min(previous,(s+self.prior+.5*previous)/(n+self.prior+1));p.append(previous)
        E=[1+sum(p[:k]) for k in range(1,6)];C=costs(context['concurrency'])
        # Calibrate predictions to the SAME current observed level. A bad
        # current block cannot punish only the measured arm while unseen arms
        # retain their optimistic absolute level. Preserve predicted k ratios.
        i=self.k-1
        scale=self.y[i]/self.n[i]/E[i] if self.n[i] else 1.
        time_scale=self.t[i]/self.n[i]/C[i] if self.n[i] else 1.
        predicted_y=[e*scale for e in E]
        predicted_t=[c*time_scale for c in C]
        scores=[(y+self.w*e)/(t+self.w*c) for y,t,e,c in zip(self.y,self.t,predicted_y,predicted_t)]
        # Estimated SE of ratio-of-totals using centered y-rate*t residuals.
        # Kish ESS alone does not lose confidence when an arm ages; cap by
        # decayed mass so stale evidence really loses influence/confidence.
        se=[]
        for j in range(5):
            n=self.n[j]
            if n<=0:se.append(0.);continue
            rate=self.y[j]/self.t[j]
            variance=max(0.,(self.y2[j]-2*rate*self.yt[j]+rate*rate*self.t2[j])/n)
            ess=min(n,n*n/self.q[j])
            # nonzero floor prevents a single constant sample claiming certainty
            se.append((variance+1.)**.5/(self.t[j]/n)/(ess+1)**.5)
        fractions=[t/(t+self.w*c) for t,c in zip(self.t,predicted_t)]
        # All model-only predictions share incumbent calibration; its common
        # absolute noise cancels for their relative ordering. Charge uncertainty
        # only for candidate's independent actual evidence contribution.
        gate=[f*(s*s+se[self.k-1]**2)**.5 for f,s in zip(fractions,se)]
        ready=bool(self.latest and self.latest['k']==self.k and self.latest['timing_valid'])
        best=self.k
        if ready:
            if not self.trimmed:
                peak=max(scores)
                cold=max(k for k in range(1,6) if scores[k-1]>=.98*peak)
                if scores[cold-1]>scores[best-1]:best=cold
            else:
                for k in range(1,min(5,self.k+1)+1):
                    if scores[k-1]-scores[self.k-1] > .03*scores[self.k-1]+gate[k-1] and scores[k-1]>scores[best-1]:best=k
        if best<5:self.trimmed=True
        self.diagnostics=dict(score=scores,se=se,gate=gate,ready=ready,trimmed=self.trimmed)
        return best

def self_test():
    p=Policy({})
    for _ in range(20):
        p.observe(dict(k=5,accepted=5,seconds=.03,output_tokens=6,timing_valid=True,load_epoch=1))
        assert p.choose(dict(current_k=5,concurrency=1))==5
    p=Policy({});p.observe(dict(k=2,accepted=1,seconds=.025,output_tokens=2,timing_valid=True,load_epoch=1))
    for _ in range(10):p.observe(dict(k=3,accepted=2,seconds=3/90,output_tokens=3,timing_valid=True,load_epoch=1))
    assert abs(p.y[1]/p.t[1]-80)<1e-8
    assert abs(p.n[1]-.95**10)<1e-8
