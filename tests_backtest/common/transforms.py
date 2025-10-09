import numpy as np
def raw_features(series):
    x = np.asarray(series, dtype=float); r = np.diff(np.log(x))
    return np.array([r.mean(), r.std(), r[-1], r.max()-r.min()])
def fft_features(series, top_n=16):
    x = np.asarray(series, dtype=float); x=(x-x.mean())/(x.std()+1e-12)
    spec=np.fft.rfft(x); amps=np.abs(spec); phases=np.angle(spec)
    idx=np.argsort(amps)[::-1][:top_n]; return np.concatenate([amps[idx], phases[idx]])
def haar_dwt_level(x):
    x=np.asarray(x,dtype=float); 
    if len(x)%2==1: x=x[:-1]
    a=(x[0::2]+x[1::2])/2.0; d=(x[0::2]-x[1::2])/2.0; return a,d
def wavelet_features(series, levels=3):
    a=np.asarray(series,dtype=float); feats=[]
    for _ in range(levels):
        a,d=haar_dwt_level(a); feats += [a.mean(),a.std(),d.mean(),d.std(),(d**2).sum()]
    return np.asarray(feats)
class ThetaBasis:
    def __init__(self,K=16,tau_im=0.25): self.K=K; self.tau_im=tau_im; self.phases=np.linspace(0,1,K,endpoint=False)
    def _theta3(self,t_over_T,q):
        n=np.arange(-20,21); return np.sum(q**(n**2)*np.cos(2*np.pi*np.outer(t_over_T,n)),axis=1)
    def design(self,T):
        q=np.exp(-np.pi*self.tau_im); t_over_T=np.linspace(0,1,T,endpoint=False)
        mats=[ self._theta3((t_over_T+phi)%1.0,q) for phi in self.phases ]
        Phi=np.stack(mats,axis=1); Phi=(Phi-Phi.mean(0))/(Phi.std(0)+1e-8); return Phi
def theta_features(series,K=16,tau_im=0.25,gram=False):
    s=np.asarray(series,dtype=float); s=(s-s.mean())/(s.std()+1e-12); T=len(s); tb=ThetaBasis(K=K,tau_im=tau_im); Phi=tb.design(T)
    if gram:
        Q=np.zeros_like(Phi)
        for i in range(Phi.shape[1]):
            v=Phi[:,i].copy()
            for j in range(i): v -= np.dot(Q[:,j],Phi[:,i])*Q[:,j]
            Q[:,i]=v/(np.linalg.norm(v)+1e-12)
        Phi=Q
    a,*_=np.linalg.lstsq(Phi,s,rcond=None); recon=Phi@a; resid=s-recon
    feats=np.concatenate([a,[resid.mean(),resid.std(),(resid**2).sum()]]); return feats
