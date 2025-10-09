import numpy as np
def train_logreg(X,y,epochs=200,lr=0.1):
    w=np.zeros(X.shape[1]); b=0.0
    for _ in range(epochs):
        z=X@w + b; p=1/(1+np.exp(-z))
        grad_w=X.T@(p-y)/len(y); grad_b=float((p-y).mean())
        w-=lr*grad_w; b-=lr*grad_b
    return w,b
def predict_logreg(X,w,b):
    z=X@w + b; return 1/(1+np.exp(-z))
class Kalman1D:
    def __init__(self,q=1e-5,r=1e-3): self.q=q; self.r=r
    def fit_predict_proba(self, returns):
        x=0.0; p=1.0; probs=[]
        for z in returns:
            x_pred=x; p_pred=p+self.q
            k=p_pred/(p_pred+self.r)
            x=x_pred + k*(z - x_pred); p=(1-k)*p_pred
            probs.append(1/(1+np.exp(-x*100)))
        return np.array(probs)
def torch_lstm_available():
    try:
        import torch
        return True
    except Exception:
        return False
def fit_lstm_and_predict(X_train, y_train, X_test, epochs=5):
    import torch, torch.nn as nn, numpy as np
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); in_dim=X_train.shape[1]
    class Net(nn.Module):
        def __init__(self): super().__init__(); self.rnn=nn.LSTM(in_dim,32,1,batch_first=True); self.fc=nn.Linear(32,1)
        def forward(self,x): out,_=self.rnn(x); out=out[:,-1,:]; return self.fc(out)
    net=Net().to(device); opt=torch.optim.Adam(net.parameters(),lr=1e-3); loss_fn=nn.BCEWithLogitsLoss()
    def to_seq(X,y):
        Xs; Ys = [], []
        for i in range(2,len(X)):
            Xs.append(np.stack([X[i-2],X[i-1],X[i]],axis=0)); Ys.append(y[i])
        return np.array(Xs), np.array(Ys)
    Xt, yt = to_seq(X_train, y_train)
    Xv, _  = to_seq(X_test, np.zeros(len(X_test)))
    Xt=torch.tensor(Xt,dtype=torch.float32,device=device); yt=torch.tensor(yt,dtype=torch.float32,device=device).unsqueeze(1)
    for _ in range(epochs):
        net.train(); opt.zero_grad(); out=net(Xt); loss=loss_fn(out,yt); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        Xv=torch.tensor(Xv,dtype=torch.float32,device=device); out=net(Xv).cpu().numpy().reshape(-1)
        probs=1/(1+np.exp(-out))
    pad=[0.5,0.5]; return np.array(pad + list(probs))
