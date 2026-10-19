#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, random, copy
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc, brier_score, integrated_brier_score


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

class Lung1CTDataset(Dataset):
    def __init__(self, df, vol_dir):
        self.df=df.reset_index(drop=True).copy(); self.vol_dir=Path(vol_dir)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r=self.df.iloc[i]; pid=str(r.PatientID)
        p=self.vol_dir/f"{pid}_standardized_192x160x224.npz"
        if not p.exists(): raise FileNotFoundError(p)
        z=np.load(p,allow_pickle=False); ct=np.asarray(z['ct_hu_clipped'],np.float32)
        ct=np.clip((ct+1000.)/1400.,0.,1.)
        return dict(ct=torch.from_numpy(ct[None]).float(),
                    time=torch.tensor(float(r['Survival.time']),dtype=torch.float32),
                    event=torch.tensor(float(r['deadstatus.event']),dtype=torch.float32),
                    patient_id=pid)

class ConvBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__(); self.net=nn.Sequential(
            nn.Conv3d(cin,cout,3,2,1,bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
            nn.Conv3d(cout,cout,3,1,1,bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)

class CTOnlyNet(nn.Module):
    def __init__(self,dropout=.30):
        super().__init__(); self.encoder=nn.Sequential(
            ConvBlock(1,8),ConvBlock(8,16),ConvBlock(16,32),ConvBlock(32,64),ConvBlock(64,96),nn.AdaptiveAvgPool3d(1))
        self.head=nn.Sequential(nn.Flatten(),nn.Dropout(dropout),nn.Linear(96,32),nn.ReLU(inplace=True),nn.Dropout(dropout),nn.Linear(32,1))
    def forward(self,x): return self.head(self.encoder(x)).squeeze(1)

def cox_efron_full(risk,time,event):
    risk=risk.reshape(-1); time=time.reshape(-1); event=event.reshape(-1)
    obs=event>0.5; nev=int(obs.sum().item())
    if nev==0: return risk.sum()*0.
    ll=risk.new_tensor(0.)
    for t in torch.unique(time[obs]):
        deaths=obs & (time==t); d=int(deaths.sum().item()); rs=time>=t
        rr=risk[rs]; rd=risk[deaths]; m=rr.max(); sr=torch.exp(rr-m).sum(); sd=torch.exp(rd-m).sum()
        ll=ll+rd.sum()
        for l in range(d):
            den=sr-(float(l)/d)*sd
            ll=ll-(m+torch.log(torch.clamp(den,min=torch.finfo(risk.dtype).tiny)))
    return -ll/float(nev)

def snap_bn(model):
    out={}
    for n,m in model.named_modules():
        if isinstance(m,nn.modules.batchnorm._BatchNorm):
            out[n]=(None if m.running_mean is None else m.running_mean.detach().clone(),
                    None if m.running_var is None else m.running_var.detach().clone(),
                    None if m.num_batches_tracked is None else m.num_batches_tracked.detach().clone())
    return out

def restore_bn(model,state):
    mods=dict(model.named_modules())
    for n,(rm,rv,nb) in state.items():
        m=mods[n]
        if rm is not None: m.running_mean.copy_(rm)
        if rv is not None: m.running_var.copy_(rv)
        if nb is not None: m.num_batches_tracked.copy_(nb)

def get_rng(dev):
    return (torch.get_rng_state().clone(), None if dev.type!='cuda' else torch.cuda.get_rng_state(dev).clone())
def set_rng(st,dev):
    torch.set_rng_state(st[0])
    if dev.type=='cuda' and st[1] is not None: torch.cuda.set_rng_state(st[1],dev)

def train_epoch_fullcox(model,ds,opt,dev,bs,nw,scaler,epoch_seed):
    model.train(); g=torch.Generator(); g.manual_seed(epoch_seed); perm=torch.randperm(len(ds),generator=g).tolist(); sub=Subset(ds,perm)
    def loader(): return DataLoader(sub,batch_size=bs,shuffle=False,num_workers=nw,pin_memory=(dev.type=='cuda'))
    bn0=snap_bn(model); risks=[]; times=[]; events=[]; rngs=[]; sizes=[]
    with torch.no_grad():
        for b in loader():
            ct=b['ct'].to(dev,non_blocking=(dev.type=='cuda')); rngs.append(get_rng(dev))
            with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=(dev.type=='cuda')): r=model(ct)
            risks.append(r.detach().float().cpu()); times.append(b['time'].double()); events.append(b['event'].double()); sizes.append(len(ct))
    rcpu=torch.cat(risks).double(); tcpu=torch.cat(times).double(); ecpu=torch.cat(events).double()
    leaf=rcpu.clone().requires_grad_(True); loss=cox_efron_full(leaf,tcpu,ecpu); loss.backward(); gr=leaf.grad.detach().float()
    restore_bn(model,bn0); opt.zero_grad(set_to_none=True); off=0; errs=[]
    for bi,b in enumerate(loader()):
        ct=b['ct'].to(dev,non_blocking=(dev.type=='cuda')); set_rng(rngs[bi],dev)
        with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=(dev.type=='cuda')): r=model(ct)
        n=sizes[bi]; target=rcpu[off:off+n].float().to(dev); errs.append(float((r.detach().float()-target).abs().max().cpu()))
        grad=gr[off:off+n].to(dev,dtype=r.dtype)
        if scaler.is_enabled(): scaler.scale(r).backward(gradient=grad)
        else: r.backward(gradient=grad)
        off+=n
    err=max(errs) if errs else 0.
    if err>2e-4: raise RuntimeError(f"Two-pass replay mismatch {err:.3g}")
    if scaler.is_enabled(): scaler.step(opt); scaler.update()
    else: opt.step()
    return float(loss.detach()),err

@torch.no_grad()
def predict(model,ds,dev,bs,nw):
    model.eval(); rows=[]
    dl=DataLoader(ds,batch_size=min(bs,4),shuffle=False,num_workers=nw,pin_memory=(dev.type=='cuda'))
    for b in dl:
        r=model(b['ct'].to(dev,non_blocking=(dev.type=='cuda'))).float().cpu().numpy()
        for pid,t,e,x in zip(b['patient_id'],b['time'].numpy(),b['event'].numpy(),r): rows.append([pid,float(t),int(e),float(x)])
    return pd.DataFrame(rows,columns=['PatientID','Survival.time','deadstatus.event','risk_score'])

def cindex(df): return float(concordance_index(df['Survival.time'].to_numpy(float),-df['risk_score'].to_numpy(float),event_observed=df['deadstatus.event'].to_numpy(int)))
def surv_arr(df):
    y=np.empty(len(df),dtype=[('event','?'),('time','<f8')]); y['event']=df['deadstatus.event'].astype(bool); y['time']=df['Survival.time'].astype(float); return y

def fit_breslow(df):
    t=df['Survival.time'].to_numpy(float); e=df['deadstatus.event'].to_numpy(int); r=df['risk_score'].to_numpy(float); shift=float(r.max()); er=np.exp(r-shift)
    et=np.sort(np.unique(t[e==1])); inc=[]
    for x in et:
        d=np.sum((t==x)&(e==1)); den=np.sum(er[t>=x]); inc.append(d/den if den>0 else 0.)
    return {'event_times':et,'cum_hazard':np.cumsum(np.asarray(inc,float)),'shift':shift}
def surv_mat(df,times,b):
    h=[]
    for t in times:
        p=np.searchsorted(b['event_times'],t,side='right')-1; h.append(b['cum_hazard'][p] if p>=0 else 0.)
    rel=np.exp(df['risk_score'].to_numpy(float)-b['shift']); return np.exp(-rel[:,None]*np.asarray(h)[None,:])
def metrics(train,ev,times=(365,730,1095)):
    lo=max(train['Survival.time'].min(),ev['Survival.time'].min()); hi=min(train['Survival.time'].max(),ev['Survival.time'].max()); tt=np.asarray([t for t in times if lo<t<hi],float)
    yt,ye=surv_arr(train),surv_arr(ev); risk=ev['risk_score'].to_numpy(float); aucs,ma=cumulative_dynamic_auc(yt,ye,risk,tt); s=surv_mat(ev,tt,fit_breslow(train)); _,bs=brier_score(yt,ye,s,tt); ibs=float(integrated_brier_score(yt,ye,s,tt))
    return {'C_index':cindex(ev),'Mean_dynamic_AUC':float(ma),'Integrated_Brier_Score':ibs},pd.DataFrame({'time_days':tt.astype(int),'AUC_dynamic':aucs,'Brier_score':bs})
def boot_c(df,B,seed):
    rng=np.random.default_rng(seed); vals=[]; n=len(df)
    for _ in range(B):
        b=df.iloc[rng.integers(0,n,n)]
        if b['deadstatus.event'].sum()==0 or b['Survival.time'].nunique()<2: continue
        try: vals.append(cindex(b))
        except: pass
    v=np.asarray(vals,float); return len(v),float(np.percentile(v,2.5)),float(np.percentile(v,97.5))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--vol-dir',required=True); ap.add_argument('--split',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--epochs',type=int,default=60); ap.add_argument('--batch-size',type=int,default=4); ap.add_argument('--num-workers',type=int,default=2); ap.add_argument('--lr',type=float,default=1e-4); ap.add_argument('--weight-decay',type=float,default=1e-4); ap.add_argument('--dropout',type=float,default=.30); ap.add_argument('--patience',type=int,default=12); ap.add_argument('--bootstrap',type=int,default=5000); ap.add_argument('--seed',type=int,default=20260926); args=ap.parse_args()
    seed_all(args.seed); out=Path(args.out); out.mkdir(parents=True,exist_ok=True); sp=pd.read_csv(args.split); sp['Split_norm']=sp['Split'].astype(str).str.lower()
    parts={k:sp[sp['Split_norm']==k][['PatientID','Survival.time','deadstatus.event']].copy() for k in ['train','validation','test']}; exp={'train':295,'validation':63,'test':64}
    for k,n in exp.items():
        if len(parts[k])!=n: raise RuntimeError(f"{k}: expected {n}, got {len(parts[k])}")
    ds={k:Lung1CTDataset(parts[k],args.vol_dir) for k in parts}; dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Device:',dev); print('Split:',{k:len(v) for k,v in ds.items()}); print('Objective: FULL training-cohort Cox partial likelihood, Efron ties; micro-batch only for memory.')
    model=CTOnlyNet(args.dropout).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=4,min_lr=1e-6); scaler=torch.amp.GradScaler('cuda',enabled=(dev.type=='cuda'))
    best=-np.inf; best_ep=-1; wait=0; hist=[]; bestp=out/'best_model.pt'
    for ep in range(1,args.epochs+1):
        loss,err=train_epoch_fullcox(model,ds['train'],opt,dev,args.batch_size,args.num_workers,scaler,args.seed+1000*ep); tr=predict(model,ds['train'],dev,args.batch_size,args.num_workers); va=predict(model,ds['validation'],dev,args.batch_size,args.num_workers); tc,vc=cindex(tr),cindex(va); sch.step(vc); hist.append({'epoch':ep,'full_cohort_cox_efron_loss':loss,'max_two_pass_risk_replay_error':err,'train_C_index':tc,'validation_C_index':vc,'lr':opt.param_groups[0]['lr']}); print(f"Epoch {ep:03d} | Cox-Efron={loss:.5f} | replay={err:.2e} | train C={tc:.4f} | val C={vc:.4f}")
        if vc>best+1e-4:
            best=vc; best_ep=ep; wait=0; torch.save({'model_state_dict':model.state_dict(),'epoch':ep,'validation_C_index':vc,'training_objective':'full_training_cohort_cox_partial_likelihood','tie_method':'Efron','gradient_method':'two_pass_exact_full_cohort_risk_gradient','micro_batch_size':args.batch_size,'args':vars(args)},bestp)
        else: wait+=1
        if wait>=args.patience: print(f"Early stop: best epoch={best_ep}, val C={best:.4f}"); break
    pd.DataFrame(hist).to_csv(out/'training_history.csv',index=False,encoding='utf-8-sig'); ck=torch.load(bestp,map_location=dev,weights_only=False); model.load_state_dict(ck['model_state_dict']); pred={k:predict(model,ds[k],dev,args.batch_size,args.num_workers) for k in ds}
    for k,d in pred.items(): d.to_csv(out/f'{k}_risk_scores.csv',index=False,encoding='utf-8-sig')
    mrows=[]; hrows=[]
    for i,k in enumerate(['validation','test']):
        m,h=metrics(pred['train'],pred[k]); B,lo,hi=boot_c(pred[k],args.bootstrap,args.seed+10000*(i+1)); mrows.append({'Subset':k,**m,'C_index_CI95_lower':lo,'C_index_CI95_upper':hi,'Bootstrap_replicates':B}); h.insert(0,'Subset',k); hrows.append(h)
    ms=pd.DataFrame(mrows); hs=pd.concat(hrows,ignore_index=True); ms.to_csv(out/'metrics_summary.csv',index=False,encoding='utf-8-sig'); hs.to_csv(out/'per_horizon_metrics.csv',index=False,encoding='utf-8-sig')
    lines=['LUNG1 CT-only 3D CNN - FULL COX-EFRON CORRECTION','',f'Device: {dev}',f'Best epoch: {best_ep}',f'Best validation C-index: {best:.4f}','Input: 192x160x224 CT','Training objective: full training-cohort Cox partial likelihood','Ties: Efron','Memory: two-pass exact dLoss/dRisk replay','Micro-batches do NOT define Cox risk sets','',f'TRAIN C-index: {cindex(pred["train"]):.4f}']
    for _,r in ms.iterrows():
        lines += ['',r['Subset'].upper()+':',f"  C-index={r.C_index:.4f} (95% CI [{r.C_index_CI95_lower:.4f}, {r.C_index_CI95_upper:.4f}])",f"  Mean dynamic AUC={r.Mean_dynamic_AUC:.4f}",f"  IBS={r.Integrated_Brier_Score:.4f}"]
        for _,q in hs[hs['Subset']==r['Subset']].iterrows(): lines.append(f"  {int(q.time_days)} days: AUC={q.AUC_dynamic:.4f}, Brier={q.Brier_score:.4f}")
    lines += ['','Validation alone used for early stopping/model selection.','Test evaluated only after best validation checkpoint was fixed.']; txt='\n'.join(lines); (out/'CT_Only_FullCox_Efron_Summary.txt').write_text(txt,encoding='utf-8'); print('\n'+txt); print('Files:',out)
if __name__=='__main__': main()
