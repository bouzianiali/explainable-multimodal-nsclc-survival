#!/usr/bin/env python
"""Public release trainer for the gated-residual A/B/C experiment.

This release mirrors the locked manuscript protocol: exact full-training-cohort
Cox partial likelihood with Efron ties, deterministic two-pass network replay,
whole-clinical ModDrop, CT-teacher anchor, five-epoch CT freeze, and tuning-only
checkpoint selection. It imports common survival/data utilities from the bundled
missingness-aware concatenation script to keep the public implementation concise.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE.parent/'src'))
import train_standard_moddrop_missingaware_fullcox_efron as common
from nsclc_survival.models import CTOnlyTeacher, GatedResidualSurvival

def load_teacher_student(teacher,student,path):
    ckpt=torch.load(path,map_location='cpu',weights_only=False); state=ckpt['model_state_dict']
    teacher.load_state_dict(state,strict=True)
    mapped={}
    for k,v in state.items():
        if k.startswith('encoder.'): mapped['ct_encoder.'+k[len('encoder.'):]]=v
        elif k.startswith('head.'): mapped['ct_risk_head.'+k[len('head.'):]]=v
    student.load_state_dict(mapped,strict=False)

def freeze_all_ct(m):
    for p in m.ct_encoder.parameters(): p.requires_grad=False
    for p in m.ct_risk_head.parameters(): p.requires_grad=False

def unfreeze_last_ct(m):
    for p in m.ct_encoder.parameters(): p.requires_grad=False
    for p in m.ct_encoder[4].parameters(): p.requires_grad=True
    for p in m.ct_risk_head.parameters(): p.requires_grad=False

def build_optimizer(m,fusion_lr,ct_lr,wd,phase2):
    fusion=list(m.ct_feature_proj.parameters())+list(m.clinical_encoder.parameters())+list(m.gate.parameters())+list(m.residual_head.parameters())
    groups=[{'params':[p for p in fusion if p.requires_grad],'lr':fusion_lr}]
    if phase2: groups.append({'params':[p for p in m.ct_encoder[4].parameters() if p.requires_grad],'lr':ct_lr})
    return torch.optim.AdamW(groups,weight_decay=wd)

def availability_from_mask(mask,device): return (~torch.as_tensor(mask,dtype=torch.bool,device=device)).float().reshape(-1,1)

def train_epoch(student,teacher,dataset,optimizer,scaler,device,amp,batch_size,num_workers,epoch_seed,moddrop,anchor_lambda):
    student.train(); teacher.eval()
    gen=torch.Generator(); gen.manual_seed(int(epoch_seed)); perm=torch.randperm(len(dataset),generator=gen).tolist(); ds=Subset(dataset,perm)
    miss=common.exact_missing_mask(len(dataset),moddrop,int(epoch_seed)+7919)
    def loader(): return DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=device.type=='cuda',drop_last=False)
    bn=common.snapshot_bn_state(student); risks=[]; times=[]; events=[]; states=[]; sizes=[]; offset=0
    with torch.no_grad():
        for batch in loader():
            ct=batch['ct'].to(device); cl=batch['clinical'].to(device); bs=len(ct); a=availability_from_mask(miss[offset:offset+bs],device)
            states.append(common.capture_rng_state(device))
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp): r,_,_,_=student(ct,cl,a)
            risks.append(r.detach().float().cpu()); times.append(batch['time'].double().cpu()); events.append(batch['event'].double().cpu()); sizes.append(bs); offset+=bs
    rcpu=torch.cat(risks).double(); tcpu=torch.cat(times).double(); ecpu=torch.cat(events).double(); leaf=rcpu.clone().requires_grad_(True)
    loss=common.cox_efron_loss_full(leaf,tcpu,ecpu); loss.backward(); grad=leaf.grad.detach().float()
    common.restore_bn_state(student,bn); optimizer.zero_grad(set_to_none=True); offset=0; replay=[]; anchor_sum=0.0
    for bi,batch in enumerate(loader()):
        ct=batch['ct'].to(device); cl=batch['clinical'].to(device); bs=sizes[bi]; a=availability_from_mask(miss[offset:offset+bs],device); common.restore_rng_state(states[bi],device)
        with torch.no_grad(): tr=teacher(ct)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp): r,base,_,_=student(ct,cl,a); al=((base-tr)**2).sum()/len(dataset)
        ref=rcpu[offset:offset+bs].float().to(device); replay.append(float(torch.max(torch.abs(r.detach().float()-ref)).cpu()))
        g=grad[offset:offset+bs].to(device=device,dtype=r.dtype)
        if scaler.is_enabled():
            scaler.scale(r).backward(gradient=g,retain_graph=(anchor_lambda!=0));
            if anchor_lambda!=0: scaler.scale(anchor_lambda*al).backward()
        else:
            r.backward(gradient=g,retain_graph=(anchor_lambda!=0));
            if anchor_lambda!=0: (anchor_lambda*al).backward()
        anchor_sum+=float(al.detach().cpu()); offset+=bs
    if max(replay,default=0)>2e-4: raise RuntimeError(f'two-pass replay mismatch {max(replay)}')
    if scaler.is_enabled(): scaler.step(optimizer); scaler.update()
    else: optimizer.step()
    return {'full_cohort_cox_efron_loss':float(loss.detach()),'max_two_pass_risk_replay_error':max(replay,default=0.0),'anchor_loss':anchor_sum,'moddrop_missing_fraction':float(miss.mean())}

@torch.no_grad()
def predict(model,ds,device,batch_size,num_workers,mask):
    ld=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=device.type=='cuda'); rows=[]; off=0; model.eval()
    mask=np.asarray(mask,bool)
    for b in ld:
        ct=b['ct'].to(device); cl=b['clinical'].to(device); bs=len(ct); a=availability_from_mask(mask[off:off+bs],device); r,_,_,_=model(ct,cl,a); r=r.cpu().numpy()
        rows.extend((pid,float(t),int(e),float(x)) for pid,t,e,x in zip(b['patient_id'],b['time'].numpy(),b['event'].numpy(),r)); off+=bs
    return pd.DataFrame(rows,columns=['PatientID','Survival.time','deadstatus.event','risk_score'])

def scenario_rows(model,train_ds,eval_ds,subset,device,args):
    rows=[]
    for frac in [0.,.1,.3,.5,1.]:
        pct=int(round(frac*100)); nr=1 if pct in (0,100) else args.scenario_repeats
        for rep in range(nr):
            tm=common.exact_missing_mask(len(train_ds),frac,args.seed+100000+1000*pct+rep); em=common.exact_missing_mask(len(eval_ds),frac,args.seed+(200000 if subset=='test' else 300000)+1000*pct+rep)
            tr=predict(model,train_ds,device,min(args.batch_size,4),args.num_workers,tm); ev=predict(model,eval_ds,device,min(args.batch_size,4),args.num_workers,em)
            c=common.harrell_cindex(ev); base=common.fit_breslow_baseline(tr); _,auc,ibs=common.time_dependent_metrics(tr,ev,base,args.times)
            rows.append({'Subset':subset,'Missing_percent':pct,'Repeat':rep,'C_index':c,'Mean_dynamic_AUC':auc,'Integrated_Brier_Score':ibs})
    raw=pd.DataFrame(rows); summ=common.summarize_scenario_metrics(raw); summ.insert(0,'Subset',subset); base0=float(summ.loc[summ.Missing_percent==0,'C_index_mean'].iloc[0]); summ['Delta_C_index_vs_0pct']=summ.C_index_mean-base0
    return raw,summ

def main():
    ap=argparse.ArgumentParser();
    for x in ['vol-dir','split','clinical-dir','ct-checkpoint','out']: ap.add_argument('--'+x,required=True)
    ap.add_argument('--epochs',type=int,default=60); ap.add_argument('--freeze-epochs',type=int,default=5); ap.add_argument('--batch-size',type=int,default=2); ap.add_argument('--num-workers',type=int,default=2)
    ap.add_argument('--fusion-lr',type=float,default=1e-4); ap.add_argument('--ct-lr',type=float,default=2e-6); ap.add_argument('--weight-decay',type=float,default=1e-4); ap.add_argument('--dropout',type=float,default=.30)
    ap.add_argument('--clinical-moddrop',type=float,default=.25); ap.add_argument('--anchor-lambda',type=float,default=1.); ap.add_argument('--patience',type=int,default=12); ap.add_argument('--scenario-repeats',type=int,default=500); ap.add_argument('--seed',type=int,default=20260926); ap.add_argument('--times',type=int,nargs='+',default=[365,730,1095]); a=ap.parse_args()
    common.set_seed(a.seed); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); split=pd.read_csv(a.split); cd=Path(a.clinical_dir)
    trc,cols=common.load_clinical_subset(cd,'train'); vac,_=common.load_clinical_subset(cd,'validation',cols); common.validate_tnm_corrected_features(cols)
    tr=common.MultimodalLung1Dataset(split[split.Split.str.lower()=='train'],trc,cols,Path(a.vol_dir)); va=common.MultimodalLung1Dataset(split[split.Split.astype(str).str.lower().isin(['validation','tuning'])],vac,cols,Path(a.vol_dir))
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); student=GatedResidualSurvival(len(cols),a.dropout); teacher=CTOnlyTeacher(a.dropout); load_teacher_student(teacher,student,Path(a.ct_checkpoint)); teacher=teacher.to(dev).eval(); student=student.to(dev); [setattr(p,'requires_grad',False) for p in teacher.parameters()]; freeze_all_ct(student)
    opt=build_optimizer(student,a.fusion_lr,a.ct_lr,a.weight_decay,False); amp=dev.type=='cuda'; scaler=torch.amp.GradScaler('cuda',enabled=amp)
    valm={p:common.exact_missing_mask(len(va),p/100,a.seed+30000+p) for p in (0,30,100)}; best=-1e9; be=-1; no=0; hist=[]
    for ep in range(1,a.epochs+1):
        if ep==a.freeze_epochs+1: unfreeze_last_ct(student); opt=build_optimizer(student,a.fusion_lr,a.ct_lr,a.weight_decay,True); scaler=torch.amp.GradScaler('cuda',enabled=amp)
        info=train_epoch(student,teacher,tr,opt,scaler,dev,amp,a.batch_size,a.num_workers,a.seed+1000*ep,a.clinical_moddrop,a.anchor_lambda)
        vcs={p:common.harrell_cindex(predict(student,va,dev,min(a.batch_size,4),a.num_workers,valm[p])) for p in (0,30,100)}; robust=float(np.mean(list(vcs.values())))
        hist.append({'epoch':ep,**info,**{f'validation_c_index_{p}pct':vcs[p] for p in vcs},'validation_robust_score':robust})
        print(f"Epoch {ep:03d} | phase={1 if ep<=a.freeze_epochs else 2} | Cox-Efron={info['full_cohort_cox_efron_loss']:.5f} | anchor={info['anchor_loss']:.6f} | replay={info['max_two_pass_risk_replay_error']:.2e} | val C [0/30/100%]={vcs[0]:.4f}/{vcs[30]:.4f}/{vcs[100]:.4f} | robust={robust:.4f}")
        if robust>best+1e-4:
            best=robust; be=ep; no=0; torch.save({'model_state_dict':student.state_dict(),'epoch':ep,'validation_robust_score':robust,'clinical_feature_cols':cols,'args':vars(a),'training_objective':'full_training_cohort_cox_partial_likelihood','tie_method':'Efron','gradient_method':'two_pass_exact_full_cohort_risk_gradient'},out/'best_model.pt')
        else: no+=1
        if no>=a.patience: break
    pd.DataFrame(hist).to_csv(out/'training_history.csv',index=False)
    ck=torch.load(out/'best_model.pt',map_location=dev,weights_only=False); student.load_state_dict(ck['model_state_dict'])
    tec,_=common.load_clinical_subset(cd,'test',cols); te=common.MultimodalLung1Dataset(split[split.Split.str.lower()=='test'],tec,cols,Path(a.vol_dir))
    rv,sv=scenario_rows(student,tr,va,'validation',dev,a); rt,st=scenario_rows(student,tr,te,'test',dev,a); pd.concat([sv,st],ignore_index=True).to_csv(out/'Robustness_Scenario_Summary.csv',index=False)
    pd.concat([rv,rt],ignore_index=True).to_csv(out/'Robustness_Raw_Repeats.csv',index=False)
    # deterministic risk-mode files for audits/XAI
    for name,ds in [('train',tr),('validation',va),('test',te)]:
        full=predict(student,ds,dev,min(a.batch_size,4),a.num_workers,np.zeros(len(ds),bool)); none=predict(student,ds,dev,min(a.batch_size,4),a.num_workers,np.ones(len(ds),bool)); z=full.rename(columns={'risk_score':'risk_full_clinical'}); z['risk_no_clinical']=none.risk_score; z.to_csv(out/f'{name}_risk_modes.csv',index=False)
    print('Completed',out,'best epoch',be,'robust',best)
if __name__=='__main__': main()
