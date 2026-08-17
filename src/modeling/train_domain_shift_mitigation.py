"""Run Task 3.13 domain-shift mitigation experiments without test leakage."""
from __future__ import annotations

import argparse, hashlib, json, platform
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np, pandas as pd, sklearn, yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

def resolve(root, value):
    p=Path(value); return p if p.is_absolute() else root/p
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def names(cfg): return [f"{c}__{s}" for c in cfg['channel_names'] for s in cfg['statistics']]
def summarize(a, stats):
    r={'mean':lambda x:np.mean(x,axis=(1,2)),'std':lambda x:np.std(x,axis=(1,2)),'min':lambda x:np.min(x,axis=(1,2)),
       'p10':lambda x:np.percentile(x,10,axis=(1,2)),'p25':lambda x:np.percentile(x,25,axis=(1,2)),
       'median':lambda x:np.median(x,axis=(1,2)),'p75':lambda x:np.percentile(x,75,axis=(1,2)),
       'p90':lambda x:np.percentile(x,90,axis=(1,2)),'max':lambda x:np.max(x,axis=(1,2))}
    bad=set(stats)-set(r)
    if bad: raise ValueError(f'Unsupported statistics: {sorted(bad)}')
    return np.stack([r[s](a) for s in stats],axis=1).reshape(-1).astype(np.float32)
def load_x(frame,root,cfg,path_col,every):
    expected=(len(cfg['channel_names']),int(cfg['expected_patch_size']),int(cfg['expected_patch_size']))
    x=np.empty((len(frame),len(names(cfg))),np.float32)
    for i,v in enumerate(frame[path_col].astype(str),1):
        with np.load(resolve(root,v),allow_pickle=False) as z: a=z[cfg['archive_key']]
        if a.shape!=expected or not np.isfinite(a).all(): raise ValueError(f'Invalid feature patch {v}: {a.shape}')
        x[i-1]=summarize(a,cfg['statistics'])
        if every>0 and (i%every==0 or i==len(frame)): print(f'Feature extraction: {i}/{len(frame)} ({100*i/len(frame):.1f}%)',flush=True)
    return x
def calc(y,p,t):
    pred=(p>=t).astype(np.int8); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'row_count':int(len(y)),'positive_count':int(y.sum()),'positive_rate':float(y.mean()),'threshold':float(t),
      'roc_auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p)),
      'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),
      'f1':float(f1_score(y,pred,zero_division=0)),'false_positive_rate':float(fp/(fp+tn)) if fp+tn else 0.0,
      'confusion_matrix':{'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}}
def choose_threshold(y,p,min_recall,max_fpr,points):
    rows=[]
    for t in np.linspace(0,1,points):
        m=calc(y,p,t); rows.append((t,m))
    eligible=[r for r in rows if r[1]['recall']>=min_recall and r[1]['false_positive_rate']<=max_fpr]
    if eligible: t,m=max(eligible,key=lambda z:(z[1]['f1'],z[1]['precision'],z[0])); met=True
    else:
        # Deterministic least-constraint-violation fallback, then best F1.
        t,m=min(rows,key=lambda z:(max(0,min_recall-z[1]['recall'])+max(0,z[1]['false_positive_rate']-max_fpr),-z[1]['f1'],-z[0])); met=False
    return float(t),m,met
def base_model(c,seed,n_jobs):
    params={k:v for k,v in c.items() if k not in {'model','feature_set','calibration'}}; params.update(random_state=seed,n_jobs=n_jobs)
    cls={'ExtraTreesClassifier':ExtraTreesClassifier,'RandomForestClassifier':RandomForestClassifier}.get(c['model'])
    if cls is None: raise ValueError(f"Unsupported model {c['model']}")
    return cls(**params)
def drift_ablation(path,all_names,comparison,counts):
    d=pd.read_csv(path)
    split_col=next((c for c in ['comparison_split','split','comparison'] if c in d),None)
    feat_col=next((c for c in ['feature','feature_name'] if c in d),None)
    score_col=next((c for c in ['weighted_smd','importance_weighted_smd','smd'] if c in d),None)
    if not split_col or not feat_col or not score_col: raise ValueError(f'Unsupported feature-shift schema: {list(d.columns)}')
    d=d[d[split_col].astype(str).str.lower().eq(comparison.lower())].copy()
    if d.empty: raise ValueError(f'No {comparison} rows in feature-shift table')
    if d[feat_col].duplicated().any() or not set(d[feat_col]).issubset(all_names): raise ValueError('Invalid or duplicate drift feature names')
    d=d.sort_values([score_col,feat_col],ascending=[False,True])
    return {f'ablate_top_{n}':d[feat_col].head(n).tolist() for n in counts},d
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config/domain_shift_mitigation.yaml'); ap.add_argument('--project-root',default='.'); ap.add_argument('--overwrite',action='store_true'); ap.add_argument('--progress-every',type=int,default=1000); a=ap.parse_args()
    root=Path(a.project_root).resolve(); cp=resolve(root,a.config); cfg=yaml.safe_load(cp.read_text(encoding='utf-8')); cols=cfg['columns']; inp=cfg['inputs']; out={k:resolve(root,v) for k,v in cfg['outputs'].items() if k!='validation_summary'}
    existing=[str(p) for p in out.values() if p.exists()]
    if existing and not a.overwrite: raise FileExistsError(f'Outputs exist; pass --overwrite: {existing}')
    mp=resolve(root,inp['modeling_manifest']); dp=resolve(root,inp['domain_shift_feature_table']); priorp=resolve(root,inp['improved_metrics'])
    for p in [mp,dp,priorp]:
        if not p.is_file(): raise FileNotFoundError(p)
    f=pd.read_csv(mp); req=[cols[k] for k in ['patch_id','aoi_id','feature_patch_path','target','split']]
    if set(req)-set(f) or f[cols['patch_id']].duplicated().any(): raise ValueError('Invalid modeling manifest')
    split=f[cols['split']].astype(str); masks={s:split.eq(s).to_numpy() for s in ['train','validation','test']}
    if set(split)!={'train','validation','test'}: raise ValueError('Expected train, validation, test splits')
    y=pd.to_numeric(f[cols['target']],errors='raise').astype(np.int8).to_numpy()
    if not np.isin(y,[0,1]).all() or any(np.unique(y[m]).size!=2 for m in masks.values()): raise ValueError('Every split must contain both classes')
    fn=names(cfg['features']); forbidden=set(cfg['features'].get('forbidden_predictor_columns',[]))
    if any(n.split('__')[0] in forbidden for n in fn): raise ValueError('Target leakage in feature configuration')
    x=load_x(f,root,cfg['features'],cols['feature_patch_path'],a.progress_every)
    mit=cfg['mitigation']; counts=sorted({int(v) for v in mit['ablate_top_n']}); ablations,drift=drift_ablation(dp,set(fn),mit['ablation_comparison_split'],counts)
    if mit['ablation_comparison_split']!='validation': raise ValueError('Task 3.13 ablation must use validation drift only')
    feature_sets={'all':list(range(len(fn)))}
    for key,dropped in ablations.items(): feature_sets[key]=[i for i,n in enumerate(fn) if n not in dropped]
    rows=[]; fitted=[]
    print(f"Running {len(mit['candidates'])} mitigation candidates using train -> validation only...",flush=True)
    for cid,c in enumerate(mit['candidates'],1):
        idx=feature_sets[c['feature_set']]; model=base_model(c,int(cfg['random_seed']),int(mit['n_jobs']))
        if c['calibration']=='sigmoid': model=CalibratedClassifierCV(model,method='sigmoid',cv=int(mit['calibration_cv_folds']),n_jobs=int(mit['n_jobs']))
        elif c['calibration']!='none': raise ValueError(f"Unsupported calibration {c['calibration']}")
        model.fit(x[masks['train']][:,idx],y[masks['train']]); p=model.predict_proba(x[masks['validation']][:,idx])[:,1]
        t,m,met=choose_threshold(y[masks['validation']],p,float(mit['minimum_recall']),float(mit['maximum_false_positive_rate']),int(mit['threshold_grid_points']))
        aoi=[]
        for name in sorted(f.loc[masks['validation'],cols['aoi_id']].astype(str).unique()):
            local=masks['validation'] & f[cols['aoi_id']].astype(str).eq(name).to_numpy(); mm=calc(y[local],model.predict_proba(x[local][:,idx])[:,1],t); aoi.append((name,mm))
        row={'candidate_id':cid,'model':c['model'],'feature_set':c['feature_set'],'calibration':c['calibration'],'feature_count':len(idx),'threshold':t,'validation_roc_auc':m['roc_auc'],'validation_pr_auc':m['pr_auc'],'validation_precision':m['precision'],'validation_recall':m['recall'],'validation_f1':m['f1'],'validation_false_positive_rate':m['false_positive_rate'],'constraints_met':met,'worst_aoi_pr_auc':min(v['pr_auc'] for _,v in aoi),'parameters_json':json.dumps(c,sort_keys=True)}
        rows.append(row); fitted.append((model,idx,aoi)); print(f"Candidate {cid}: PR-AUC={m['pr_auc']:.4f}, recall={m['recall']:.4f}, FPR={m['false_positive_rate']:.4f}, constraints={'met' if met else 'fallback'}",flush=True)
    eligible=[i for i,r in enumerate(rows) if r['constraints_met']]; pool=eligible or list(range(len(rows)))
    wi=max(pool,key=lambda i:(rows[i]['validation_pr_auc'],rows[i]['worst_aoi_pr_auc'],rows[i]['validation_f1'],-rows[i]['candidate_id']))
    winner=rows[wi]; model,idx,aoi=fitted[wi]; threshold=float(winner['threshold'])
    # Locked model, feature set, and threshold exist before this first test inference.
    probs={s:model.predict_proba(x[m][:,idx])[:,1] for s,m in masks.items()}; metrics={s:calc(y[masks[s]],probs[s],threshold) for s in masks}
    prior=json.loads(priorp.read_text(encoding='utf-8')); deltas={s:{k:metrics[s][k]-prior['metrics_by_split'][s][k] for k in ['roc_auc','pr_auc','precision','recall','f1']} for s in ['validation','test']}
    gates_cfg=cfg['quality_gates']; gates={'test_roc_auc':{'value':metrics['test']['roc_auc'],'limit':float(gates_cfg['test_roc_auc_min']),'operator':'>='},'test_pr_auc':{'value':metrics['test']['pr_auc'],'limit':float(gates_cfg['test_pr_auc_min']),'operator':'>='},'test_recall':{'value':metrics['test']['recall'],'limit':float(gates_cfg['test_recall_min']),'operator':'>='},'test_false_positive_rate':{'value':metrics['test']['false_positive_rate'],'limit':float(gates_cfg['test_false_positive_rate_max']),'operator':'<='}}
    for g in gates.values(): g['passed']=g['value']>=g['limit'] if g['operator']=='>=' else g['value']<=g['limit']
    for p in out.values(): p.parent.mkdir(parents=True,exist_ok=True)
    artifact={'model':model,'threshold':threshold,'all_feature_names':fn,'selected_feature_indices':idx,'selected_feature_names':[fn[i] for i in idx],'ablated_feature_names':[n for i,n in enumerate(fn) if i not in idx],'selected_candidate_id':winner['candidate_id'],'selected_candidate':mit['candidates'][wi],'selection_used_splits':['train','validation'],'test_used_for_selection':False,'test_evaluations_after_selection':1,'ablation_comparison_split':'validation','manifest_sha256':sha256(mp),'drift_table_sha256':sha256(dp),'config_sha256':sha256(cp)}
    joblib.dump(artifact,out['model_artifact'],compress=3)
    parts=[]
    for s,m in masks.items():
        q=f.loc[m,[cols['patch_id'],cols['aoi_id'],cols['split'],cols['target']]].copy(); q['probability']=probs[s]; q['prediction']=(probs[s]>=threshold).astype(np.int8); parts.append(q)
    pd.concat(parts,ignore_index=True).to_csv(out['predictions'],index=False,lineterminator='\n')
    ef=pd.DataFrame(rows); ef['selected']=ef.candidate_id.eq(winner['candidate_id']); ef.to_csv(out['experiment_results'],index=False,lineterminator='\n')
    abrows=[]
    for key,v in ablations.items():
        for rank,n in enumerate(v,1): abrows.append({'feature_set':key,'rank':rank,'feature':n,'source_comparison_split':'validation'})
    pd.DataFrame(abrows).to_csv(out['ablated_features'],index=False,lineterminator='\n')
    pd.DataFrame([{'candidate_id':winner['candidate_id'],'aoi_id':n,**m} for n,m in aoi]).to_csv(out['validation_aoi_metrics'],index=False,lineterminator='\n')
    raw=model.calibrated_classifiers_[0].estimator if hasattr(model,'calibrated_classifiers_') else model
    imp=getattr(raw,'feature_importances_',np.full(len(idx),np.nan)); pd.DataFrame({'feature':[fn[i] for i in idx],'importance':imp}).sort_values('importance',ascending=False,na_position='last').to_csv(out['feature_importance'],index=False,lineterminator='\n')
    report={'task':'3.13','status':'COMPLETED','created_at_utc':datetime.now(timezone.utc).isoformat(),'protocol':{'fit_split':'train','selection_split':'validation','ablation_drift_split':'validation','test_used_for_selection':False,'test_evaluations_after_selection':1},'selected_candidate':winner,'metrics_by_split':metrics,'task_3_11_comparison_delta':deltas,'quality_gates':gates,'quality_gate_status':'PASSED' if all(g['passed'] for g in gates.values()) else 'FAILED','artifacts':{k:cfg['outputs'][k] for k in out},'input_sha256':{'manifest':sha256(mp),'feature_shift':sha256(dp),'improved_metrics':sha256(priorp)},'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__,'joblib':joblib.__version__}}
    out['metrics'].write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f"Selected candidate: {winner['candidate_id']} ({winner['model']}, {winner['feature_set']}, calibration={winner['calibration']}); threshold={threshold:.3f}")
    for s in ['validation','test']:
        m=metrics[s]; print(f"{s}: ROC-AUC={m['roc_auc']:.4f}, PR-AUC={m['pr_auc']:.4f}, precision={m['precision']:.4f}, recall={m['recall']:.4f}, F1={m['f1']:.4f}, FPR={m['false_positive_rate']:.4f}")
    print(f"Quality gates: {report['quality_gate_status']}"); print(f"Metrics: {out['metrics']}"); print('Task 3.13 domain-shift mitigation: COMPLETED'); return 0
if __name__=='__main__': raise SystemExit(main())
