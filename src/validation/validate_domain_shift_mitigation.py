"""Validate Task 3.13 artifacts, metrics, ranking, and holdout safeguards."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import joblib, numpy as np, pandas as pd, yaml
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

def resolve(root,v):
    p=Path(v); return p if p.is_absolute() else root/p
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def metrics(y,p,pred):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'roc_auc':roc_auc_score(y,p),'pr_auc':average_precision_score(y,p),'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0),'f1':f1_score(y,pred,zero_division=0),'false_positive_rate':fp/(fp+tn) if fp+tn else 0.0}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config/domain_shift_mitigation.yaml'); ap.add_argument('--project-root',default='.'); a=ap.parse_args()
    root=Path(a.project_root).resolve(); cp=resolve(root,a.config); cfg=yaml.safe_load(cp.read_text(encoding='utf-8')); out=cfg['outputs']; summary=resolve(root,out['validation_summary']); checks=[]; errors=[]
    def check(name,passed,detail):
        checks.append({'name':name,'passed':bool(passed),'detail':detail})
        if not passed: errors.append(f'{name}: {detail}')
    paths={k:resolve(root,v) for k,v in out.items() if k!='validation_summary'}; missing=[str(p) for p in paths.values() if not p.is_file()]
    check('required_files_exist',not missing,f'missing={missing}')
    if missing:
        result={'task':'3.13','status':'FAILED','checks':checks,'errors':errors}; summary.parent.mkdir(parents=True,exist_ok=True); summary.write_text(json.dumps(result,indent=2)+'\n'); print(f'Validation summary: {summary}'); print('Task 3.13 validation: FAILED'); return 1
    art=joblib.load(paths['model_artifact']); report=json.loads(paths['metrics'].read_text()); pred=pd.read_csv(paths['predictions']); exp=pd.read_csv(paths['experiment_results']); abl=pd.read_csv(paths['ablated_features']); aoi=pd.read_csv(paths['validation_aoi_metrics']); imp=pd.read_csv(paths['feature_importance']); manifest=pd.read_csv(resolve(root,cfg['inputs']['modeling_manifest'])); c=cfg['columns']
    required={'model','threshold','all_feature_names','selected_feature_indices','selected_feature_names','ablated_feature_names','selected_candidate_id','selection_used_splits','test_used_for_selection','test_evaluations_after_selection','ablation_comparison_split'}
    check('model_bundle_schema',required<=set(art),f'missing={sorted(required-set(art))}')
    protocol=report.get('protocol',{}); safe=art.get('selection_used_splits')==['train','validation'] and art.get('test_used_for_selection') is False and art.get('test_evaluations_after_selection')==1 and art.get('ablation_comparison_split')=='validation' and protocol.get('test_used_for_selection') is False and protocol.get('ablation_drift_split')=='validation'
    check('strict_holdout_protocol',safe,'fit/selection/ablation must exclude test; test evaluated once after lock')
    hashes=report.get('input_sha256',{}); hash_ok=hashes.get('manifest')==sha256(resolve(root,cfg['inputs']['modeling_manifest'])) and hashes.get('feature_shift')==sha256(resolve(root,cfg['inputs']['domain_shift_feature_table'])) and hashes.get('improved_metrics')==sha256(resolve(root,cfg['inputs']['improved_metrics']))
    check('input_integrity',hash_ok,'all recorded input SHA-256 values must reproduce')
    selected=exp[exp.selected.astype(str).str.lower().eq('true')]; check('one_selected_candidate',len(selected)==1,f'selected_rows={len(selected)}')
    eligible=exp[exp.constraints_met.astype(str).str.lower().eq('true')]; pool=eligible if len(eligible) else exp
    expected=pool.sort_values(['validation_pr_auc','worst_aoi_pr_auc','validation_f1','candidate_id'],ascending=[False,False,False,True]).iloc[0].candidate_id
    check('validation_only_ranking_reproduces',int(expected)==int(art['selected_candidate_id']),f"expected={int(expected)}, selected={art['selected_candidate_id']}")
    alln=art['all_feature_names']; idx=art['selected_feature_indices']; schema=[alln[i] for i in idx]==art['selected_feature_names'] and set(art['selected_feature_names']).isdisjoint(art['ablated_feature_names']) and set(art['selected_feature_names'])|set(art['ablated_feature_names'])==set(alln)
    check('selected_feature_schema',schema,f"selected={len(art['selected_feature_names'])}, ablated={len(art['ablated_feature_names'])}")
    forbidden=set(cfg['features'].get('forbidden_predictor_columns',[])); leaked=[n for n in alln if n.split('__')[0] in forbidden]; check('no_target_leakage',not leaked,f'leaked={leaked}')
    check('ablation_uses_validation_only',set(abl.source_comparison_split.astype(str))=={'validation'},f"sources={sorted(set(abl.source_comparison_split.astype(str)))}")
    ab_ok=True
    for key,n in [(f'ablate_top_{int(n)}',int(n)) for n in cfg['mitigation']['ablate_top_n']]:
        q=abl[abl.feature_set.eq(key)].sort_values('rank'); ab_ok &= len(q)==n and q['rank'].tolist()==list(range(1,n+1)) and q.feature.is_unique
    check('ablation_sets_complete',ab_ok,f'rows={len(abl)}')
    required_pred={c['patch_id'],c['aoi_id'],c['split'],c['target'],'probability','prediction'}; check('prediction_schema',required_pred<=set(pred),f'missing={sorted(required_pred-set(pred))}')
    check('one_prediction_per_patch',len(pred)==len(manifest) and pred[c['patch_id']].is_unique and set(pred[c['patch_id']])==set(manifest[c['patch_id']]),f'predictions={len(pred)}, manifest={len(manifest)}')
    t=float(art['threshold']); check('threshold_applied',0<=t<=1 and pred.prediction.astype(np.int8).eq(pred.probability.ge(t).astype(np.int8)).all(),f'threshold={t}')
    diffs={}
    for s in ['train','validation','test']:
        q=pred[pred[c['split']].eq(s)]; got=metrics(q[c['target']].astype(np.int8),q.probability,q.prediction.astype(np.int8))
        for k,v in got.items(): diffs[f'{s}.{k}']=abs(float(v)-float(report['metrics_by_split'][s][k]))
    check('reported_metrics_reproduce',max(diffs.values(),default=0)<=1e-12,f"maximum_absolute_difference={max(diffs.values(),default=0):.3g}")
    prior=json.loads(resolve(root,cfg['inputs']['improved_metrics']).read_text()); delta_ok=all(np.isclose(report['task_3_11_comparison_delta'][s][k],report['metrics_by_split'][s][k]-prior['metrics_by_split'][s][k],atol=1e-12) for s in ['validation','test'] for k in ['roc_auc','pr_auc','precision','recall','f1'])
    check('task_3_11_deltas_reproduce',delta_ok,'mitigated minus improved metrics')
    expected_aois=set(manifest.loc[manifest[c['split']].eq('validation'),c['aoi_id']].astype(str)); check('validation_aoi_profiles_complete',set(aoi.aoi_id.astype(str))==expected_aois and aoi.aoi_id.is_unique,f'expected={sorted(expected_aois)}, actual={sorted(set(aoi.aoi_id.astype(str)))}')
    check('feature_importance_schema',set(imp.columns)=={'feature','importance'} and set(imp.feature)==set(art['selected_feature_names']),f'rows={len(imp)}')
    gate_ok=all(bool(g['passed'])==(g['value']>=g['limit'] if g['operator']=='>=' else g['value']<=g['limit']) for g in report['quality_gates'].values()); check('quality_gate_reporting',gate_ok,f"status={report['quality_gate_status']}")
    result={'task':'3.13','status':'PASSED' if not errors else 'FAILED','experiment_status':report['status'],'model_quality_gate_status':report['quality_gate_status'],'checks':checks,'errors':errors}; summary.parent.mkdir(parents=True,exist_ok=True); summary.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(f"Artifact validation: {result['status']}"); print(f"Model quality gates: {result['model_quality_gate_status']}"); print(f'Validation summary: {summary}'); print(f"Task 3.13 validation: {result['status']}"); return 0 if not errors else 1
if __name__=='__main__': raise SystemExit(main())
