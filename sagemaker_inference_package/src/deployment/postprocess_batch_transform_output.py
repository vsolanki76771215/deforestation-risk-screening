#!/usr/bin/env python3
"""Turn Batch Transform output into the local Streamlit inference outputs."""
from __future__ import annotations
import argparse, csv, json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd, rasterio
from pyproj import Transformer
def poly(transform,row,col,size,xform):
 pts=[]
 for c,r in [(col,row),(col+size,row),(col+size,row+size),(col,row+size)]:
  x,y=transform*(c,r);pts.append([round(v,8) for v in xform.transform(x,y)])
 return pts+[pts[0]]
def main():
 p=argparse.ArgumentParser();p.add_argument('--batch-output-jsonl',required=True);p.add_argument('--reference-raster',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--run-id',required=True);a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 rows=[json.loads(x) for x in Path(a.batch_output_jsonl).read_text(encoding='utf-8').splitlines() if x.strip()]; frame=pd.DataFrame(rows)
 if frame.empty or not {'patch_id','aoi_id','row_off','col_off','probability','prediction','threshold'}.issubset(frame): raise ValueError('Batch output is empty or does not match the inference contract')
 frame.insert(0,'run_id',a.run_id);frame.to_csv(out/'predictions.csv',index=False,quoting=csv.QUOTE_MINIMAL);pd.DataFrame(columns=['patch_id','aoi_id','quality_flag','rejection_reason']).to_csv(out/'rejected_patches.csv',index=False)
 summary=frame.groupby('aoi_id').agg(patch_count=('patch_id','size'),alert_count=('prediction','sum'),mean_probability=('probability','mean'),max_probability=('probability','max')).reset_index();summary['alert_rate']=summary.alert_count/summary.patch_count;summary.to_csv(out/'aoi_summary.csv',index=False)
 with rasterio.open(a.reference_raster) as raster:
  xform=Transformer.from_crs(raster.crs,'EPSG:4326',always_xy=True);size=32; features=[{'type':'Feature','properties':{'patch_id':str(r.patch_id),'aoi_id':str(r.aoi_id),'probability':round(float(r.probability),7),'prediction':int(r.prediction),'threshold':round(float(r.threshold),7),'quality_flag':str(getattr(r,'quality_flag','accepted'))},'geometry':{'type':'Polygon','coordinates':[poly(raster.transform,int(r.row_off),int(r.col_off),size,xform)]}} for r in frame.itertuples(index=False)]
 (out/'prediction_map.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features},indent=2)+'\n',encoding='utf-8')
 (out/'run_report.json').write_text(json.dumps({'run_id':a.run_id,'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'COMPLETED','task_scope':'deforestation-risk probability; not confirmed illegal-mining detection','accepted_patch_count':len(frame),'rejected_patch_count':0,'alert_count':int(frame.prediction.sum()),'outputs':{'predictions':'predictions.csv','rejected':'rejected_patches.csv','aoi_summary':'aoi_summary.csv','map':'prediction_map.geojson'}},indent=2)+'\n',encoding='utf-8')
 print(f'Streamlit-ready outputs: {out}')
if __name__=='__main__': main()
