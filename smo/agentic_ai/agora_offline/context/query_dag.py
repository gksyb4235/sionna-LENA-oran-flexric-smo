"""Read-only retrieval for an agent. Python standard library only; never executes RAN actions."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_bundle(directory=None):
    directory=Path(directory) if directory else Path(__file__).resolve().parent
    manifest=read(directory/'bundle_manifest.json')
    for name in ['dag.json','edge_evidence.json']:
        with (directory/name).open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=manifest['files_sha256'][name]:raise ValueError('Bundle checksum mismatch: '+name)
    return read(directory/'dag.json'),read(directory/'edge_evidence.json')


def finite_number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def applicability(graph,feature,current_rcp=None,after=None,mode=None,duration_s=None):
    baseline=graph['baseline_rcp'];action=graph['actions'][feature];mismatches=[];missing=[]
    if current_rcp is None:missing.append('current_rcp: all 11 RCP values')
    elif not isinstance(current_rcp,dict):raise ValueError('current_rcp must be an object with named RCP values')
    else:
        unknown=set(current_rcp)-set(baseline)
        if unknown:raise ValueError('Unknown RCPs: '+','.join(sorted(unknown)))
        for name,expected in baseline.items():
            if name not in current_rcp:missing.append('current_rcp.'+name)
            elif not finite_number(current_rcp[name]):raise ValueError('Non-finite/non-numeric RCP: '+name)
            elif not math.isclose(current_rcp[name],expected,rel_tol=0,abs_tol=1e-9):
                mismatches.append(dict(field='current_rcp.'+name,reference=expected,requested=current_rcp[name]))
    if after is None:missing.append('proposed_after')
    elif not finite_number(after):raise ValueError('after must be a finite number')
    elif not math.isclose(after,action['after'],rel_tol=0,abs_tol=1e-9):mismatches.append(dict(field='after',reference=action['after'],requested=after))
    if mode is None:missing.append('mode: fixed_setting from t=0')
    elif mode!='fixed_setting':mismatches.append(dict(field='mode',reference='fixed_setting',requested=mode))
    if duration_s is None:missing.append('duration_s')
    elif not finite_number(duration_s):raise ValueError('duration_s must be a finite number')
    elif duration_s!=900:mismatches.append(dict(field='duration_s',reference=900,requested=duration_s))
    status='out_of_scope' if mismatches else 'insufficient_context' if missing else 'reference_conditions_match'
    return dict(status=status,mismatches=mismatches,missing=missing,
        checked='joint RCP baseline, one tested target value, fixed_setting at t=0, 900s duration',
        environment_match='not_checked: simulator, trace/demand/channel configuration and deployment context still matter',
        population_generalization_established=False,simultaneous_or_sequential_action_effects_supported=False,
        action_execution_implemented=False)


def query(graph,details,feature,cell=None,kpi=None,current_rcp=None,after=None,mode=None,duration_s=None,
          include_derived=False,include_details=False):
    if not isinstance(feature,str) or feature not in graph['actions']:raise ValueError('Unknown/simultaneous action; request one tested RCP')
    if cell is not None and cell not in {n.get('cell') for n in graph['nodes'] if n['kind']=='kpi'}:raise ValueError('Unknown cell: '+str(cell))
    if kpi in ('rsrp_p10_dbm','ho_successes','delay_over_100ms_pct','rsrp_mean_p10_gap_db'):
        return dict(artifact_kind='no_reference_evidence_for_v16_target',results=[],
            applicability=dict(status='out_of_scope'),
            retrieval=dict(returned_relations=0,filters=dict(feature=feature,cell=cell,kpi=kpi)),
            decision_status='Use numeric v16 predictions and coordinate-reset contrasts; historical DAG has no evidence for this target',
            returned_effects_are_reference_observations=False,transported_effect_estimate=None)
    if kpi is not None and kpi not in {n.get('kpi') for n in graph['nodes'] if n['kind']=='kpi'}:raise ValueError('Unknown KPI: '+str(kpi))
    results=[]
    for e in graph['edges']:
        if e['feature']!=feature or (cell and e['cell']!=cell) or (kpi and e['kpi']!=kpi):continue
        if e['derived'] and not include_derived:continue
        evidence=details[e['id']]
        result={**e,'trace_effects_900s':evidence['trace_effects_900s'],
            'original_complete_cases_900s':evidence['original_complete_cases_900s'],
            'supplement_only_900s':evidence['supplement_only_900s'],
            'window_sign_pattern':evidence['window_sign_summary']['sign_pattern']}
        if e['kpi']=='rsrp_interval_mean_dbm':
            result['v16_definition_warning']='Historical RSRP mean used different aggregation; v16 uses arithmetic mean of pooled UE-time dBm samples. Do not transport this effect numerically.'
        if include_details:result['evidence']=evidence
        results.append(result)
    if not results:raise ValueError('No rows under these filters. Derived KPI retrieval requires include_derived=True / --include-derived.')
    return dict(artifact_kind=graph['artifact_kind'],reference_action=graph['actions'][feature],
        reference_baseline_rcp=graph['baseline_rcp'],applicability=applicability(graph,feature,current_rcp,after,mode,duration_s),
        retrieval=dict(returned_relations=len(results),total_tested_relations_for_action=42,
            filters=dict(cell=cell,kpi=kpi,include_derived=include_derived),
            omitted_relationships='not assessed by this response, not proven absent'),
        graph_semantics=graph['semantics'],scope=graph['scope'],reasoning_rules=graph['reasoning_rules'],
        results=results,returned_effects_are_reference_observations=True,transported_effect_estimate=None,
        decision_status='evidence_retrieval_only; objectives, acceptable harm and deployment applicability are not decided')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature',required=True);parser.add_argument('--cell');parser.add_argument('--kpi')
    parser.add_argument('--current-rcp',type=Path,help='JSON file containing all 11 current RCP values')
    parser.add_argument('--after',type=float);parser.add_argument('--mode');parser.add_argument('--duration-s',type=float)
    parser.add_argument('--include-derived',action='store_true');parser.add_argument('--details',action='store_true')
    parser.add_argument('--output',type=Path,help='Optional retrieval JSON output file; no simulator execution')
    args=parser.parse_args()
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    try:
        graph,details=load_bundle()
        result=query(graph,details,args.feature,args.cell,args.kpi,read(args.current_rcp) if args.current_rcp else None,
            args.after,args.mode,args.duration_s,args.include_derived,args.details)
        body=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        if args.output:
            if args.output.resolve().exists():raise ValueError('Output already exists; choose a new path')
            args.output.write_text(body,encoding='utf-8')
        else:print(body,end='')
    except (ValueError,KeyError,OSError) as error:
        print(json.dumps({'error':str(error)},ensure_ascii=False),file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
