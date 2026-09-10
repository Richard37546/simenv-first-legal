#!/usr/bin/env python3
"""Offline truth/IMU analysis; no ROS publication or online control input."""
import json, math, pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=pathlib.Path('/home/richard/simenv_official_clean')
OLD=ROOT/'debug/a1_upright_control_audit/fixedstand_runs'
OUT=ROOT/'debug/fixedstand_imu_timing_audit'
OUT.mkdir(parents=True, exist_ok=True)

def rpy(q):
 w,x,y,z=(q[k] for k in ('w','x','y','z'))
 return (math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)), math.asin(max(-1,min(1,2*(w*y-z*x)))), math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
def norm(q): return math.sqrt(sum(q[k]*q[k] for k in ('w','x','y','z')))
def upright(q):
 w,x,y,z=(q[k] for k in ('w','x','y','z'))
 return 1-2*(x*x+y*y)
def nearest(rows,t): return min(rows,key=lambda x:abs(x.get('t',0)-t)) if rows else None
def phase(rows): return [x for x in rows if x.get('phase')=='fixedstand']

items=[]
for path in sorted(OLD.glob('fixedstand_*/capture.json')):
 d=json.loads(path.read_text()); truth=phase(d['truth_offline']); imu=phase(d['imu']);
 if not truth: continue
 pairs=[]
 for x in truth[::max(1,len(truth)//500)]:
  i=nearest(imu,x['t'])
  if i: pairs.append({'t':x['t'],'truth_upright_score':upright({'w':math.cos(0),'x':0,'y':0,'z':0}) if False else 1-2*(0), 'truth_z':x['z'],'truth_rpy':[x['roll'],x['pitch'],x['yaw']], 'imu_rpy':[i['roll'],i['pitch'],i['yaw']]})
 # truth capture stores RPY but not raw quaternion.  Reconstruct upright = cos(roll)*cos(pitch).
 truth_scores=[math.cos(x['roll'])*math.cos(x['pitch']) for x in truth]
 imu_scores=[math.cos(x['roll'])*math.cos(x['pitch']) for x in imu]
 first_cmd=min((r['t'] for rows in d['joint_commands'].values() for r in rows),default=None)
 fixed_cmd=min((r['t'] for rows in d['joint_commands'].values() for r in phase(rows)),default=None)
 first_imu=min((r['t'] for r in d['imu']),default=None)
 items.append({'source':str(path),'path':'A_historical_keyboard','truth_height_range_m':[min(x['z'] for x in truth),max(x['z'] for x in truth)],'truth_upright_score_range':[min(truth_scores),max(truth_scores)],'imu_upright_score_range':[min(imu_scores),max(imu_scores)] if imu_scores else None,'first_joint_command_ros_time':first_cmd,'first_imu_ros_time':first_imu,'fixedstand_command_ros_time':fixed_cmd,'truth_samples':truth,'imu_samples':imu})

failure=ROOT/'debug/a1_safe_startup_supervisor_v1/runs/fixedstand_01_20260723T083350Z_pid1179753/timeline.json'
failure_data=json.loads(failure.read_text()) if failure.exists() else []
last=failure_data[-1] if failure_data else {}
out={'historical_a_keyboard_runs':items,'fixedstand_01_reclassified':'FIXEDSTAND_HEALTH_PRECONDITION_FAILED_UNCLASSIFIED','fixedstand_01_available_nontruth_evidence':last}
(OUT/'abc_run_results.json').write_text(json.dumps(out,indent=2)+'\n')
(OUT/'upright_score_comparison.json').write_text(json.dumps({'historical_a':[{k:v for k,v in x.items() if k not in ('truth_samples','imu_samples')} for x in items],'fixedstand_01_truth_evidence':'not_recorded; cannot classify physical fall from supervisor-only run'},indent=2)+'\n')
(OUT/'request_timing_comparison.json').write_text(json.dumps({'historical_A_keyboard':[{k:x[k] for k in ('source','first_joint_command_ros_time','first_imu_ros_time','fixedstand_command_ros_time')} for x in items],'B_current_supervisor_request_time':'not recorded in fixedstand_01 timeline; only post-transition mode evidence exists'},indent=2)+'\n')
fig,ax=plt.subplots(2,1,figsize=(11,6),sharex=True)
for item in items:
 t=[x['t'] for x in item['truth_samples']];ax[0].plot(t,[x['z'] for x in item['truth_samples']],label=pathlib.Path(item['source']).parent.name);ax[1].plot(t,[math.cos(x['roll'])*math.cos(x['pitch']) for x in item['truth_samples']])
ax[0].set_ylabel('truth base height (m)');ax[0].legend(fontsize=7);ax[1].set_ylabel('truth upright score');ax[1].set_xlabel('ROS simulation time (s)');fig.tight_layout();fig.savefig(OUT/'truth_height_upright_score.png',dpi=140);plt.close(fig)
fig,ax=plt.subplots(2,1,figsize=(11,6),sharex=True)
for item in items:
 t=[x['t'] for x in item['imu_samples']];ax[0].plot(t,[math.cos(x['roll'])*math.cos(x['pitch']) for x in item['imu_samples']],label=pathlib.Path(item['source']).parent.name);ax[1].plot(t,[x['roll'] for x in item['imu_samples']],label='roll');ax[1].plot(t,[x['pitch'] for x in item['imu_samples']],ls='--',label='pitch')
ax[0].set_ylabel('IMU/base upright score');ax[0].legend(fontsize=7);ax[1].set_ylabel('Euler rad');ax[1].set_xlabel('ROS simulation time (s)');fig.tight_layout();fig.savefig(OUT/'imu_euler_quaternion_comparison.png',dpi=140);plt.close(fig)
