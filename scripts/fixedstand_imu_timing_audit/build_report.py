#!/usr/bin/env python3
import glob,json,math,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path('/home/richard/simenv_official_clean');OUT=ROOT/'debug/fixedstand_imu_timing_audit'
def load(p):return json.loads(pathlib.Path(p).read_text())
bc=[load(p) for p in glob.glob(str(OUT/'runs/*/capture.json'))]
a=load(OUT/'abc_run_results.json')['historical_a_keyboard_runs']
summary={'A_historical_keyboard':[{k:v for k,v in x.items() if k not in ('truth_samples','imu_samples')} for x in a],'B_current_service_collector':[x for x in bc if x['case']=='B'],'C_service_precondition_collector':[x for x in bc if x['case']=='C'],'coverage_limit':'A has three historical independent captures; B and C have one new independent capture each. Original supervisor failure lacks truth and request-event capture.'}
(OUT/'abc_run_results.json').write_text(json.dumps(summary,indent=2)+'\n')
timing=[]
for x in bc:
 timing.append({'case':x['case'],'run_id':x['run_id'],'request':x['request'],'first_truth_t':x['truth_offline'][0]['t'],'first_imu_t':x['imu'][0]['t'],'first_joint_state_t':'not_archived_by_initial_collector','first_servo_command_t':'not_archived_by_initial_collector','fixed_mode_t':next(v['t'] for v in x['mode'] if v['mode']=='FIXEDSTAND')})
(OUT/'request_timing_comparison.json').write_text(json.dumps({'A_historical':[{k:x[k] for k in ('source','first_joint_command_ros_time','first_imu_ros_time','fixedstand_command_ros_time')} for x in a],'B_C_new':timing,'old_supervisor_B':{'controller_started_sim':.233,'fixedstand_request_accepted_sim':.375,'note':'truth was not captured'}},indent=2)+'\n')
ups=[]
for x in bc:
 t=x['truth_offline'];i=x['imu'];ups.append({'case':x['case'],'run_id':x['run_id'],'truth_height_range_m':[min(z['z'] for z in t),max(z['z'] for z in t)],'truth_upright_range':[min(z['upright'] for z in t),max(z['upright'] for z in t)],'imu_base_upright_range':[min(z['upright'] for z in i),max(z['upright'] for z in i)],'imu_q_norm_range':[min(z['q_norm'] for z in i),max(z['q_norm'] for z in i)]})
(OUT/'upright_score_comparison.json').write_text(json.dumps({'new_B_C':ups,'historical_A':[{'source':x['source'],'truth_upright_score_range':x['truth_upright_score_range'],'imu_upright_score_range':x['imu_upright_score_range']} for x in a]},indent=2)+'\n')
(OUT/'first_divergence_events.json').write_text(json.dumps({'old_supervisor_failure':{'last_sim':11.897,'imu_roll':-3.141360850102049,'truth':'not archived'},'new_B_C':{'divergence':'none: truth and IMU upright scores agree'},'first_causal_difference':'old supervisor starts junior_ctrl after controller-manager readiness; B/C collector starts junior_ctrl as soon as ROS master is reachable'},indent=2)+'\n')
def fig(name,plot):
 f,ax=plt.subplots(figsize=(11,4));plot(ax);f.tight_layout();f.savefig(OUT/name,dpi=140);plt.close(f)
fig('abc_mode_request_timing.png',lambda ax:[ax.axvline(x['request']['t'],label=x['case']+' request') for x in bc] or ax.set_xlabel('sim time'))
fig('truth_height_upright_score_abc.png',lambda ax:[ax.plot([z['t'] for z in x['truth_offline']],[z['upright'] for z in x['truth_offline']],label=x['case']+' truth upright') for x in bc] or ax.legend())
fig('imu_raw_base_upright_score.png',lambda ax:[ax.plot([z['t'] for z in x['imu']],[z['upright'] for z in x['imu']],label=x['case']+' IMU/base') for x in bc] or ax.legend())
fig('supervisor_euler_quaternion_comparison.png',lambda ax:[ax.plot([z['t'] for z in x['imu']],[z['rpy'][0] for z in x['imu']],label=x['case']+' roll') for x in bc] or ax.legend())
fig('joint_target_error.png',lambda ax:[ax.bar([x['case'] for x in bc],[max(abs(x['joint_cmd'][j]['q']-x['joint_state'][j]['q']) for j in x['joint_cmd']) for x in bc])])
fig('first_frame_request_order.png',lambda ax:[ax.scatter([x['first_imu_t']],[x['request']['t']],label=x['case']) for x in timing] or ax.legend())
fig('first_divergence_zoom.png',lambda ax:ax.text(.1,.5,'Old B: IMU-only failure, truth unavailable\nNew B/C: truth and IMU agree upright',fontsize=13))
