#!/usr/bin/env python3
"""A/B/C FixedStand audit collector. Truth is recorded only, never gated on."""
import argparse,json,math,os,pathlib,signal,subprocess,sys,time
import rospy,rosgraph
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from std_srvs.srv import Trigger
from unitree_legged_msgs.msg import MotorCmd,MotorState

ROOT=pathlib.Path('/home/richard/simenv_official_clean'); OUT=ROOT/'debug/fixedstand_imu_timing_audit/runs'
JOINTS=('FR_hip','FR_thigh','FR_calf','FL_hip','FL_thigh','FL_calf','RR_hip','RR_thigh','RR_calf','RL_hip','RL_thigh','RL_calf')
def rpy(q):
 return (math.atan2(2*(q.w*q.x+q.y*q.z),1-2*(q.x*q.x+q.y*q.y)),math.asin(max(-1,min(1,2*(q.w*q.y-q.z*q.x)))),math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)))
def upright(q): return 1-2*(q.x*q.x+q.y*q.y)
def finiteq(q): return all(math.isfinite(v) for v in (q.x,q.y,q.z,q.w)) and abs(math.sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w)-1)<.02
def call(cmd,log,env):
 h=log.open('w');p=subprocess.Popen(['/bin/bash','-lc',cmd],cwd=str(ROOT),stdout=h,stderr=subprocess.STDOUT,start_new_session=True,env=env);p._h=h;return p
def stop(p):
 if p and p.poll() is None:
  os.killpg(p.pid,signal.SIGINT); end=time.monotonic()+8
  while p.poll() is None and time.monotonic()<end:time.sleep(.1)
  if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
 if p:p._h.close()
class C:
 def __init__(self):
  self.t=0.;self.truth=[];self.imu=[];self.mode=[];self.cmd={};self.state={};self.events=[];self.nonzero=False
  self.s=[rospy.Subscriber('/clock',Clock,self.clock,queue_size=1000),rospy.Subscriber('/gazebo/model_states',ModelStates,self.truthcb,queue_size=1000),rospy.Subscriber('/trunk_imu',Imu,self.imucb,queue_size=1000),rospy.Subscriber('/unitree/controller_mode',String,self.modecb,queue_size=100)]
  for j in JOINTS:
   x='/a1_gazebo/'+j+'_controller';self.s += [rospy.Subscriber(x+'/command',MotorCmd,self.cmdcb,j,queue_size=1000),rospy.Subscriber(x+'/state',MotorState,self.statecb,j,queue_size=1000)]
 def clock(self,m):self.t=m.clock.to_sec()
 def truthcb(self,m):
  if 'a1_gazebo' in m.name:
   p=m.pose[m.name.index('a1_gazebo')];a,b,c=rpy(p.orientation);self.truth.append({'t':self.t,'z':p.position.z,'q':{'x':p.orientation.x,'y':p.orientation.y,'z':p.orientation.z,'w':p.orientation.w},'rpy':[a,b,c],'upright':upright(p.orientation)})
 def imucb(self,m):
  a,b,c=rpy(m.orientation);self.imu.append({'t':self.t,'frame':m.header.frame_id,'q':{'x':m.orientation.x,'y':m.orientation.y,'z':m.orientation.z,'w':m.orientation.w},'q_norm':math.sqrt(sum(v*v for v in (m.orientation.x,m.orientation.y,m.orientation.z,m.orientation.w))),'rpy':[a,b,c],'upright':upright(m.orientation),'finite':finiteq(m.orientation)})
 def modecb(self,m):self.mode.append({'t':self.t,'mode':m.data})
 def cmdcb(self,m,j):self.cmd[j]={'t':self.t,'q':m.q,'dq':m.dq,'kp':m.Kp,'kd':m.Kd}
 def statecb(self,m,j):self.state[j]={'t':self.t,'q':m.q,'dq':m.dq,'tau':m.tauEst}
 def ready(self,c_mode=False):
  return len(self.imu)>0 and all(j in self.state for j in JOINTS) and (not c_mode or self.mode and self.mode[-1]['mode']=='PASSIVE')
 def stablepre(self):return self.ready(True) and all(self.t-self.state[j]['t']<.1 and self.t-self.cmd.get(j,{'t':-99})['t']<.1 for j in JOINTS) and self.imu[-1]['finite'] and self.t>=.2
 def close(self):
  for x in self.s:x.unregister()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--case',choices=('B','C'),required=True);ap.add_argument('--run-id',required=True);ap.add_argument('--duration-sim',type=float,default=10.0);a=ap.parse_args()
 run=OUT/(a.run_id+'_'+time.strftime('%Y%m%dT%H%M%SZ')+'_pid'+str(os.getpid()));run.mkdir(parents=True,exist_ok=False)
 env=os.environ.copy();env.update({'GUI':'false','PAUSED':'false','UNITREE_CTRL_DT':'0.006','BUILDING_WORLD_FILE':str(ROOT/'generated_building/competition_scene.world')});prefix='source /opt/ros/noetic/setup.bash; source '+str(ROOT)+'/devel/setup.bash; export GAZEBO_MODEL_PATH='+str(ROOT)+'/generated_building:$(rospack find unitree_gazebo)/models:${GAZEBO_MODEL_PATH:-}; '
 p={};c=None;timer=None;request=None;reason='';startwall=time.monotonic()
 try:
  p['gazebo']=call(prefix+'exec roslaunch unitree_guide multi_floor_gazeboSim.launch gui:=false paused:=false user_debug:=False rname:=a1 robot_x:=0.0 robot_y:=-2.2 robot_z:=0.6 robot_yaw:=1.5708',run/'gazebo.log',env)
  master=rosgraph.Master('fixedstand_imu_timing_audit');deadline=time.monotonic()+90
  while time.monotonic()<deadline:
   try:master.getPid();break
   except:time.sleep(.1)
  else:raise RuntimeError('ros_master_not_ready')
  rospy.init_node('fixedstand_imu_timing_audit',anonymous=True,disable_signals=True);c=C();zero=rospy.Publisher('/cmd_vel',Twist,queue_size=10);timer=rospy.Timer(rospy.Duration(.1),lambda e:zero.publish(Twist()))
  p['junior_ctrl']=call(prefix+'exec '+str(ROOT/'devel/lib/unitree_guide/junior_ctrl'),run/'junior_ctrl.log',env)
  deadline=time.monotonic()+180
  while time.monotonic()<deadline and not rospy.is_shutdown():
   ready=c.ready(True) if a.case=='B' else c.stablepre()
   if ready:break
   time.sleep(.005)
  if not ready:raise RuntimeError('request_preconditions_not_met')
  request={'t':c.t,'case':a.case,'precondition':'current_supervisor_mode_and_joint_states' if a.case=='B' else 'continuous_joint_command_state_plus_finite_imu_plus_clock'}
  rospy.wait_for_service('/unitree/request_fixedstand',timeout=5);resp=rospy.ServiceProxy('/unitree/request_fixedstand',Trigger)();request['response_success']=resp.success;request['response_message']=resp.message
  target=c.t+a.duration_sim
  while c.t<target and time.monotonic()-startwall<900 and not rospy.is_shutdown():time.sleep(.005)
  reason='completed'
 finally:
  if timer:timer.shutdown()
  data={'case':a.case,'run_id':a.run_id,'complete':reason=='completed','reason':reason,'request':request,'truth_offline':c.truth if c else [],'imu':c.imu if c else [],'mode':c.mode if c else [],'joint_cmd':c.cmd if c else {},'joint_state':c.state if c else {},'final_sim_time':c.t if c else None,'truth_used_for_control':False,'nonzero_cmd_published':False}
  (run/'capture.json').write_text(json.dumps(data,indent=2)+'\n');
  if c:c.close()
  for x in reversed(list(p.values())):stop(x)
if __name__=='__main__':main()
