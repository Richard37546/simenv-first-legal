/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#ifndef FSM_H
#define FSM_H

// FSM States
#include "FSM/FSMState.h"
#include "FSM/State_FixedStand.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FreeStand.h"
#include "FSM/State_Trotting.h"
#include "FSM/State_BalanceTest.h"
#include "FSM/State_SwingTest.h"
#include "FSM/State_StepTest.h"
#include "common/enumClass.h"
#include "control/CtrlComponents.h"
#ifdef COMPILE_WITH_MOVE_BASE
    #include "FSM/State_move_base.h"
#endif  // COMPILE_WITH_MOVE_BASE
#include "FSM/State_RL_test.h"
#include <mutex>
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <std_srvs/Trigger.h>

struct FSMStateList{
    FSMState *invalid;
    State_Passive *passive;
    State_FixedStand *fixedStand;
    State_FreeStand *freeStand;
    State_Trotting *trotting;
    State_BalanceTest *balanceTest;
    State_SwingTest *swingTest;
    State_StepTest *stepTest;
#ifdef COMPILE_WITH_MOVE_BASE
    State_move_base *moveBase;
#endif  // COMPILE_WITH_MOVE_BASE
    State_RL *rl;

    void deletePtr(){
        delete invalid;
        delete passive;
        delete fixedStand;
        delete freeStand;
        delete trotting;
        delete balanceTest;
        delete swingTest;
        delete stepTest;
#ifdef COMPILE_WITH_MOVE_BASE
        delete moveBase;
#endif  // COMPILE_WITH_MOVE_BASE
        delete rl;
    }
};

class FSM{
public:
    FSM(CtrlComponents *ctrlComp);
    ~FSM();
    void initialize();
    void run();
private:
    bool requestFixedStand(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &);
    bool requestRl(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &);
    bool requestSafeHold(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &);
    void queueModeRequest(FSMStateName target, bool safeHold, const char *source);
    bool consumeModeRequest(FSMStateName *target, bool *safeHold);
    FSMStateName keyboardRequestedMode(UserCommand command);
    bool serviceRequestAllowed(FSMStateName target) const;
    bool jointsFinite() const;
    const char *modeName(FSMStateName mode) const;
    void publishControllerMode();
    FSMState* getNextState(FSMStateName stateName);
    bool checkSafty();
    CtrlComponents *_ctrlComp;
    FSMState *_currentState;
    FSMState *_nextState;
    FSMStateName _nextStateName;
    FSMStateList _stateList;
    FSMMode _mode;
    long long _startTime;
    int count;
    ros::NodeHandle _node;
    ros::Publisher _controllerModePub;
    ros::ServiceServer _fixedStandService;
    ros::ServiceServer _rlService;
    ros::ServiceServer _safeHoldService;
    mutable std::mutex _modeRequestMutex;
    FSMStateName _reportedStateName;
    FSMStateName _pendingStateName;
    bool _hasPendingStateRequest;
    bool _pendingSafeHold;
};


#endif  // FSM_H
