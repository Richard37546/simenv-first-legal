/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#include "FSM/FSM.h"
#include <iostream>
#include <cmath>

FSM::FSM(CtrlComponents *ctrlComp)
    :_ctrlComp(ctrlComp),
     _reportedStateName(FSMStateName::PASSIVE),
     _pendingStateName(FSMStateName::INVALID),
     _hasPendingStateRequest(false),
     _pendingSafeHold(false){

    _stateList.invalid = nullptr;
    _stateList.passive = new State_Passive(_ctrlComp);
    _stateList.fixedStand = new State_FixedStand(_ctrlComp);
    _stateList.freeStand = new State_FreeStand(_ctrlComp);
    _stateList.trotting = new State_Trotting(_ctrlComp);
    _stateList.balanceTest = new State_BalanceTest(_ctrlComp);
    _stateList.swingTest = new State_SwingTest(_ctrlComp);
    _stateList.stepTest = new State_StepTest(_ctrlComp);
#ifdef COMPILE_WITH_MOVE_BASE
    _stateList.moveBase = new State_move_base(_ctrlComp);
#endif  // COMPILE_WITH_MOVE_BASE
    _stateList.rl = new State_RL(_ctrlComp);
    _controllerModePub = _node.advertise<std_msgs::String>("/unitree/controller_mode", 1, true);
    _fixedStandService = _node.advertiseService("/unitree/request_fixedstand", &FSM::requestFixedStand, this);
    _rlService = _node.advertiseService("/unitree/request_rl", &FSM::requestRl, this);
    _safeHoldService = _node.advertiseService("/unitree/request_safe_hold", &FSM::requestSafeHold, this);
    initialize();
}

FSM::~FSM(){
    _stateList.deletePtr();
}

void FSM::initialize(){
    _currentState = _stateList.passive;
    _currentState -> enter();
    _nextState = _currentState;
    _mode = FSMMode::NORMAL;
    _reportedStateName = _currentState->_stateName;
    publishControllerMode();
}

void FSM::run(){
    _startTime = getSystemTime();
    _ctrlComp->sendRecv();
    _ctrlComp->ioInterFreeDog->sendRecv();
    _ctrlComp->runWaveGen();
    _ctrlComp->estimator->run();
    if(!checkSafty()){
        // _ctrlComp->ioInter->setPassive();
    }

    if(_mode == FSMMode::NORMAL){
        _currentState->run();
        FSMStateName requested = keyboardRequestedMode(_ctrlComp->lowState->userCmd);
        bool safeHold = false;
        FSMStateName serviceTarget = FSMStateName::INVALID;
        if (consumeModeRequest(&serviceTarget, &safeHold)) {
            if (safeHold) {
                requested = (checkSafty() && jointsFinite()) ? FSMStateName::FIXEDSTAND : FSMStateName::PASSIVE;
            } else {
                requested = serviceTarget;
            }
        }
        _nextStateName = (requested != FSMStateName::INVALID) ? requested : _currentState->checkChange();
        if(_nextStateName != _currentState->_stateName){
            _mode = FSMMode::CHANGE;
            _nextState = getNextState(_nextStateName);
            std::cout << "Switched from " << _currentState->_stateNameString
                      << " to " << _nextState->_stateNameString << std::endl;
        }
    }
    else if(_mode == FSMMode::CHANGE){
        _currentState->exit();
        _currentState = _nextState;
        _currentState->enter();
        _reportedStateName = _currentState->_stateName;
        publishControllerMode();
        _mode = FSMMode::NORMAL;
        _currentState->run();
    }

    absoluteWait(_startTime, (long long)(_ctrlComp->dt * 1000000));
}

bool FSM::requestFixedStand(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &response){
    if (!serviceRequestAllowed(FSMStateName::FIXEDSTAND)) {
        response.success = false;
        response.message = std::string("rejected current_mode=") + modeName(_reportedStateName) + "; allowed=PASSIVE,FIXEDSTAND,RL";
        return true;
    }
    queueModeRequest(FSMStateName::FIXEDSTAND, false, "service");
    response.success = true;
    response.message = std::string("accepted target=FIXEDSTAND current_mode=") + modeName(_reportedStateName);
    return true;
}

bool FSM::requestRl(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &response){
    if (!serviceRequestAllowed(FSMStateName::RL)) {
        response.success = false;
        response.message = std::string("rejected current_mode=") + modeName(_reportedStateName) + "; required=FIXEDSTAND";
        return true;
    }
    queueModeRequest(FSMStateName::RL, false, "service");
    response.success = true;
    response.message = std::string("accepted target=RL current_mode=") + modeName(_reportedStateName);
    return true;
}

bool FSM::requestSafeHold(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &response){
    queueModeRequest(FSMStateName::FIXEDSTAND, true, "service");
    response.success = true;
    response.message = std::string("accepted target=SAFE_HOLD current_mode=") + modeName(_reportedStateName);
    return true;
}

void FSM::queueModeRequest(FSMStateName target, bool safeHold, const char *){
    std::lock_guard<std::mutex> lock(_modeRequestMutex);
    if (!safeHold && _reportedStateName == target) {
        return;
    }
    _pendingStateName = target;
    _pendingSafeHold = safeHold;
    _hasPendingStateRequest = true;
}

bool FSM::consumeModeRequest(FSMStateName *target, bool *safeHold){
    std::lock_guard<std::mutex> lock(_modeRequestMutex);
    if (!_hasPendingStateRequest) {
        return false;
    }
    *target = _pendingStateName;
    *safeHold = _pendingSafeHold;
    _hasPendingStateRequest = false;
    _pendingStateName = FSMStateName::INVALID;
    _pendingSafeHold = false;
    return true;
}

FSMStateName FSM::keyboardRequestedMode(UserCommand command){
    if (command == UserCommand::L2_A) {
        queueModeRequest(FSMStateName::FIXEDSTAND, false, "keyboard");
    } else if (command == UserCommand::RL) {
        queueModeRequest(FSMStateName::RL, false, "keyboard");
    }
    return FSMStateName::INVALID;
}

bool FSM::serviceRequestAllowed(FSMStateName target) const{
    std::lock_guard<std::mutex> lock(_modeRequestMutex);
    if (target == FSMStateName::FIXEDSTAND) {
        return _reportedStateName == FSMStateName::PASSIVE || _reportedStateName == FSMStateName::FIXEDSTAND || _reportedStateName == FSMStateName::RL;
    }
    return target == FSMStateName::RL && (_reportedStateName == FSMStateName::FIXEDSTAND || _reportedStateName == FSMStateName::RL);
}

bool FSM::jointsFinite() const{
    for (int i = 0; i < 12; ++i) {
        if (!std::isfinite(_ctrlComp->lowState->motorState[i].q) || !std::isfinite(_ctrlComp->lowState->motorState[i].dq)) {
            return false;
        }
    }
    return true;
}

const char *FSM::modeName(FSMStateName mode) const{
    switch (mode) {
    case FSMStateName::PASSIVE: return "PASSIVE";
    case FSMStateName::FIXEDSTAND: return "FIXEDSTAND";
    case FSMStateName::RL: return "RL";
    case FSMStateName::FREESTAND: return "FREESTAND";
    case FSMStateName::TROTTING: return "TROTTING";
    default: return "OTHER";
    }
}

void FSM::publishControllerMode(){
    std_msgs::String mode;
    mode.data = modeName(_reportedStateName);
    _controllerModePub.publish(mode);
}

FSMState* FSM::getNextState(FSMStateName stateName){
    switch (stateName)
    {
    case FSMStateName::INVALID:
        return _stateList.invalid;
        break;
    case FSMStateName::PASSIVE:
        return _stateList.passive;
        break;
    case FSMStateName::FIXEDSTAND:
        return _stateList.fixedStand;
        break;
    case FSMStateName::FREESTAND:
        return _stateList.freeStand;
        break;
    case FSMStateName::TROTTING:
        return _stateList.trotting;
        break;
    case FSMStateName::BALANCETEST:
        return _stateList.balanceTest;
        break;
    case FSMStateName::SWINGTEST:
        return _stateList.swingTest;
        break;
    case FSMStateName::STEPTEST:
        return _stateList.stepTest;
        break;
#ifdef COMPILE_WITH_MOVE_BASE
    case FSMStateName::MOVE_BASE:
        return _stateList.moveBase;
        break;
#endif  // COMPILE_WITH_MOVE_BASE
    case FSMStateName::RL:
        return _stateList.rl;
    break;
    default:
        return _stateList.invalid;
        break;
    }
}

bool FSM::checkSafty(){
    // The angle with z axis less than 60 degree
    if(_ctrlComp->lowState->getRotMat()(2,2) < 0.5 ){
        return false;
    }else{
        return true;
    }
}
