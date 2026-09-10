/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#include "FSM/FSMState.h"
#include <iomanip>
#include <sstream>

FSMState::FSMState(CtrlComponents *ctrlComp, FSMStateName stateName, std::string stateNameString)
            :_ctrlComp(ctrlComp), _stateName(stateName), _stateNameString(stateNameString){
    _lowCmd = _ctrlComp->lowCmd;
    _lowState = _ctrlComp->lowState;
    stair_cmd_receipt_pub_ = nh.advertise<std_msgs::String>("/audit/stair_cmd_receipt", 20);
}

long long FSMState::getRosTime()
{
    return _ctrlComp->ioInter->current_time;
}

long long  FSMState::getTime()
{
    if (real == false)
    {
        return getRosTime();
    }
    else
    {
        return getSystemTime();
    }
}

void FSMState::wait(long long startTime, long long waitTime){
    if (real == false)
    {
        rosAbsoluteWait(startTime,waitTime);
    }
    else
    {
        absoluteWait(startTime,waitTime);
    }
}

void FSMState::rosAbsoluteWait(long long startTime, long long waitTime){
    overtime = 0;
    if(getRosTime() - startTime > waitTime){
        std::cout << "[WARNING] The waitTime=" << waitTime << " of function absoluteWait is not enough!" << std::endl
        << "The program has already cost " << getSystemTime() - startTime << "us." << std::endl;
    }
    while((getRosTime() - startTime < waitTime) && (overtime < OVERTIME)){

        // std::cout << "getRosTime()" << getRosTime() << std::endl;
        // std::cout << "startTime" << startTime << std::endl;
        // std::cout << "wait" << std::endl;
        // std::cout << "overtime" <<  overtime << std::endl;
        usleep(50);
        overtime += 50;
    }

}

//设置cmd_vel的回调函数，将move_base转化为
void FSMState::cmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg){
   if (msg) {
        current_cmd_vel_.linear_x = msg->linear.x;
        current_cmd_vel_.linear_y = msg->linear.y;
        current_cmd_vel_.angular_z = msg->angular.z;
        current_cmd_vel_.valid = true;
        current_cmd_vel_.stamp = ros::Time::now();
        std_msgs::String receipt;
        std::ostringstream payload;
        payload << std::fixed << std::setprecision(6)
                << "{\"receipt_sequence\":" << ++stair_cmd_receipt_sequence_
                << ",\"received_stamp_sim\":" << current_cmd_vel_.stamp.toSec()
                << ",\"stored_linear_x\":" << current_cmd_vel_.linear_x
                << ",\"stored_linear_y\":" << current_cmd_vel_.linear_y
                << ",\"stored_angular_z\":" << current_cmd_vel_.angular_z
                << ",\"stored_valid\":true"
                << ",\"controller_mode\":\"" << _stateNameString << "\"}";
        receipt.data = payload.str();
        stair_cmd_receipt_pub_.publish(receipt);
    }
    // std::cout << "cmd_vel_linear_x"<< this->current_cmd_vel_.linear_x<< std::endl;
    // std::cout << "cmd_vel_linear_y"<< this->current_cmd_vel_.linear_y<< std::endl;
    // std::cout << "cmd_vel_angular_z"<< this->current_cmd_vel_.angular_z<< std::endl;
}
