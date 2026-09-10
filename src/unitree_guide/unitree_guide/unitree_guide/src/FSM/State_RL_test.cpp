/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include "FSM/State_RL_test.h"

State_RL::State_RL(CtrlComponents *ctrlComp)
                :FSMState(ctrlComp, FSMStateName::RL, "RL")
{
    selected_policy_pub_ = nh.advertise<std_msgs::String>("/audit/p2kg15/selected_rl_policy", 1, true);
    load_policy();
    gravity(0,0) = 0.0;
    gravity(1,0) = 0.0;
    gravity(2,0) = -0.98;
    //在构造函数中初始化，订阅
    this->Sub_=nh.subscribe<geometry_msgs::Twist>("/cmd_vel",1000,boost::bind(&FSMState::cmdVelCallback,this,_1));
    rl_mode_ready_pub_ = nh.advertise<std_msgs::Bool>("/unitree/rl_mode_ready", 1, true);
    stair_policy_cmd_input_pub_ = nh.advertise<std_msgs::String>("/audit/stair_policy_cmd_input", 10);
    ros::NodeHandle private_nh("~");
    private_nh.param("minimum_zero_progress_policy_cmd_input_rate_hz", stair_policy_cmd_input_rate_hz_, 10.0);
    private_nh.param("p2kg15_angular_actuation_telemetry", angular_actuation_telemetry_enabled_, false);
    private_nh.param("p2kg15_angular_actuation_telemetry_rate_hz", angular_actuation_telemetry_rate_hz_, 10.0);
    if (angular_actuation_telemetry_enabled_) {
        angular_actuation_state_pub_ = nh.advertise<std_msgs::String>("/audit/p2kg15/angular_actuation_state", 10);
    }
    publishRlModeReady(false);

}


void State_RL::enter(){
     // if (real == false){
        for(int i=0; i<12; i++){
            _lowCmd->motorCmd[i].q = _lowState->motorState[i].q;
            _startPos[i] = _lowState->motorState[i].q;
            _lowCmd->motorCmd[i].mode = 10;
            _lowCmd->motorCmd[i].dq = 0;
            _lowCmd->motorCmd[i].Kp = 80;
            _lowCmd->motorCmd[i].Kd = 1;
            _lowCmd->motorCmd[i].tau = 0;
        }
        for(int i=0; i<4; i++){
             if(_ctrlComp->ctrlPlatform == CtrlPlatform::GAZEBO){
                 _lowCmd->setSimStanceGain(i);
             }
             else if(_ctrlComp->ctrlPlatform == CtrlPlatform::REALROBOT){
                 _lowCmd->setRealStanceGain(i);
             }
             _lowCmd->setZeroDq(i);
             _lowCmd->setZeroTau(i);
        }
    // }
    // else if(real == true)
    // {
        for(int i=0; i<12; i++){
            float c_joint = _ctrlComp->ioInterFreeDog->low_state.motorState_free_dog[i].q;
            std::vector<double> joint{c_joint, 0, 0, 80, 1};
            _ctrlComp->ioInterFreeDog->setCmd(i,joint);
        }
    // }
    for (int i = 0; i < HISTORY_LEN; i++)
    {
        refresh_rl_obs();
    }
    infer_thread = new std::thread(&State_RL::infer_thread_callback,this);
    infer_thread_runnning = State_RL::RUNNING;
    if (debug == true){
        amp_obs_thread = new std::thread(&State_RL::save_amp_obs_thread,this);
        ampthreadRunning = State_RL::RUNNING;
    }
    publishRlModeReady(true);
}

void State_RL::run(){
    const ros::WallTime now = ros::WallTime::now();
    if (last_rl_mode_publish_wall_.isZero() || (now - last_rl_mode_publish_wall_).toSec() >= 0.2) {
        publishRlModeReady(true);
    }
}

void State_RL::exit(){
    publishRlModeReady(false);
    _percent = 0;
    ampthreadRunning = State_RL::STOP;
    if (amp_obs_thread != nullptr && amp_obs_thread->joinable()) {
        amp_obs_thread->join();
    }
    infer_thread_runnning = State_RL::STOP;
    if (infer_thread != nullptr && infer_thread->joinable()) {
        infer_thread->join();
    }
    std::cout << "amp_obs_thread退出!" << std::endl;
    if (outfile.is_open()) {
        outfile.close();
        std::cout << "文件关闭成功!" << std::endl;
    }
}

void State_RL::publishRlModeReady(bool ready){
    std_msgs::Bool msg;
    msg.data = ready;
    rl_mode_ready_pub_.publish(msg);
    last_rl_mode_publish_wall_ = ros::WallTime::now();
}

void State_RL::publishPolicyCmdInputAudit(){
    if (stair_policy_cmd_input_rate_hz_ <= 0.0) {
        return;
    }
    const ros::WallTime wall_now = ros::WallTime::now();
    const double min_interval_s = 1.0 / stair_policy_cmd_input_rate_hz_;
    if (!last_stair_policy_cmd_input_wall_.isZero() &&
        (wall_now - last_stair_policy_cmd_input_wall_).toSec() < min_interval_s) {
        return;
    }
    const ros::Time sim_now = ros::Time::now();
    std_msgs::String message;
    std::ostringstream payload;
    payload << std::fixed << std::setprecision(6)
            << "{\"sim_stamp\":" << sim_now.toSec()
            << ",\"fsm_mode\":\"RL\""
            << ",\"source_cmd_received_stamp_sim\":"
            << (current_cmd_vel_.stamp.isZero() ? -1.0 : current_cmd_vel_.stamp.toSec())
            << ",\"source_cmd_valid\":" << (current_cmd_vel_.valid ? "true" : "false")
            << ",\"command_tensor\":[" << commands_tensor[0].item<float>() << ","
            << commands_tensor[1].item<float>() << "," << commands_tensor[2].item<float>() << "]"
            << ",\"command_scale\":[2.000000,2.000000,0.250000]"
            << ",\"scaled_command_tensor\":[" << (commands_tensor[0].item<float>() * 2.0f) << ","
            << (commands_tensor[1].item<float>() * 2.0f) << "," << (commands_tensor[2].item<float>() * 0.25f) << "]"
            << ",\"audit_rate_limit_hz\":" << stair_policy_cmd_input_rate_hz_ << "}";
    message.data = payload.str();
    stair_policy_cmd_input_pub_.publish(message);
    last_stair_policy_cmd_input_wall_ = wall_now;
}

void State_RL::publishAngularActuationTelemetry(
    float received_cmd_angular_z,
    bool received_cmd_valid,
    const ros::Time& received_cmd_stamp,
    float policy_cmd_tensor_angular_z,
    float scaled_policy_cmd_angular_z,
    const std::vector<float>& history_cmd_yaw,
    float base_yaw_rate_observation,
    const std::vector<float>& previous_actions,
    const std::vector<float>& policy_actions_raw,
    const std::vector<float>& low_level_q_targets,
    const std::vector<float>& joint_q_actual,
    const std::vector<float>& joint_dq_actual) {
    if (!angular_actuation_telemetry_enabled_ || angular_actuation_telemetry_rate_hz_ <= 0.0) {
        return;
    }
    const ros::WallTime wall_now = ros::WallTime::now();
    const double min_interval_s = 1.0 / angular_actuation_telemetry_rate_hz_;
    if (!last_angular_actuation_telemetry_wall_.isZero() &&
        (wall_now - last_angular_actuation_telemetry_wall_).toSec() < min_interval_s) {
        return;
    }

    const auto summarize = [](const std::vector<float>& values, float* min_value, float* max_value, float* l1) {
        *min_value = 0.0f;
        *max_value = 0.0f;
        *l1 = 0.0f;
        if (values.empty()) {
            return;
        }
        *min_value = values.front();
        *max_value = values.front();
        for (const float value : values) {
            *min_value = std::min(*min_value, value);
            *max_value = std::max(*max_value, value);
            *l1 += std::abs(value);
        }
    };
    float previous_min = 0.0f, previous_max = 0.0f, previous_l1 = 0.0f;
    float action_min = 0.0f, action_max = 0.0f, action_l1 = 0.0f;
    float target_min = 0.0f, target_max = 0.0f, target_l1 = 0.0f;
    summarize(previous_actions, &previous_min, &previous_max, &previous_l1);
    summarize(policy_actions_raw, &action_min, &action_max, &action_l1);
    summarize(low_level_q_targets, &target_min, &target_max, &target_l1);

    const ros::Time sim_now = ros::Time::now();
    const double cmd_age_sim_s =
        (received_cmd_valid && !received_cmd_stamp.isZero())
            ? std::max(0.0, (sim_now - received_cmd_stamp).toSec())
            : -1.0;
    const auto append_array = [](std::ostringstream* stream, const char* key, const std::vector<float>& values) {
        *stream << ",\"" << key << "\":[";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0) {
                *stream << ",";
            }
            *stream << values[i];
        }
        *stream << "]";
    };
    std_msgs::String message;
    std::ostringstream payload;
    payload << std::fixed << std::setprecision(6)
            << "{\"sim_stamp\":" << sim_now.toSec()
            << ",\"fsm_mode\":\"RL\""
            << ",\"received_cmd_angular_z\":" << received_cmd_angular_z
            << ",\"received_cmd_valid\":" << (received_cmd_valid ? "true" : "false")
            << ",\"received_cmd_stamp_sim\":"
            << (received_cmd_stamp.isZero() ? -1.0 : received_cmd_stamp.toSec())
            << ",\"received_cmd_age_sim_s\":" << cmd_age_sim_s
            << ",\"policy_cmd_tensor_angular_z\":" << policy_cmd_tensor_angular_z
            << ",\"scaled_policy_cmd_angular_z\":" << scaled_policy_cmd_angular_z
            << ",\"base_yaw_rate_observation\":" << base_yaw_rate_observation
            << ",\"observation_history_length\":" << HISTORY_LEN
            << ",\"previous_action_min\":" << previous_min
            << ",\"previous_action_max\":" << previous_max
            << ",\"previous_action_l1\":" << previous_l1
            << ",\"policy_action_min\":" << action_min
            << ",\"policy_action_max\":" << action_max
            << ",\"policy_action_l1\":" << action_l1
            << ",\"low_level_target_min\":" << target_min
            << ",\"low_level_target_max\":" << target_max
            << ",\"low_level_target_l1\":" << target_l1;
    append_array(&payload, "history_cmd_yaw", history_cmd_yaw);
    append_array(&payload, "previous_action", previous_actions);
    append_array(&payload, "policy_action_raw", policy_actions_raw);
    append_array(&payload, "low_level_q_target", low_level_q_targets);
    append_array(&payload, "joint_q_actual", joint_q_actual);
    append_array(&payload, "joint_dq_actual", joint_dq_actual);
    payload << ",\"wave_contact\":[";
    for (int i = 0; i < 4; ++i) {
        if (i > 0) {
            payload << ",";
        }
        payload << (*_ctrlComp->contact)(i);
    }
    payload << "],\"wave_phase\":[";
    for (int i = 0; i < 4; ++i) {
        if (i > 0) {
            payload << ",";
        }
        payload << (*_ctrlComp->phase)(i);
    }
    payload << "]"
            << ",\"telemetry_rate_limit_hz\":" << angular_actuation_telemetry_rate_hz_
            << "}";
    message.data = payload.str();
    angular_actuation_state_pub_.publish(message);
    last_angular_actuation_telemetry_wall_ = wall_now;
}

FSMStateName State_RL::checkChange(){
    if(_lowState->userCmd == UserCommand::L2_B){
        return FSMStateName::PASSIVE;
    }
    else if(_lowState->userCmd == UserCommand::L2_A){
        return FSMStateName::FIXEDSTAND;
    }
    else if(_lowState->userCmd == UserCommand::L1_X){
        if (_last_cmd==static_cast<int>(UserCommand::RL))
        {
            _cnt = (_cnt+1)%(sizeof(_targetPos_map) / sizeof(_targetPos_map[0]));
            if (real == false){
                for(int i=0; i<12; i++){
                    _lowCmd->motorCmd[i].q = _lowState->motorState[i].q;
                    _startPos[i] = _lowState->motorState[i].q;
                }
            }
            else if(real == true){
                for(int i=0; i<12; i++){
                    _startPos[i] = _ctrlComp->ioInterFreeDog->low_state.motorState_free_dog[i].q;
                }
            }
            _percent = 0;
            std::cout << "cnt: " << _cnt << std::endl;
            // open_amp_save_file();
            dofPosSwitBeginTime = getTime();
        }
        _last_cmd = static_cast<int>(_lowState->userCmd);
        return FSMStateName::RL;
    }
    else{
        _last_cmd = static_cast<int>(_lowState->userCmd);
        return FSMStateName::RL;
    }
}

void State_RL::infer_thread_callback()
{
    while(infer_thread_runnning == State_RL::RUNNING)
    {
        long long _start_time = getTime();
        // std::cout << "_start_time" << _start_time << std::endl;
        refresh_rl_obs();
        float received_cmd_angular_z = 0.0f;
        bool received_cmd_valid = false;
        ros::Time received_cmd_stamp;
        float policy_cmd_tensor_angular_z = 0.0f;
        float scaled_policy_cmd_angular_z = 0.0f;
        float base_yaw_rate_observation = 0.0f;
        std::vector<float> history_cmd_yaw;
        std::vector<float> previous_actions;
        if (angular_actuation_telemetry_enabled_) {
            // Callback-side snapshot: it can change independently of this inference cycle.
            received_cmd_angular_z = current_cmd_vel_.angular_z;
            received_cmd_valid = current_cmd_vel_.valid;
            received_cmd_stamp = current_cmd_vel_.stamp;
            // Inference-cycle snapshots: refresh_rl_obs has already populated these tensors.
            policy_cmd_tensor_angular_z = commands_tensor[2].item<float>();
            scaled_policy_cmd_angular_z = (commands_tensor[2] * commands_scale[2]).item<float>();
            base_yaw_rate_observation = base_ang_vel_tensor[2].item<float>();
            const torch::Tensor history_cpu = obs_history_tensor.to(torch::kCPU).contiguous();
            const float* history_data = history_cpu.data_ptr<float>();
            const int yaw_command_offset = 8;  // 3 angular velocity + 3 gravity + yaw in 3 command slots.
            history_cmd_yaw.reserve(HISTORY_LEN);
            for (int row = 0; row < HISTORY_LEN; ++row) {
                history_cmd_yaw.push_back(history_data[row * 45 + yaw_command_offset]);
            }
            previous_actions.assign(
                actions_tensor.data_ptr<float>(),
                actions_tensor.data_ptr<float>() + actions_tensor.numel());
        }
        torch::Tensor flattened_obs = obs_history_tensor.view({1, HISTORY_LEN * 45});
        if (debug == true)
        {
            const std::vector<int> sub_sizes = {3, 3, 3, 12, 12, 12};
            int segment_size = 45;
            // std::cout << "printSegments" << std::endl;
            // printSegments(flattened_obs.squeeze(), segment_size, sub_sizes);
        }
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(flattened_obs);
        // std::cout << "flattened_obs: " << flattened_obs << std::endl;
        actions_tensor = model.get_method("act_inference")(inputs).toTensor().to(torch::kCPU).squeeze();
        std::vector<float> policy_actions_raw;
        if (angular_actuation_telemetry_enabled_) {
            policy_actions_raw.assign(
                actions_tensor.data_ptr<float>(),
                actions_tensor.data_ptr<float>() + actions_tensor.numel());
        }
        if (debug==true){
            torch::Tensor input_tensor = torch::arange(1, 226).view({1, 225}).to(torch::kFloat32).to(device); // 注意范围是 [start, end)
            std::vector<torch::jit::IValue> test;
            test.push_back(input_tensor);
            torch::Tensor output = model.get_method("act_inference")(test).toTensor().to(torch::kCPU).squeeze();
            // printTensorHorizontal(output, "same_net_work_test");
        }
        actions_tensor_scaled = actions_tensor.clone() * 0.25;
        std::vector<float> actions(actions_tensor_scaled.data_ptr<float>(),
                           actions_tensor_scaled.data_ptr<float>() + actions_tensor_scaled.numel());
        std::vector<float> low_level_q_targets;
        std::vector<float> joint_q_actual;
        std::vector<float> joint_dq_actual;
        if (debug == true) std::cout << "actions[reindex[j]]  + default_dof_pos" << std::endl;
        for(int i=0; i<12; i++){
            if (real == false)
            {
                const float target = actions[reindex[i]]  + default_dof_pos_tensor[reindex[i]].item<float>();
                _lowCmd->motorCmd[i].q = target;
                _lowCmd->motorCmd[i].Kp = 80;
                _lowCmd->motorCmd[i].Kd = 1;
            }
            else if (real == true)
            {
                // float t_joint = actions[reindex[i]]  + default_dof_pos_tensor[reindex[i]].item<float>();
                // std::vector<double> joint{t_joint, 0, 0, 80, 1};
                // _ctrlComp->ioInterFreeDog->setCmd(i,joint);
            }
            if (debug == true) std::cout << actions[reindex[i]]  + default_dof_pos_tensor[reindex[i]].item<float>() << " ";
        }
        if (debug == true)
            std::cout << std::endl;
        if (angular_actuation_telemetry_enabled_) {
            low_level_q_targets.reserve(12);
            joint_q_actual.reserve(12);
            joint_dq_actual.reserve(12);
            for (int i = 0; i < 12; ++i) {
                low_level_q_targets.push_back(_lowCmd->motorCmd[i].q);
                joint_q_actual.push_back(_lowState->motorState[i].q);
                joint_dq_actual.push_back(_lowState->motorState[i].dq);
            }
            publishAngularActuationTelemetry(
                received_cmd_angular_z,
                received_cmd_valid,
                received_cmd_stamp,
                policy_cmd_tensor_angular_z,
                scaled_policy_cmd_angular_z,
                history_cmd_yaw,
                base_yaw_rate_observation,
                previous_actions,
                policy_actions_raw,
                low_level_q_targets,
                joint_q_actual,
                joint_dq_actual);
        }
        // std::cout << "actions_tensor: " << actions_tensor << std::endl;
        wait(_start_time, (long long)(infer_duration * 1000000));
    }
    infer_thread_runnning = State_RL::OVER;
}

void State_RL::save_amp_obs_thread()
{
    while(ampthreadRunning == State_RL::RUNNING)
    {
        long long _start_time = getTime();
        if ((getTime() - dofPosSwitBeginTime)<_duration) {
            _percent = (float)(getTime() - dofPosSwitBeginTime)/_duration;
            _percent = _percent > 1 ? 1 : _percent;
            std::cout << "_percent" << _percent << std::endl;
            // if (real == false){
                std::cout << "_lowCmd->motorCmd ";
                for(int j=0; j<12; j++){
                    std::cout << _targetPos_map[_cnt][reindex[j]] << " ";
                    _lowCmd->motorCmd[j].q = (1 - _percent)*_startPos[j] + _percent*_targetPos_map[_cnt][reindex[j]];
                }
                std::cout << _lowCmd->motorCmd << std::endl;
                std::cout << std::endl;
            // }
            // else if (real == true){
                std::cout << "target_joint";
                for(int j=0; j<12; j++){
                    std::cout << _targetPos_map[_cnt][j] << " ";
                    float t_joint = (1 - _percent)*_startPos[j] + _percent*_targetPos_map[_cnt][reindex[j]];
                    std::vector<double> joint{t_joint, 0, 0, 80, 1};
                    _ctrlComp->ioInterFreeDog->setCmd(j,joint);
                }
                std::cout << std::endl;
            // }
            if ((float)(getTime() - dofPosSwitBeginTime)>(float)_duration*0.95)
                close_amp_save_file();
        }
        if (outfile.is_open())
        {
            std::cout << "save data" << std::endl;
            refresh_amp_obs();
        }
        wait(_start_time, (long long)(infer_duration * 1000000));
    }
    ampthreadRunning = State_RL::OVER;
}



void State_RL::refresh_rl_obs(){
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    //gazebo simulation mode
    if (real == false)
    {
        for (int i=0; i<4; i++) {
            base_w_orientation[i] = _ctrlComp->ioInter->_base_w_ori[i];
        }
        for (int i=0; i<3; i++) {
            base_w_angular_vel[i] = _ctrlComp->ioInter->_base_w_angular_vel[i];
        }
        torch::Tensor orientation_tensor = torch::from_blob(base_w_orientation.data(), {int64_t(base_w_orientation.size())}, opts).unsqueeze(0).clone();
        torch::Tensor w_angular_vel_tensor = torch::from_blob(base_w_angular_vel.data(), {int64_t(base_w_angular_vel.size())}, opts).unsqueeze(0).clone();
        base_ang_vel_tensor = quat_rotate_inverse(orientation_tensor, w_angular_vel_tensor).squeeze().clone();
        projected_gravity_tensor = quat_rotate_inverse(orientation_tensor, gravity_tensor.unsqueeze(0)).squeeze().clone();
        
        //订阅cmd_vel
        // this->Sub_=nh.subscribe<geometry_msgs::Twist>("/cmd_vel",1000,boost::bind(&FSMState::cmdVelCallback,this,_1));

        // commands_tensor[0] = _ctrlComp->ioInter->axes[1];
        // commands_tensor[1] = _ctrlComp->ioInter->axes[0];
        // commands_tensor[2] = _ctrlComp->ioInter->axes[3]*3.14;

        commands_tensor[0] = this->current_cmd_vel_.linear_x;
        commands_tensor[1] = this->current_cmd_vel_.linear_y;
        commands_tensor[2] = this->current_cmd_vel_.angular_z;
        publishPolicyCmdInputAudit();


        // std::cout << _ctrlComp->ioInter->axes << std::endl;
        // std::cout << "commands_tensor: " << commands_tensor << std::endl;
        for(int i=0; i<12; i++){
            joint_pos[i] = _lowState->motorState[reindex[i]].q;
        }
        dof_pos_tensor = torch::from_blob(joint_pos.data(), {int64_t(joint_pos.size())}, opts).clone();
        // printTensorHorizontal(dof_pos_tensor,"dof_pos_tensor");
        for(int i=0; i<12; i++){
            joint_vel[i] = _lowState->motorState[reindex[i]].dq;
        }
        dof_vel_tensor = torch::from_blob(joint_vel.data(), {int64_t(joint_vel.size())}, opts).clone();
        obs_tensor = torch::cat({
            base_ang_vel_tensor * obs_scales_ang_vel,
            projected_gravity_tensor,
            commands_tensor * commands_scale,
            (dof_pos_tensor - default_dof_pos_tensor) * obs_scales_dof_pos,
            dof_vel_tensor * obs_scales_dof_vel,
            actions_tensor
        }, -1).to(device);
        obs_history_tensor = torch::cat({
            obs_history_tensor.slice(0, 1, HISTORY_LEN).to(device),  // 删除最早的一步
            obs_tensor.unsqueeze(0)  // 将当前 obs_tensor 插入到历史中
        }, 0);  // 按行（第0维）拼接
    }
    else if (real == true)
    {
        _B2G_RotMat = _ctrlComp->ioInterFreeDog->getRotMat();
        _G2B_RotMat = _B2G_RotMat.transpose();
        Vec3 projected_gravity = _G2B_RotMat*gravity;
        projected_gravity_tensor = torch::tensor({projected_gravity(0,0), projected_gravity(1,0), projected_gravity(2,0)});
         for (int i=0; i<3; i++) {
            base_ang_vel_tensor[i] = _ctrlComp->ioInterFreeDog->low_state.imu_gyroscope[i];
        }
        commands_tensor[0] = _ctrlComp->ioInter->axes[1];
        commands_tensor[1] = _ctrlComp->ioInter->axes[0];
        commands_tensor[2] = _ctrlComp->ioInter->axes[3]*3.14;
        for(int i=0; i<12; i++){
            joint_pos[i] = _ctrlComp->ioInterFreeDog->low_state.motorState_free_dog[reindex[i]].q;
        }
        dof_pos_tensor = torch::from_blob(joint_pos.data(), {int64_t(joint_pos.size())}, opts).clone();
        for(int i=0; i<12; i++){
            joint_vel[i] = _ctrlComp->ioInterFreeDog->low_state.motorState_free_dog[reindex[i]].dq;
        }
        dof_vel_tensor = torch::from_blob(joint_vel.data(), {int64_t(joint_vel.size())}, opts).clone();
        obs_tensor = torch::cat({
            base_ang_vel_tensor * obs_scales_ang_vel,
            projected_gravity_tensor,
            commands_tensor * commands_scale,
            (dof_pos_tensor - default_dof_pos_tensor) * obs_scales_dof_pos,
            dof_vel_tensor * obs_scales_dof_vel,
            actions_tensor
        }, -1).to(device);
        obs_history_tensor = torch::cat({
            obs_history_tensor.slice(0, 1, HISTORY_LEN).to(device),  // 删除最早的一步
            obs_tensor.unsqueeze(0)  // 将当前 obs_tensor 插入到历史中
        }, 0);  // 按行（第0维）拼接
    }
}


void State_RL::refresh_rl_obs_real_robot(){

}

void State_RL::refresh_amp_obs(){
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    motion_time = static_cast<float>(getRosTime() - dofPosSwitBeginTime)/1e6;
    outfile << "motion_time: " << motion_time << std::endl;
    outfile << "base_w_pos: ";
    for (int i=0; i<3; i++) {
        base_w_pos[i] = _ctrlComp->ioInter->_base_w_pos[i];
        outfile << base_w_pos[i] << " ";
    }
    outfile << std::endl;

    outfile << "base_ori: ";
    for (int i=0; i<4; i++) {
        base_w_orientation[i] = _ctrlComp->ioInter->_base_w_ori[i];
        outfile << base_w_orientation[i] << " ";
    }
    outfile << std::endl;

    outfile << "dof_pos: ";
    for(int i=0; i<12; i++){
        joint_pos[i] = _lowState->motorState[reindex[i]].q;
        outfile << joint_pos[i] << " ";
    }
    outfile << std::endl;

    outfile << "foot_pos: ";
    for (int i=0; i<3; i++){
        foot_pos[0*3+i] = _ctrlComp->ioInter->_FL_foot_pos[i];
        foot_vel[0*3+i] = _ctrlComp->ioInter->_FL_foot_vel[i];
    }
    for (int i=0; i<3; i++){
        foot_pos[1*3+i] = _ctrlComp->ioInter->_FR_foot_pos[i];
        foot_vel[1*3+i] = _ctrlComp->ioInter->_FR_foot_vel[i];
    }
    for (int i=0; i<3; i++){
        foot_pos[2*3+i] = _ctrlComp->ioInter->_RL_foot_pos[i];
        foot_vel[2*3+i] = _ctrlComp->ioInter->_RL_foot_vel[i];
    }
    for (int i=0; i<3; i++){
        foot_pos[3*3+i] = _ctrlComp->ioInter->_RR_foot_pos[i];
        foot_vel[3*3+i] = _ctrlComp->ioInter->_RR_foot_vel[i];
    }
    for (  int  i=0;i<12;i++ )
    {
        outfile << foot_pos[i] << " ";
    }
    outfile << std::endl;

    outfile << "base_w_linear_vel: ";
    for (int i=0; i<3; i++) {
        base_w_linear_vel[i] = _ctrlComp->ioInter->_base_w_linear_vel[i];
    }
    torch::Tensor orientation_tensor = torch::from_blob(base_w_orientation.data(), {int64_t(base_w_orientation.size())}, opts).unsqueeze(0).clone();
    torch::Tensor w_linear_vel_tensor = torch::from_blob(base_w_linear_vel.data(), {int64_t(base_w_linear_vel.size())}, opts).unsqueeze(0).clone();
    torch::Tensor result = quat_rotate_inverse(orientation_tensor, w_linear_vel_tensor).squeeze().clone();
    for (int i = 0; i < 3; ++i) {
        base_linear_vel[i] = result[i].item<float>();
        // std::cout << base_linear_vel[i] << " ";
        outfile << base_linear_vel[i] << " ";
    }
    outfile << std::endl;

    outfile << "base_w_angular_vel: ";
    for (int i=0; i<3; i++) {
        base_w_angular_vel[i] = _ctrlComp->ioInter->_base_w_angular_vel[i];
    }
    torch::Tensor w_angular_vel_tensor = torch::from_blob(base_w_angular_vel.data(), {int64_t(base_w_angular_vel.size())}, opts).unsqueeze(0).clone();
    result = quat_rotate_inverse(orientation_tensor, w_angular_vel_tensor).squeeze().clone();
    for (int i = 0; i < 3; ++i) {
        base_angular_vel[i] = result[i].item<float>();
        // std::cout << base_angular_vel[i] << " ";
        outfile << base_angular_vel[i] << " ";
    }
    outfile << std::endl;

    outfile << "dof_vel: ";
    for(int i=0; i<12; i++){
        joint_vel[i] = _lowState->motorState[reindex[i]].dq;
        outfile << joint_vel[i] << " ";
    }
    outfile << std::endl;

    outfile << "foot_vel";
    for (  int  i=0;i<12;i++ )
    {
        outfile << foot_vel[i] << " ";
    }
    outfile << std::endl;
    outfile << std::endl;
    outfile << std::endl;
}

void State_RL::open_amp_save_file()
{
    // 打开文件输出流
    // 获取当前系统时间
    std::time_t cTime = std::time(nullptr);
    std::tm* currentTm = std::localtime(&cTime);
    // 构建文件名，格式为 systime + 年-月-日.txt
    std::ostringstream fileNameStream;
    fileNameStream << "/home/chy/log/gazebo/" << angle_names[_cnt];
    std::string fileName = fileNameStream.str();
    // 以追加模式打开文件
    outfile = std::ofstream(fileName, std::ios::out | std::ios::app);
    if (!outfile) {
        std::cerr << "无法打开文件!" << std::endl;
    } else {
        // std::cout << "文件打开成功!" << std::endl;
    }
}

void State_RL::close_amp_save_file()
{
    if (outfile.is_open()) {
        outfile.close();
        // std::cout << "文件关闭保存成功!" << std::endl;
    }
}

torch::Tensor State_RL::quat_rotate_inverse(const torch::Tensor& q, const torch::Tensor& v) {
    // Ensure q and v are of the correct shape: (batch_size, 4) for quaternions and (batch_size, 3) for vectors
    auto shape = q.sizes();
    // std::cout << "shape: " << shape << std::endl;
    auto q_w = q.index({torch::indexing::Slice(), 3});  // last column is the w component
    // std::cout << "q_w: " << q_w << std::endl;
    auto q_vec = q.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)});  // first three columns are the vector part
    // std::cout << "q_vec: " << q_vec << std::endl;
    // a = v * (2.0 * q_w^2 - 1.0).unsqueeze(-1)
    auto a = v * (2.0 * q_w.pow(2) - 1.0).unsqueeze(-1);
    // std::cout << "a: " << a << std::endl;
    // b = cross(q_vec, v) * q_w.unsqueeze(-1) * 2.0
    auto b = torch::cross(q_vec, v, /*dim=*/-1) * q_w.unsqueeze(-1) * 2.0;
    // std::cout << "b: " << b << std::endl;
    // c = q_vec * torch::bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    auto q_vec_reshaped = q_vec.view({shape[0], 1, 3});
    // std::cout << "q_vec_reshaped: " << q_vec_reshaped << std::endl;
    auto v_reshaped = v.view({shape[0], 3, 1});
    // std::cout << "v_reshaped: " << v_reshaped << std::endl;
    auto c = q_vec * torch::bmm(q_vec_reshaped, v_reshaped).squeeze(-1) * 2.0;
    // std::cout << "c: " << c << std::endl;
    // Return a - b + c
    // std::cout << "a - b + c: " << a - b + c << std::endl;
    return a - b + c;
}

void State_RL::load_policy()
{
    const char *requested_policy = std::getenv("UNITREE_RL_POLICY");
    selected_policy_name_ = requested_policy == nullptr ? "stair" : requested_policy;
    if (selected_policy_name_ == "stair") {
        selected_policy_path_ = "src/unitree_guide/logs/policy_act_inference_stair.pt";
        selected_policy_sha256_ = "2d5aa72511c0c6609c02f4105845eee6974d3d73431497f8f35306da9588fe14";
    } else if (selected_policy_name_ == "plane") {
        selected_policy_path_ = "src/unitree_guide/logs/policy_act_inference_plane.pt";
        selected_policy_sha256_ = "e886847fe266e3c2f7c08825fceeaecfa75c7eac5f780b25b6d4dca173ff8bef";
    } else {
        throw std::runtime_error(
            "invalid UNITREE_RL_POLICY='" + selected_policy_name_ + "'; expected exactly 'stair' or 'plane'");
    }
    model_path = selected_policy_path_;
    std::cout << model_path << std::endl;
    // load model from check point
    std::cout << "cuda::is_available():" << torch::cuda::is_available() << std::endl;
    device= torch::kCPU;
    if (torch::cuda::is_available()){
        device = torch::kCUDA;
    }
    model = torch::jit::load(model_path);
    std::cout << "load model is successed!" << std::endl;
    model.to(device);
    std::cout << "load model to device!" << std::endl;
    model.eval();
    std_msgs::String selected_policy;
    selected_policy.data = "{\\\"selected_policy\\\":\\\"" + selected_policy_name_
        + "\\\",\\\"model_path\\\":\\\"" + selected_policy_path_
        + "\\\",\\\"sha256\\\":\\\"" + selected_policy_sha256_ + "\\\"}";
    selected_policy_pub_.publish(selected_policy);
    std::cout << "selected_policy=" << selected_policy_name_
              << " model_path=" << selected_policy_path_
              << " sha256=" << selected_policy_sha256_ << std::endl;
}

void State_RL::printSegments(const torch::Tensor& tensor, int segment_size, const std::vector<int>& sub_sizes) {
    int num_segments = tensor.size(0) / segment_size;
    std::cout << "num_segments" << num_segments << tensor.size(0) << segment_size << std::endl;
    for (int seg = 0; seg < num_segments; ++seg) {
        auto segment = tensor.slice(0, seg * segment_size, (seg + 1) * segment_size);
        std::cout << "Segment " << seg + 1 << ":\n";

        int start = 0;
        for (size_t i = 0; i < sub_sizes.size(); ++i) {
            int size = sub_sizes[i];
            auto sub_segment = segment.slice(0, start, start + size);  // 按列（第1维）分割
            std::cout << "  Sub-segment " << i + 1 << " (" << size << " elements): ";
            std::string output_str = "  Sub-segment " + std::to_string(i + 1);
            printTensorHorizontal(sub_segment, output_str);
            start += size;
        }
    }
}

// 横排打印函数
void State_RL::printTensorHorizontal(const torch::Tensor& tensor, const std::string& name) {
    std::cout << name << " (" << tensor.sizes() << "): [ ";
    auto tensor_cpu = tensor.to(torch::kCPU);  // 确保张量在 CPU 上
    auto accessor = tensor_cpu.accessor<float, 1>();  // 假设是一维张量

    for (int i = 0; i < tensor.size(0); ++i) {
        std::cout << accessor[i] << " ";
    }
    std::cout << "]\n";
}
