"""Unit test for robosuite_interface.py script."""

# Copyright (c) 2023, ABB
# All rights reserved.
#
# Redistribution and use in source and binary forms, with
# or without modification, are permitted provided that
# the following conditions are met:
#
#   * Redistributions of source code must retain the
#     above copyright notice, this list of conditions
#     and the following disclaimer.
#   * Redistributions in binary form must reproduce the
#     above copyright notice, this list of conditions
#     and the following disclaimer in the documentation
#     and/or other materials provided with the
#     distribution.
#   * Neither the name of ABB nor the names of its
#     contributors may be used to endorse or promote
#     products derived from this software without
#     specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import random
from typing import Any
import numpy as np
from simulation.maple_benchmarks import robosuite_interface

class RobosuiteInterfaceTest(robosuite_interface.RobosuiteInterface):
    """ Simplified interface for testing """
    def __init__(self, parameters: Any = None, seed: int = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.ready_for_action = False  # At most one action each tick
        self.type = parameters.type
        self.action_type = 0
        self.actions = []
        self.use_addl_info_func = parameters.use_addl_info_func
        self.addl_infos = []
        self.image_obs_in_info = parameters.image_obs_in_info
        self.path_length = 0
        self.rewards = []
        self.env_infos = []
        self.done = False
        self.grasped_object = "none"

    def get_object_position(self, target_object):
        """ Returns object position from observation """
        if target_object == '90deg_plus_x':
            return np.array([1.0, 0.0, 0.0])
        elif target_object == '90deg_plus_y':
            return np.array([0.0, 1.0, 0.0])
        else:
            return np.array([0.0, 0.0, 0.0])
            
    def get_object_yaw(self, target_object):
        """ Returns object yaw angle from observation """
        object_yaw = 0.0
        if target_object == 'none':
            object_yaw = 0.0
        elif target_object == '90deg' or target_object == '90deg_plus_x' or target_object == '90deg_plus_y':
            object_yaw = np.pi/2
        elif target_object == 'm90deg':
            object_yaw = -np.pi/2

        return object_yaw


def test_quat_to_euler():
    """ test quat_to_euler function """
    euler = robosuite_interface.quat_to_euler([0.0, 0.0, 0.0, 1.0])
    assert (euler == [0.0, 0.0, 0.0]).all()


def test_rotate_frame_z():
    """ test rotate_frame_z function """
    rotated_frame = np.round(robosuite_interface.rotate_frame_z([1.0, 0.0, 0.0], np.pi/2), 2)
    assert (rotated_frame == [0.0, 1.0, 0.0]).all()

    rotated_frame = np.round(robosuite_interface.rotate_frame_z([1.0, 0.0, 0.0], -np.pi/2), 2)
    assert (rotated_frame == [0.0, -1.0, 0.0]).all()

    rotated_frame = np.round(robosuite_interface.rotate_frame_z([0.0, 1.0, 0.0], np.pi/2), 2)
    assert (rotated_frame == [-1.0, 0.0, 0.0]).all()


def test_get_normalized_value():
    """ test get_normalized_value function """
    assert robosuite_interface.get_normalized_value(1.0, 0.0, 1.0) == 1.0

    assert robosuite_interface.get_normalized_value(0.0, 0.0, 1.0) == -1.0

    assert robosuite_interface.get_normalized_value(0.0, -1.0, 1.0) == 0.0


def test_get_normalized_angle():
    """ test get_normalized_angle function """
    assert robosuite_interface.get_normalized_angle(np.pi) == 0.0

    assert robosuite_interface.get_normalized_angle(np.pi/2) == -1.0

    assert robosuite_interface.get_normalized_angle(-np.pi/2) == 1.0

def test_to_global_frame():
    """ test to_global_frame function """
    interface = RobosuiteInterfaceTest(robosuite_interface.RobosuiteParameters(), 0)
    position = np.array([1.0, 0.0, 0.0])

    assert (np.round(interface.to_global_frame(position, input_frame='90deg'), 3) == [0.0, 1.0, 0.0]).all()
    assert (np.round(interface.to_global_frame(position, input_frame='m90deg'), 3) == [0.0, -1.0, 0.0]).all()

    assert (np.round(interface.to_global_frame(position, input_frame='90deg_plus_x'), 3) == [1.0, 1.0, 0.0]).all()
    assert (np.round(interface.to_global_frame(position, input_frame='90deg_plus_y'), 3) == [0.0, 2.0, 0.0]).all()

def test_move_robot():
    """
    tests setting robot references and verifying that robot moves there
    tests these functions:
    set_action_type
    set_target_and_offset
    step
    get_robot_position
    at_pos

    and indirectly tests these functions:
    get_xyz_bounds
    get_normalized_pos
    """
    robosuite_parameters = robosuite_interface.RobosuiteParameters()
    robosuite_parameters.type = 'lift'
    interface = robosuite_interface.RobosuiteInterface(robosuite_parameters, 0)

    # Test moving to origin
    interface.set_action_type(robosuite_interface.ActionTypes.REACH)
    target_object = 'none'
    target_offset = [0.0, 0.0, 0.05]
    interface.set_target_and_offset(target_object, target_offset)
    interface.step()

    robot_position = np.round(interface.get_robot_position(), 2)
    assert np.linalg.norm(robot_position - [-0.075, 0.0, 0.85]) < 0.02
    assert np.linalg.norm(interface.get_robot_yaw()) < 0.2

    assert interface.at_pos(target_object, target_offset)

    # Close gripper
    interface.set_action_type(robosuite_interface.ActionTypes.ATOMIC)
    interface.set_gripper(0.99)
    for _ in range(4):
        interface.step()
    assert interface.gripper_at_ref(0.99)
    assert not interface.gripper_at_ref(-1.0)
    
    # Open gripper
    interface.set_action_type(robosuite_interface.ActionTypes.OPEN)
    interface.step()
    assert interface.gripper_at_ref(-1.0)

    # Move relative to current robot position
    target_object = 'robot_ee'
    target_offset = [0.1, 0.1, 0.0]
    robot_position = np.round(interface.get_robot_position(), 2)
    interface.set_action_type(robosuite_interface.ActionTypes.REACH)
    interface.set_target_and_offset(target_object, target_offset)
    interface.step()
    robot_position_after = np.round(interface.get_robot_position(), 2)
    assert (np.round(robot_position_after - robot_position, 1) == target_offset).all()
    assert not interface.at_pos(target_object, target_offset)

    # Move relative to cube
    target_object = 'cube'
    target_offset = [0.1, 0.1, 0.0]
    interface.set_action_type(robosuite_interface.ActionTypes.REACH)
    interface.set_target_and_offset(target_object, target_offset)
    interface.step()
    assert interface.at_pos(target_object, target_offset)

    # Grasp cube
    target_object = 'cube'
    target_offset = [0.0, 0.0, 0.0]
    cube_position = np.round(interface.get_object_position(target_object), 2)
    cube_yaw = interface.get_object_yaw(target_object)
    interface.set_action_type(robosuite_interface.ActionTypes.GRASP)
    interface.set_target_and_offset(target_object, target_offset)
    interface.set_yaw(cube_yaw)
    interface.step()
    assert interface.at_yaw(cube_yaw)

    # Lift cube and rotate
    interface.set_action_type(robosuite_interface.ActionTypes.PUSH)
    target_object = 'robot_ee'
    target_offset = [0.0, 0.0, 0.0]
    yaw_offset = 0.1
    delta_offset = [0.1, 0.0, 0.0]
    interface.set_target_and_offset(target_object, target_offset)
    interface.set_yaw(cube_yaw + yaw_offset)
    interface.set_delta_offset(delta_offset)
    interface.step()
    lifted_cube_position = np.round(interface.get_object_position('cube'), 2)
    assert (np.round(lifted_cube_position - cube_position, 1) == delta_offset).all()
    lifted_cube_yaw = np.round(interface.get_object_yaw('cube'), 2)
    assert (np.round(lifted_cube_yaw - cube_yaw, 1) == yaw_offset).all()


def test_push():
    """ Separate tests for the push action """
    robosuite_parameters = robosuite_interface.RobosuiteParameters()
    robosuite_parameters.type = 'lift'

    interface = robosuite_interface.RobosuiteInterface(robosuite_parameters, 1)
    interface.set_action_type(robosuite_interface.ActionTypes.REACH)
    interface.set_target_and_offset('none', [0.0, 0.0, 0.05])
    for _ in range(2):
        interface.step()
    robot_position = np.round(interface.get_robot_position(), 2)

    delta_offset = [0.0, 0.0, 0.0]
    interface.set_action_type(robosuite_interface.ActionTypes.PUSH)
    interface.set_target_and_offset('none', [0.0, 0.0, 0.05])
    interface.set_delta_offset(delta_offset)
    interface.step()
    robot_position1 = np.round(interface.get_robot_position(), 2)
    assert (robot_position == robot_position1).all()

    delta_offset = [-0.2, -0.2, 0.0]
    interface.set_delta_offset(delta_offset)
    interface.step()
    robot_position2 = np.round(interface.get_robot_position(), 2)
    assert (np.round(robot_position2 - robot_position, 1) == delta_offset).all()
