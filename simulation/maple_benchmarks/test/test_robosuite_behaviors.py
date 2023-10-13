"""Unit test for robosuite_behaviors.py script."""

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
import py_trees as pt
from simulation.maple_benchmarks import robosuite_interface
import simulation.maple_benchmarks.robosuite_behaviors as behaviors


def test_at_pos():
    """ test AtPos behavior"""
    robosuite_parameters = robosuite_interface.RobosuiteParameters()
    robosuite_parameters.type = 'lift'
    interface = robosuite_interface.RobosuiteInterface(robosuite_parameters, 0)
    behavior = behaviors.AtPos('at', ['cube', 'none', [0.0, 0.0, 0.05], 0.0], interface)
    behavior.initialise()
    assert behavior.update() == pt.common.Status.FAILURE

    grasp_behavior = behaviors.Grasp('grasp', ['cube', [0.0, 0.0, 0.0], 0.0], interface)
    grasp_behavior.initialise()
    grasp_behavior.update()
    interface.step()

    atomic_behavior = behaviors.Atomic('atomic', ['cube', [0.0, 0.0, 0.05], 0.0, 0.0], interface)
    atomic_behavior.initialise()
    for _ in range(15):
        atomic_behavior.update()
        interface.step()

    assert behavior.update() == pt.common.Status.SUCCESS

    behavior_string = 'cube at none (1.0, 2.0, 3.0)'
    parameters = behaviors.AtPos.parse_parameters(behavior_string)
    assert parameters[0] == 'cube'
    assert parameters[1] == 'none'
    assert parameters[2] == (1.0, 2.0, 3.0)

    behavior_string = 'cube at none'
    parameters = behaviors.AtPos.parse_parameters(behavior_string)
    assert parameters[0] == 'cube'
    assert parameters[1] == 'none'
    assert parameters[2] == (0.0, 0.0, 0.0)


def test_at_pos_free():
    """ test AtPosFree behavior"""
    robosuite_parameters = robosuite_interface.RobosuiteParameters()
    robosuite_parameters.type = 'lift'
    interface = robosuite_interface.RobosuiteInterface(robosuite_parameters, 0)
    behavior = behaviors.AtPosFree('at', ['cube', 'none', [0.0, 0.0, 0.05], 0.0], interface)
    behavior.initialise()
    assert behavior.update() == pt.common.Status.FAILURE

    grasp_behavior = behaviors.Grasp('grasp', ['cube', [0.0, 0.0, 0.0], 0.0], interface)
    grasp_behavior.initialise()
    grasp_behavior.update()
    interface.step()
    grasp_behavior.update()

    atomic_behavior = behaviors.Atomic('atomic', ['cube', [0.0, 0.0, 0.05], 0.0, 0.0], interface)
    atomic_behavior.initialise()
    for _ in range(15):
        atomic_behavior.update()
        interface.step()

    assert behavior.update() == pt.common.Status.FAILURE

    behavior_string = 'cube at none (1.0, 2.0, 3.0)'
    parameters = behaviors.AtPosFree.parse_parameters(behavior_string)
    assert parameters[0] == 'cube'
    assert parameters[1] == 'none'
    assert parameters[2] == (1.0, 2.0, 3.0)

    behavior_string = 'cube at none'
    parameters = behaviors.AtPosFree.parse_parameters(behavior_string)
    assert parameters[0] == 'cube'
    assert parameters[1] == 'none'
    assert parameters[2] == (0.0, 0.0, 0.0)


def test_poscheck():
    """ test PosCheck behavior """
    robosuite_parameters = robosuite_interface.RobosuiteParameters()
    robosuite_parameters.type = 'lift'
    interface = robosuite_interface.RobosuiteInterface(robosuite_parameters, 0)
    behavior = behaviors.PosCheck('pos', ['cube', 0, 0.5, True], interface)
    behavior.initialise()
    assert behavior.update() == pt.common.Status.FAILURE

    behavior = behaviors.PosCheck('pos', ['cube', 0, -0.5, True], interface)
    behavior.initialise()
    assert behavior.update() == pt.common.Status.SUCCESS

    behavior = behaviors.PosCheck('pos', ['cube', 1, 0.5, True], interface)
    behavior.initialise()
    assert behavior.update() == pt.common.Status.FAILURE

    behavior = behaviors.PosCheck('pos', ['cube', 1, -0.5, True], interface)
    behavior.initialise()
    assert behavior.update() == pt.common.Status.SUCCESS

    behavior = behaviors.PosCheck('pos', ['cube', 2, 1.5, True], interface)
    behavior.initialise()
    assert behavior.update() == pt.common.Status.FAILURE

    behavior = behaviors.PosCheck('pos', ['cube', 2, -0.1, True], interface)
    behavior.initialise()
    assert behavior.update() == pt.common.Status.SUCCESS

    behavior_string = 'pos cube.0 > 5.5'
    parameters = behaviors.PosCheck.parse_parameters(behavior_string)
    assert parameters[0] == 'cube'
    assert parameters[1] == 0
    assert parameters[2] == 5.5


def test_handle_angle():
    """ test HandleAngle behavior """
    robosuite_parameters = robosuite_interface.RobosuiteParameters()
    robosuite_parameters.type = 'door'
    interface = robosuite_interface.RobosuiteInterface(robosuite_parameters, 0)
    behavior = behaviors.HandleAngle('handle angle', [0.5], interface)
    behavior.initialise()
    assert behavior.update() == pt.common.Status.FAILURE

    behavior = behaviors.HandleAngle('handle angle', [-0.5], interface)
    behavior.initialise()
    assert behavior.update() == pt.common.Status.SUCCESS

    behavior_string = 'handle angle > 0.5'
    parameters = behaviors.HandleAngle.parse_parameters(behavior_string)
    assert parameters[0] == 0.5


def test_door_angle():
    """ test DoorAngle behavior """
    robosuite_parameters = robosuite_interface.RobosuiteParameters()
    robosuite_parameters.type = 'door'
    interface = robosuite_interface.RobosuiteInterface(robosuite_parameters, 0)
    behavior = behaviors.DoorAngle('door angle', [0.5], interface)
    behavior.initialise()
    assert behavior.update() == pt.common.Status.FAILURE

    behavior = behaviors.DoorAngle('door angle', [-0.5], interface)
    behavior.initialise()
    assert behavior.update() == pt.common.Status.SUCCESS

    behavior_string = 'door angle > 0.5'
    parameters = behaviors.DoorAngle.parse_parameters(behavior_string)
    assert parameters[0] == 0.5


def test_anglecheck():
    """ test AngleCheck behavior """
    robosuite_parameters = robosuite_interface.RobosuiteParameters()
    robosuite_parameters.type = 'lift'
    interface = robosuite_interface.RobosuiteInterface(robosuite_parameters, 0)
    behavior = behaviors.AngleCheck('angle', ['cube', 0.0], interface)
    behavior.initialise()
    assert behavior.update() == pt.common.Status.FAILURE

    grasp_behavior = behaviors.Grasp('grasp', ['cube', [0.0, 0.0, 0.0], 0.0], interface)

    grasp_behavior.initialise()
    grasp_behavior.update()
    interface.step()

    atomic_behavior = behaviors.Atomic('atomic', ['cube', [0.0, 0.0, 0.05], 0.0, 0.0], interface)

    atomic_behavior.initialise()
    for _ in range(15):
        atomic_behavior.update()
        interface.step()

    assert behavior.update() == pt.common.Status.SUCCESS

    behavior_string = 'cube angle == 5.5'
    parameters = behaviors.AngleCheck.parse_parameters(behavior_string)
    assert parameters[0] == 'cube'
    assert parameters[1] == 5.5

    behavior_string = 'cube angle =='
    parameters = behaviors.AngleCheck.parse_parameters(behavior_string)
    assert parameters[0] == 'cube'
    assert parameters[1] == 0.0


def test_atomic():
    """ test Atomic behavior """
    robosuite_parameters = robosuite_interface.RobosuiteParameters()
    robosuite_parameters.type = 'lift'
    interface = robosuite_interface.RobosuiteInterface(robosuite_parameters, 0)
    behavior = behaviors.Atomic('atomic', ['robot_ee', [0.1, 0.1, 0.05], 0.2, 0.0], interface)

    behavior.initialise()
    for _ in range(10):
        behavior.update()
        interface.step()
    assert behavior.update() == pt.common.Status.SUCCESS

    behavior_string = 'atomic none (1.0, 2.0, 3.0) 4.0 5.0'
    parameters = behaviors.Atomic.parse_parameters(behavior_string)
    assert parameters[0] == 'none'
    assert parameters[1] == (1.0, 2.0, 3.0)
    assert parameters[2] == 4.0
    assert parameters[3] == 5.0

    behavior_string = 'atomic none'
    parameters = behaviors.Atomic.parse_parameters(behavior_string)
    assert parameters[0] == 'none'
    assert parameters[1] == (0.0, 0.0, 0.0)
    assert parameters[2] == 0.0
    assert parameters[3] == 0.0


def test_reach():
    """ test Reach behavior """
    robosuite_parameters = robosuite_interface.RobosuiteParameters()
    robosuite_parameters.type = 'lift'
    interface = robosuite_interface.RobosuiteInterface(robosuite_parameters, 0)
    behavior = behaviors.Reach('reach', ['none', [0.0, 0.0, 0.05]], interface)

    behavior.initialise()
    behavior.update()
    interface.step()
    assert behavior.update() == pt.common.Status.SUCCESS

    behavior_string = 'reach none (1.0, 2.0, 3.0)'
    parameters = behaviors.Reach.parse_parameters(behavior_string)
    assert parameters[0] == 'none'
    assert parameters[1] == (1.0, 2.0, 3.0)

    behavior_string = 'reach none'
    parameters = behaviors.Reach.parse_parameters(behavior_string)
    assert parameters[0] == 'none'
    assert parameters[1] == (0.0, 0.0, 0.0)


def test_grasp():
    """ test Grasp behavior """
    robosuite_parameters = robosuite_interface.RobosuiteParameters()
    robosuite_parameters.type = 'lift'
    interface = robosuite_interface.RobosuiteInterface(robosuite_parameters, 0)
    behavior = behaviors.Grasp('grasp', ['cube', [0.0, 0.0, 0.0], 0.0], interface)

    behavior.initialise()
    behavior.update()
    interface.step()
    assert behavior.update() == pt.common.Status.SUCCESS

    behavior_string = 'grasp cube (1.0, 2.0, 3.0) 4.0'
    parameters = behaviors.Grasp.parse_parameters(behavior_string)
    assert parameters[0] == 'cube'
    assert parameters[1] == (1.0, 2.0, 3.0)
    assert parameters[2] == 4.0

    behavior_string = 'grasp cube'
    parameters = behaviors.Grasp.parse_parameters(behavior_string)
    assert parameters[0] == 'cube'
    assert parameters[1] == (0.0, 0.0, 0.0)
    assert parameters[2] == 0.0


def test_push():
    """ test Push behavior """
    robosuite_parameters = robosuite_interface.RobosuiteParameters()
    robosuite_parameters.type = 'lift'
    interface = robosuite_interface.RobosuiteInterface(robosuite_parameters, 1)

    # This move in beginning is just so that the cube settles in place
    behavior = behaviors.Reach('reach', ['none', [0.0, 0.0, 0.0]], interface)
    behavior.initialise()
    behavior.update()
    interface.step()

    behavior = behaviors.Push('push', ['cube', [-0.1, 0.0, 0.0], 0.0, [0.1, 0.1, 0.0]], interface)

    behavior.initialise()
    behavior.update()
    interface.step()
    assert behavior.update() == pt.common.Status.SUCCESS

    behavior_string = 'push cube (1.0, 2.0, 3.0) 4.0 (5.0, 6.0, 7.0)'
    parameters = behaviors.Push.parse_parameters(behavior_string)
    assert parameters[0] == 'cube'
    assert parameters[1] == (1.0, 2.0, 3.0)
    assert parameters[2] == 4.0
    assert parameters[3] == (5.0, 6.0, 7.0)

    behavior_string = 'push cube'
    parameters = behaviors.Push.parse_parameters(behavior_string)
    assert parameters[0] == 'cube'
    assert parameters[1] == (0.0, 0.0, 0.0)
    assert parameters[2] == 0.0
    assert parameters[3] == (0.0, 0.0, 0.0)


def test_push_towards():
    """ test PushTowards behavior """
    robosuite_parameters = robosuite_interface.RobosuiteParameters()
    robosuite_parameters.type = 'lift'
    interface = robosuite_interface.RobosuiteInterface(robosuite_parameters, 1)

    # This move in beginning is just so that the cube settles in place
    behavior = behaviors.Reach('reach', ['none', [0.0, 0.0, 0.0]], interface)
    behavior.initialise()
    behavior.update()
    interface.step()

    behavior = behaviors.PushTowards('push', ['cube', [0.1, 0.1, 0.0], 0.1], interface)

    behavior.initialise()
    behavior.update()
    interface.step()
    assert behavior.update() == pt.common.Status.SUCCESS

    behavior_string = 'push cube (1.0, 2.0, 3.0) 0.1'
    parameters = behaviors.PushTowards.parse_parameters(behavior_string)
    assert parameters[0] == 'cube'
    assert parameters[1] == (1.0, 2.0, 3.0)
    assert parameters[2] == 0.1

    behavior_string = 'push cube'
    parameters = behaviors.PushTowards.parse_parameters(behavior_string)
    assert parameters[0] == 'cube'
    assert parameters[1] == (0.0, 0.0, 0.0)
    assert parameters[2] == 0.0


def test_open():
    """ test Open behavior """
    robosuite_parameters = robosuite_interface.RobosuiteParameters()
    robosuite_parameters.type = 'lift'
    interface = robosuite_interface.RobosuiteInterface(robosuite_parameters, 0)

    behavior = behaviors.Open('open', [], interface)
    behavior.initialise()
    behavior.update()
    interface.step()

    assert behavior.update() == pt.common.Status.SUCCESS

    behavior_string = 'open'
    parameters = behaviors.Open.parse_parameters(behavior_string)
    assert not parameters
