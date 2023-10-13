"""Define the behavior list for robosuite tasks."""

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
import math
from copy import deepcopy
from behaviors import behavior_lists as bl
from behaviors.common_behaviors import NodeParameter, ParameterizedNode, ParameterTypes
from simulation.maple_benchmarks import robosuite_behaviors


def get_behavior_list(objects=None,
                      at_pos_threshold=None,
                      at_pos_free=False,
                      random_step=False,
                      angle_control=True,
                      gripper_control=True,
                      has_door=False,
                      large_object=False,
                      sequence_nodes=None,
                      root_nodes=None) -> bl.BehaviorLists:
    """Return the behavior list."""
    if objects is None:
        objects = []
    if angle_control:
        angle_parameter = NodeParameter([], round(-math.pi / 2, 2), round(math.pi / 2, 2), round(math.pi / 2, 2),
                                        data_type=ParameterTypes.FLOAT,
                                        random_step=random_step, standard_deviation=math.pi / 4)
    else:
        angle_parameter = NodeParameter([0.0])
    if large_object:
        grasp_parameter = NodeParameter([], (-0.1, -0.1, 0.0), (0.1, 0.1, 0.05), (0.05, 0.05, 0.05),
                                        data_type=ParameterTypes.POSITION, random_step=random_step, standard_deviation=0.035)
    else:
        grasp_parameter = NodeParameter([], (-0.05, -0.05, 0.0), (0.05, 0.05, 0.05), (0.05, 0.05, 0.05),
                                        data_type=ParameterTypes.POSITION, random_step=random_step, standard_deviation=0.015)

    condition_nodes = []
    if at_pos_free:
        condition_nodes.append(ParameterizedNode(
            name='at ',
            behavior=robosuite_behaviors.AtPosFree,
            parameters=[NodeParameter(objects, placement=0),
                        NodeParameter(['none'] + objects),
                        NodeParameter([], (-0.15, -0.15, -0.1), (0.15, 0.15, 0.1), (0.05, 0.05, 0.05),
                                      data_type=ParameterTypes.POSITION, random_step=random_step, standard_deviation=0.025),
                        NodeParameter([], at_pos_threshold, at_pos_threshold, at_pos_threshold,
                                      data_type=ParameterTypes.FLOAT, random_step=random_step, standard_deviation=0.0)],
            condition=True
            ))
    else:
        condition_nodes.append(ParameterizedNode(
            name='at ',
            behavior=robosuite_behaviors.AtPos,
            parameters=[NodeParameter(objects, placement=0),
                        NodeParameter(['none'] + objects),
                        NodeParameter([], (-0.15, -0.15, -0.1), (0.15, 0.15, 0.1), (0.05, 0.05, 0.05),
                                      data_type=ParameterTypes.POSITION, random_step=random_step, standard_deviation=0.025),
                        NodeParameter([], at_pos_threshold, at_pos_threshold, at_pos_threshold,
                                      data_type=ParameterTypes.FLOAT, random_step=random_step, standard_deviation=0.0)],
            condition=True
            ))

    if angle_control:
        condition_nodes.append(
            ParameterizedNode(
                name='aligned',
                behavior=robosuite_behaviors.Aligned,
                parameters=[NodeParameter(objects)],
                condition=True
                ))
    if has_door:
        condition_nodes.append(
            ParameterizedNode(
                name='handle angle >',
                behavior=robosuite_behaviors.HandleAngle,
                parameters=[NodeParameter([], 0.0, 1.0, 0.1, data_type=ParameterTypes.FLOAT,
                                          random_step=random_step, standard_deviation=0.05)],
                condition=True
                ))
        condition_nodes.append(
            ParameterizedNode(
                name='door angle >',
                behavior=robosuite_behaviors.DoorAngle,
                parameters=[NodeParameter([], 0.0, 1.0, 0.1, data_type=ParameterTypes.FLOAT,
                                          random_step=random_step, standard_deviation=0.05)],
                condition=True
                ))

    action_nodes = [
        ParameterizedNode(
            name='atomic',
            behavior=robosuite_behaviors.Atomic,
            parameters=[NodeParameter(objects),
                        NodeParameter([], (-0.15, -0.15, 0.0), (0.15, 0.15, 0.1), (0.05, 0.05, 0.05),
                                      data_type=ParameterTypes.POSITION, random_step=random_step, standard_deviation=0.025),
                        deepcopy(angle_parameter),
                        NodeParameter([], 0.0, 0.0, 0.0, data_type=ParameterTypes.FLOAT,
                                      random_step=random_step, standard_deviation=0.5)],
            condition=False
        ),
        ParameterizedNode(
            name='reach',
            behavior=robosuite_behaviors.Reach,
            parameters=[NodeParameter(['none'] + objects),
                        NodeParameter([], (-0.15, -0.15, 0.0), (0.15, 0.15, 0.15), (0.05, 0.05, 0.05),
                                      data_type=ParameterTypes.POSITION, random_step=random_step, standard_deviation=0.025)],
            condition=False
        )]
    if gripper_control:
        action_nodes.append(
            ParameterizedNode(
                name='grasp',
                behavior=robosuite_behaviors.Grasp,
                parameters=[NodeParameter(objects),
                            deepcopy(grasp_parameter),
                            deepcopy(angle_parameter)],
                condition=False
            ))
    action_nodes.append(
        ParameterizedNode(
            name='push',
            behavior=robosuite_behaviors.PushTowards,
            parameters=[NodeParameter(objects),
                        NodeParameter([], (-0.15, -0.15, 0.0), (0.15, 0.15, 0.0), (0.05, 0.05, 0.0),
                                      data_type=ParameterTypes.POSITION, random_step=random_step, standard_deviation=0.025),
                        NodeParameter([], 0.1, 0.2, 0.05, data_type=ParameterTypes.FLOAT,
                                      random_step=random_step, standard_deviation=0.025)],
            condition=False
        ))
    if gripper_control:
        action_nodes.append(
            ParameterizedNode(
                name='open',
                behavior=robosuite_behaviors.Open,
                condition=False
            ))

    if root_nodes is None:
        root_nodes = ['s(']

    behavior_list = bl.BehaviorLists(sequence_nodes=sequence_nodes, condition_nodes=condition_nodes, action_nodes=action_nodes,
                                     root_nodes=root_nodes)

    return behavior_list
