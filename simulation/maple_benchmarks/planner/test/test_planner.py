"""Unit test for planner."""

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
from behaviors.behavior_lists import BehaviorLists
from bt_learning.planner import planner
from simulation.maple_benchmarks.planner.world_interface import WorldInterface
from simulation.maple_benchmarks.planner import planner_behaviors


def test_lift():
    """ Test run of lift scenario """

    world_interface = WorldInterface('lift', ['cube'])
    goals = [planner_behaviors.AtPos('at ', ['cube', 'none', '(0.0, 0.0, 0.1)', True], world_interface)]
    string_bt, _ = planner.plan(world_interface, planner_behaviors, goals)

    assert string_bt == ['f(', 'cube at none (0.0, 0.0, 0.1)',
                               's(', 'grasp cube', 'reach none', ')', ')']


def test_door():
    """ Test run of door scenario """

    world_interface = WorldInterface('door', ['handle'])
    goals = [planner_behaviors.DoorOpen('door open', ['0.3'], world_interface)]
    string_bt, _ = planner.plan(world_interface, planner_behaviors, goals)

    assert string_bt == ['f(', 'door angle > 0.3',
                               's(', 'f(', 'handle angle >',
                                           's(', 'grasp handle', 'reach handle', ')', ')',
                                     'reach handle', ')',
                               ')']


def test_pnp():
    """ Test run of pnp scenario """

    world_interface = WorldInterface('pnp', ['can'])
    goals = [planner_behaviors.AtPos('at ', ['can', 'none', '(0.0, 0.15, 0.0)', False], world_interface)]
    string_bt, _ = planner.plan(world_interface, planner_behaviors, goals)

    assert string_bt == ['f(', 'can at none (0.0, 0.15, 0.0)',
                               's(', 'grasp can', 'reach none', 'open', ')',
                               ')']


def test_wipe():
    """ Test run of wipe scenario """

    world_interface = WorldInterface('wipe', ['centroid'])
    goals = [planner_behaviors.Wiped('wiped', ['centroid'], world_interface)]
    string_bt, _ = planner.plan(world_interface, planner_behaviors, goals, BehaviorLists(sequence_nodes=['s(', 'sm(']))

    assert string_bt == ['sm(', 'push centroid', 'push centroid', 'push centroid', 'push centroid', ')']


def test_stack():
    """ Test run of stack scenario """

    world_interface = WorldInterface('stack', ['red', 'green'])
    goals = [planner_behaviors.AtPos('at ', ['red', 'green', '(0.0, 0.0, 0.02)', False], world_interface)]
    string_bt, _ = planner.plan(world_interface, planner_behaviors, goals)

    assert string_bt == ['f(', 'red at green (0.0, 0.0, 0.02)',
                               's(', 'grasp red', 'reach green', 'open', ')',
                               ')']


def test_nut():
    """ Test run of nut scenario """

    world_interface = WorldInterface('nut', ['nut'])
    goals = [planner_behaviors.AtPos('at ', ['nut', 'none', '(0.125, -0.1, 0.0)', False], world_interface)]
    string_bt, _ = planner.plan(world_interface, planner_behaviors, goals)

    assert string_bt == ['f(', 'nut at none (0.125, -0.1, 0.0)',
                               's(', 'grasp nut', 'reach none', 'open', ')',
                               ')']


def test_cleanup():
    """ Test run of cleanup scenario """

    world_interface = WorldInterface('cleanup', ['spam', 'jello'], ['spam'])
    goals = [planner_behaviors.AtPos('at ', ['spam', 'none', '(-0.15, -0.15, 0.0)', False], world_interface),
             planner_behaviors.AtPos('at ', ['jello', 'none', '(-0.15, 0.15, 0.0)', False], world_interface)]
    string_bt, _ = planner.plan(world_interface, planner_behaviors, goals)

    assert string_bt == ['s(', 'f(', 'spam at none (-0.15, -0.15, 0.0)',
                                     's(', 'grasp spam', 'reach none', 'open', ')', ')',
                               'f(', 'jello at none (-0.15, 0.15, 0.0)', 'push jello', ')', ')']


def test_peg_ins():
    """ Test run of peg_ins scenario """

    world_interface = WorldInterface('peg_ins', ['peg'])
    goals = [planner_behaviors.Inserted('at ', ['peg', 'none', '(0.0, -0.15, 0.1)'], world_interface)]
    string_bt, _ = planner.plan(world_interface, planner_behaviors, goals)

    assert string_bt == ['f(', 'peg at none (0.0, -0.15, 0.1)',
                               's(', 'f(', 'aligned peg',
                                           's(', 'grasp peg', 'atomic peg', ')', ')',
                                     'atomic peg', ')', ')']


if __name__ == "__main__":
    test_pnp()
