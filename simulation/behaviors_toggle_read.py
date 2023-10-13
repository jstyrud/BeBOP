"""Implement various py trees behaviors for testing."""

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
from behaviors import common_behaviors


def get_node(string, world_interface, verbose=False):
    """Return a py trees behavior or composite given the string."""
    has_children = False
    behavior = BEHAVIORS_DICT.get(string)
    if behavior is not None:
        node = behavior(string, world_interface)
    else:
        node, has_children = common_behaviors.get_node(string, world_interface, verbose)
    return node, has_children


class ANode(pt.behaviour.Behaviour):
    """Simple node for testing."""

    def __init__(self, act, _):
        self.act = act

        super().__init__(str(self.act))

    def update(self):
        print(self.act)
        return pt.common.Status.SUCCESS


class Toggle1(pt.behaviour.Behaviour):
    """Simple node for testing."""

    def __init__(self, name, world_interface):
        self.world_interface = world_interface
        super().__init__(str(name))

    def update(self):
        self.world_interface.toggle1()
        return pt.common.Status.SUCCESS


class Toggle2(pt.behaviour.Behaviour):
    """Simple node for testing."""

    def __init__(self, name, world_interface):
        self.world_interface = world_interface
        super().__init__(str(name))

    def update(self):
        self.world_interface.toggle2()
        return pt.common.Status.SUCCESS


class Toggle3(pt.behaviour.Behaviour):
    """Simple node for testing."""

    def __init__(self, name, world_interface):
        self.world_interface = world_interface
        super().__init__(str(name))

    def update(self):
        self.world_interface.toggle3()
        return pt.common.Status.SUCCESS


class Toggle4(pt.behaviour.Behaviour):
    """Simple node for testing."""

    def __init__(self, name, world_interface):
        self.world_interface = world_interface
        super().__init__(str(name))

    def update(self):
        self.world_interface.toggle4()
        return pt.common.Status.SUCCESS


class Read1(pt.behaviour.Behaviour):
    """Simple node for testing."""

    def __init__(self, name, world_interface):
        self.world_interface = world_interface
        super().__init__(str(name))

    def update(self):
        if self.world_interface.read1():
            return pt.common.Status.SUCCESS
        return pt.common.Status.FAILURE


class Read2(pt.behaviour.Behaviour):
    """Simple node for testing."""

    def __init__(self, name, world_interface):
        self.world_interface = world_interface
        super().__init__(str(name))

    def update(self):
        if self.world_interface.read2():
            return pt.common.Status.SUCCESS
        return pt.common.Status.FAILURE


class Read3(pt.behaviour.Behaviour):
    """Simple node for testing."""

    def __init__(self, name, world_interface):
        self.world_interface = world_interface
        super().__init__(str(name))

    def update(self):
        if self.world_interface.read3():
            return pt.common.Status.SUCCESS
        return pt.common.Status.FAILURE


class Read4(pt.behaviour.Behaviour):
    """Simple node for testing."""

    def __init__(self, name, world_interface):
        self.world_interface = world_interface
        super().__init__(str(name))

    def update(self):
        if self.world_interface.read4():
            return pt.common.Status.SUCCESS
        return pt.common.Status.FAILURE


BEHAVIORS_DICT = {
        'a': ANode,
        'a0': ANode,
        'a1': ANode,
        't1': Toggle1,
        't2': Toggle2,
        't3': Toggle3,
        't4': Toggle4,
        'r1': Read1,
        'r2': Read2,
        'r3': Read3,
        'r4': Read4
    }
