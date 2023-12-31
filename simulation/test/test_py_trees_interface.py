"""Unit test for py_trees_interface.py module."""

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
from behaviors import behavior_list_test_settings
from behaviors.behavior_lists import BehaviorLists
import py_trees as pt

import simulation.py_trees_interface as interface
from simulation import behaviors_toggle_read as behaviors


parameters = interface.PyTreeParameters()
parameters.behavior_lists = BehaviorLists(
    condition_nodes=behavior_list_test_settings.get_condition_nodes(),
    action_nodes=behavior_list_test_settings.get_action_nodes()
)
parameters.behaviors = behaviors


def test_pytree():
    """Tests the PyTree class initialization."""
    bt = ['f(', 'a', 'a', ')']
    py_tree = interface.PyTree(bt, parameters=parameters)
    assert not bt
    assert len(py_tree.root.children) == 2
    print(pt.display.ascii_tree(py_tree.root))

    bt = ['f(', 'f(', 'a', 'a', ')', ')']
    py_tree = interface.PyTree(bt, parameters=parameters)
    assert not bt
    assert len(py_tree.root.children) == 1
    print(pt.display.ascii_tree(py_tree.root))

    bt = ['f(', 'f(', 'a', 'a', ')', 's(', 'a', 'a', ')', ')']
    py_tree = interface.PyTree(bt, parameters=parameters)
    assert not bt
    assert len(py_tree.root.children) == 2
    print(pt.display.ascii_tree(py_tree.root))

    bt = ['f(', 'f(', 'a', 'a', ')', 'f(', 's(', 'a', ')', ')', ')']
    py_tree = interface.PyTree(bt, parameters=parameters)
    assert not bt
    assert len(py_tree.root.children) == 2
    print(pt.display.ascii_tree(py_tree.root))

    bt = ['f(', 'f(', 'a', 'a', ')']
    py_tree = interface.PyTree(bt, parameters=parameters)
    assert not bt
    assert len(py_tree.root.children) == 1
    print(pt.display.ascii_tree(py_tree.root))

    bt = ['f(', 'f(', 'a', ')', ')', 'a', ')']
    py_tree = interface.PyTree(bt,  parameters=parameters)
    assert bt
    print(pt.display.ascii_tree(py_tree.root))


def test_get_bt_from_root():
    """Specific test for get_string_from_root function."""
    bt = ['f(', 'f(', 'a', 'a', ')', 'f(', 's(', 'a', ')', ')', ')']
    py_tree = interface.PyTree(bt[:], parameters=parameters)
    assert py_tree.get_bt_from_root() == bt

    bt = ['f(', 'f(', 'a', ')', 'a', ')']
    py_tree = interface.PyTree(bt[:], parameters=parameters)
    assert py_tree.get_bt_from_root() == bt

    py_tree_copy = interface.PyTree([], root=py_tree.root, parameters=parameters)
    assert py_tree_copy.get_bt_from_root() == bt


def test_save_fig():
    """ Tests save fig function """
    bt = ['s(', 'node1', 'node2', 'f(', 'node3', ')', ')']

    pytree = interface.PyTree(bt, parameters, None)
    pytree.save_fig("", name="test_tree")
