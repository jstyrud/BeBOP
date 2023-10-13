"""Behaviors used for planning."""

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

from behaviors.common_behaviors import Behavior
from behaviors import common_behaviors
import py_trees as pt


def get_node(node_descriptor, world_interface, verbose=False):
    """Return a py trees behavior or composite given the string."""
    return common_behaviors.get_node(node_descriptor, world_interface, verbose)


class PlannedBehavior():
    """
    Class template for planned behaviors
    """
    def __init__(self, preconditions, postconditions):
        self.preconditions = preconditions
        self.postconditions = postconditions

    def get_preconditions(self):
        """ Returns list of preconditions """
        return self.preconditions

    def get_postconditions(self):
        """ Returns list of postconditions """
        return self.postconditions

    def has_postcondition_check(self):
        """
        Most behaviors check postconditions first and
        only execute if they are not already fulfilled
        """
        return True


class AtPos(pt.behaviour.Behaviour, PlannedBehavior):
    """
    Check if object is at position
    """
    def __init__(self, name, parameters, world_interface):
        self.world_interface = world_interface
        if len(parameters) > 0:
            self.target_object = parameters[0]
            name = self.target_object + " " + name
        if len(parameters) > 1:
            self.relative_object = parameters[1]
            name += self.relative_object
        if len(parameters) > 2:
            self.offset = parameters[2]
            if self.offset != 'unknown':
                name += " " + self.offset
        if len(parameters) > 3:
            self.grasped_is_ok = parameters[3]
        pt.behaviour.Behaviour.__init__(self, name)
        PlannedBehavior.__init__(self, [], [])

    def __eq__(self, other) -> bool:
        if not isinstance(other, AtPos):
            # don't attempt to compare against unrelated types
            return False

        return self.name == other.name and self.target_object == other.target_object and \
            self.relative_object == other.relative_object and self.offset == other.offset and \
            self.grasped_is_ok == other.grasped_is_ok

    def get_condition_parameters(self):
        """ Returns parameters of the condition """
        return [self.target_object, self.relative_object, self.offset, self.grasped_is_ok]

    def update(self):
        if self.world_interface.object_at(self.target_object, self.relative_object + "+" + self.offset) and \
           (self.grasped_is_ok or self.world_interface.get_grasped_object() == "none"):
            return pt.common.Status.SUCCESS
        return pt.common.Status.FAILURE


class Grasped(pt.behaviour.Behaviour, PlannedBehavior):
    """
    Check if object is grasped
    """
    def __init__(self, name, parameters, world_interface):
        pt.behaviour.Behaviour.__init__(self, name)
        PlannedBehavior.__init__(self, [], [])
        self.world_interface = world_interface
        self.target_object = parameters[0]

    def __eq__(self, other) -> bool:
        if not isinstance(other, Grasped):
            # don't attempt to compare against unrelated types
            return False

        return self.name == other.name and self.target_object == other.target_object

    def get_condition_parameters(self):
        """ Returns parameters of the condition """
        return [self.target_object]

    def update(self):
        if self.world_interface.get_grasped_object() == self.target_object:
            return pt.common.Status.SUCCESS
        return pt.common.Status.FAILURE


class Grasp(Behavior, PlannedBehavior):
    """
    Grasp an object
    """
    def __init__(self, name, parameters, world_interface, verbose=False):
        if not name:
            name = "grasp"
        postconditions = []
        if len(parameters) > 0:
            self.target_object = parameters[0]
            name += " " + self.target_object
            if self.target_object != 'none':
                postconditions = [Grasped('grasped', [self.target_object], world_interface)]
        Behavior.__init__(self, name, world_interface, verbose, max_ticks=1)
        PlannedBehavior.__init__(self, [], postconditions)

    def cost(self) -> int:
        """Define the cost of the action."""
        return 1

    def check_for_success(self):
        """Check if object is grasped."""
        if self.world_interface.get_grasped_object() == self.target_object:
            self.success()

    def update(self):
        """Executes behavior """
        self.check_for_success()
        Behavior.update(self)

        if self.state is pt.common.Status.RUNNING:
            self.world_interface.grasp(self.target_object)
        return self.state


class Reach(Behavior, PlannedBehavior):
    """
    Reach position
    """
    def __init__(self, name, parameters, world_interface, verbose=False):
        if not name:
            name = "reach"
        preconditions = []
        postconditions = []
        if len(parameters) > 0:
            self.target_object = parameters[0]
            preconditions = [Grasped('grasped', [self.target_object], world_interface)]
        if len(parameters) > 1:
            self.relative_object = parameters[1]
            name += " " + self.relative_object
        else:
            self.relative_object = ""
        if len(parameters) > 2:
            self.target_position = parameters[2]
            if world_interface.is_graspable(self.target_object):
                postconditions = [AtPos('at ', [self.target_object,
                                                self.relative_object,
                                                self.target_position,
                                                True],
                                        world_interface)]
        else:
            self.target_position = ""
        Behavior.__init__(self, name, world_interface, verbose, max_ticks=1)
        PlannedBehavior.__init__(self, preconditions, postconditions)

    def cost(self) -> int:
        """Define the cost of the action."""
        return 1

    def check_for_success(self):
        """Check if object is at target position."""
        if self.world_interface.object_at(self.target_object, self.target_position):
            self.success()

    def update(self):
        """Executes behavior """
        self.check_for_success()
        Behavior.update(self)

        if self.state is pt.common.Status.RUNNING:
            self.world_interface.reach(self.relative_object + "+" + self.target_position)
        return self.state


class Place(Behavior, PlannedBehavior):
    """
    Place object on position position
    """
    def __init__(self, name, parameters, world_interface, verbose=False):
        if not name:
            name = "place"
        preconditions = []
        postconditions = []
        if len(parameters) > 0:
            self.target_object = parameters[0]
            preconditions = [Grasped('grasped', [self.target_object], world_interface)]
        if len(parameters) > 1:
            self.relative_object = parameters[1]
        else:
            self.relative_object = ""
        if len(parameters) > 2:
            self.target_position = parameters[2]
            if world_interface.is_graspable(self.target_object):
                postconditions = [AtPos('at ', [self.target_object,
                                                self.relative_object,
                                                self.target_position,
                                                True],
                                        world_interface),
                                  AtPos('at ', [self.target_object,
                                                self.relative_object,
                                                self.target_position,
                                                False],
                                        world_interface)]
        else:
            self.target_position = ""
        Behavior.__init__(self, name, world_interface, verbose, max_ticks=1)
        PlannedBehavior.__init__(self, preconditions, postconditions)

    def get_composite_subtree(self):
        """ Get subtree to replace this node in final tree """
        subtree = pt.composites.Sequence('s(', memory=False)

        subtree.add_child(Reach('reach', [self.target_object, self.relative_object, self.target_position], self.world_interface))
        subtree.add_child(Open('open', [], self.world_interface))
        return subtree

    def cost(self) -> int:
        """Define the cost of the action."""
        return 2

    def check_for_success(self):
        """Check if object is at target position."""
        if self.world_interface.object_at(self.target_object, self.target_position):
            self.success()

    def update(self):
        """Executes behavior """
        self.check_for_success()
        Behavior.update(self)

        if self.state is pt.common.Status.RUNNING:
            self.world_interface.reach(self.relative_object + "+" + self.target_position)
            self.world_interface.grasp('none')
        return self.state


class Push(Behavior, PlannedBehavior):
    """
    Push object
    """
    def __init__(self, name, parameters, world_interface, verbose=False):
        if not name:
            name = "push"

        preconditions = []
        postconditions = []
        if len(parameters) > 0:
            self.target_object = parameters[0]
            name += " " + self.target_object
        if len(parameters) > 2:
            self.offset = parameters[2]
            postconditions = [AtPos('at ', [self.target_object,
                                            'none',
                                            self.offset,
                                            False],
                                    world_interface)]
        else:
            self.offset = ""
        self.success_on_next = False
        Behavior.__init__(self, name, world_interface, verbose, max_ticks=1)
        PlannedBehavior.__init__(self, preconditions, postconditions)

    def has_postcondition_check(self):
        """ Push always execution, no internal postcondition check """
        return False

    def cost(self) -> int:
        """Define the cost of the action."""
        return 3

    def initialise(self) -> None:
        super().initialise()
        self.success_on_next = False

    def check_for_success(self):
        """Check push always runs once and then reports success."""
        if self.success_on_next:
            self.success()
        self.success_on_next = True

    def update(self):
        """Executes behavior """
        self.check_for_success()
        Behavior.update(self)

        if self.state is pt.common.Status.RUNNING:
            self.world_interface.move(self.target_object, 'none+' + self.offset)
        return self.state


class Open(Behavior, PlannedBehavior):
    """
    Open gripper
    """
    def __init__(self, name, _parameters, world_interface, verbose=False):
        if not name:
            name = "open"

        Behavior.__init__(self, name, world_interface, verbose, max_ticks=1)
        PlannedBehavior.__init__(self, [], [Grasped('grasped', ['none'], world_interface)])

    def cost(self) -> int:
        """Define the cost of the action."""
        return 1

    def check_for_success(self):
        """Check if object is at target position."""
        if self.world_interface.get_grasped_object() == 'none':
            self.success()

    def update(self):
        """Executes behavior """
        self.check_for_success()
        Behavior.update(self)

        if self.state is pt.common.Status.RUNNING:
            self.world_interface.grasp('none')
        return self.state


class DoorOpen(pt.behaviour.Behaviour, PlannedBehavior):
    """
    Check if door is open
    """
    def __init__(self, name, parameters, world_interface):
        name = "door angle >"  # Non-planner behavior to replace with

        if len(parameters) > 0:
            self.angle = parameters[0]
            name += " " + self.angle
        else:
            self.angle = 0.0

        pt.behaviour.Behaviour.__init__(self, name)
        PlannedBehavior.__init__(self, [], [])
        self.world_interface = world_interface

    def __eq__(self, other) -> bool:
        if not isinstance(other, DoorOpen):
            # don't attempt to compare against unrelated types
            return False

        return self.name == other.name and self.angle == other.angle

    def get_condition_parameters(self):
        """ Returns parameters of the condition """
        return [self.angle]

    def update(self):
        if self.world_interface.is_door_open():
            return pt.common.Status.SUCCESS
        return pt.common.Status.FAILURE


class HandleDown(pt.behaviour.Behaviour, PlannedBehavior):
    """
    Check if handle is down
    """
    def __init__(self, name, parameters, world_interface):
        pt.behaviour.Behaviour.__init__(self, name)
        PlannedBehavior.__init__(self, [], [])
        self.world_interface = world_interface
        self.handle = parameters[0]

    def __eq__(self, other) -> bool:
        if not isinstance(other, HandleDown):
            # don't attempt to compare against unrelated types
            return False

        return self.name == other.name and self.handle == other.handle

    def get_condition_parameters(self):
        """ Returns parameters of the condition """
        return [self.handle]

    def update(self):
        if self.world_interface.is_handle_down(self.handle):
            return pt.common.Status.SUCCESS
        return pt.common.Status.FAILURE


class OpenDoor(Behavior, PlannedBehavior):
    """
    Open door
    """
    def __init__(self, name, parameters, world_interface, verbose=False):
        name = "reach handle"  # Non-planner behavior to replace with

        Behavior.__init__(self, name, world_interface, verbose, max_ticks=1)
        PlannedBehavior.__init__(self, [HandleDown('handle angle >', ['handle'], world_interface)],
                                 [DoorOpen('door angle', parameters, world_interface)])

    def cost(self) -> int:
        """Define the cost of the action."""
        return 1

    def check_for_success(self):
        """Check if object is at target position."""
        if self.world_interface.is_door_open():
            self.success()

    def update(self):
        """Executes behavior """
        self.check_for_success()
        Behavior.update(self)

        if self.state is pt.common.Status.RUNNING:
            self.world_interface.open_door()
        return self.state


class TurnHandle(Behavior, PlannedBehavior):
    """
    Open door
    """
    def __init__(self, name, parameters, world_interface, verbose=False):
        name = "reach handle"  # Non-planner behavior to replace with
        if len(parameters) > 0:
            self.handle = parameters[0]

        Behavior.__init__(self, name, world_interface, verbose, max_ticks=1)
        PlannedBehavior.__init__(self, [Grasped('grasped', ['handle'], world_interface)],
                                 [HandleDown('handle angle >', ['handle'], world_interface)])

    def cost(self) -> int:
        """Define the cost of the action."""
        return 1

    def check_for_success(self):
        """Check if object is at target position."""
        if self.world_interface.is_handle_down(self.handle):
            self.success()

    def update(self):
        """Executes behavior """
        self.check_for_success()
        Behavior.update(self)

        if self.state is pt.common.Status.RUNNING:
            self.world_interface.turn_handle(self.handle)
        return self.state


class Wiped(pt.behaviour.Behaviour, PlannedBehavior):
    """
    Check if target object is wipe
    """
    def __init__(self, name, parameters, world_interface):
        pt.behaviour.Behaviour.__init__(self, name)
        PlannedBehavior.__init__(self, [], [])
        self.world_interface = world_interface
        self.target_object = parameters[0]

    def __eq__(self, other) -> bool:
        if not isinstance(other, Wiped):
            # don't attempt to compare against unrelated types
            return False

        return self.name == other.name and self.target_object == other.target_object

    def get_condition_parameters(self):
        """ Returns parameters of the condition """
        return [self.target_object]

    def update(self):
        if self.world_interface.is_wiped(self.target_object):
            return pt.common.Status.SUCCESS
        return pt.common.Status.FAILURE


class Wipe(Behavior, PlannedBehavior):
    """
    Wipe target object
    """
    def __init__(self, name, parameters, world_interface, verbose=False):
        name = "wipe"  # Non-planner behavior to replace with
        postconditions = []
        if len(parameters) > 0:
            self.target_object = parameters[0]
            postconditions = [Wiped('wiped', [self.target_object], world_interface)]
        Behavior.__init__(self, name, world_interface, verbose, max_ticks=1)
        PlannedBehavior.__init__(self, [], postconditions)

    def get_composite_subtree(self):
        """ Get subtree to replace this node in final tree """
        subtree = pt.composites.Sequence('sm(', memory=True)
        for _ in range(4):
            subtree.add_child(Push('push', [self.target_object], self.world_interface))
        return subtree

    def cost(self) -> int:
        """Define the cost of the action."""
        return 1

    def check_for_success(self):
        """Check if object is at target position."""
        if self.world_interface.is_wiped(self.target_object):
            self.success()

    def update(self):
        """Executes behavior """
        self.check_for_success()
        Behavior.update(self)

        if self.state is pt.common.Status.RUNNING:
            self.world_interface.wipe(self.target_object)
        return self.state


class Aligned(pt.behaviour.Behaviour, PlannedBehavior):
    """
    Check if target object is aligned
    """
    def __init__(self, name, parameters, world_interface):
        name = 'aligned'  # Non-planner behavior to replace with
        self.world_interface = world_interface
        self.target_object = parameters[0]
        name = name + " " + self.target_object
        pt.behaviour.Behaviour.__init__(self, name)
        PlannedBehavior.__init__(self, [], [])

    def __eq__(self, other) -> bool:
        if not isinstance(other, Aligned):
            # don't attempt to compare against unrelated types
            return False

        return self.name == other.name and self.target_object == other.target_object

    def get_condition_parameters(self):
        """ Returns parameters of the condition """
        return [self.target_object]

    def update(self):
        if self.world_interface.is_aligned(self.target_object):
            return pt.common.Status.SUCCESS
        return pt.common.Status.FAILURE


class Align(Behavior, PlannedBehavior):
    """
    Align target object
    """
    def __init__(self, name, parameters, world_interface, verbose=False):
        name = "atomic"  # Non-planner behavior to replace with
        preconditions = []
        postconditions = []
        if len(parameters) > 0:
            self.target_object = parameters[0]
            name += " " + self.target_object
            preconditions = [Grasped('grasped', [self.target_object], world_interface)]
            postconditions = [Aligned('aligned', [self.target_object], world_interface)]
        Behavior.__init__(self, name, world_interface, verbose, max_ticks=1)
        PlannedBehavior.__init__(self, preconditions, postconditions)

    def cost(self) -> int:
        """Define the cost of the action."""
        return 1

    def check_for_success(self):
        """Check if object is at target position."""
        if self.world_interface.is_aligned(self.target_object):
            self.success()

    def update(self):
        """Executes behavior """
        self.check_for_success()
        Behavior.update(self)

        if self.state is pt.common.Status.RUNNING:
            self.world_interface.align(self.target_object)
        return self.state


class Inserted(pt.behaviour.Behaviour, PlannedBehavior):
    """
    Check if target object is inserted
    """
    def __init__(self, name, parameters, world_interface):
        self.world_interface = world_interface
        self.target_object = parameters[0]
        name = self.target_object + " " + name
        if len(parameters) > 1:
            self.relative_object = parameters[1]
            name += self.relative_object
        if len(parameters) > 2:
            self.offset = parameters[2]
            if self.offset != 'unknown':
                name += " " + self.offset
        else:
            self.offset = ""
        pt.behaviour.Behaviour.__init__(self, name)
        PlannedBehavior.__init__(self, [], [])

    def __eq__(self, other) -> bool:
        if not isinstance(other, Inserted):
            # don't attempt to compare against unrelated types
            return False

        return self.name == other.name and self.target_object == other.target_object and \
            self.relative_object == other.relative_object and self.offset == other.offset

    def get_condition_parameters(self):
        """ Returns parameters of the condition """
        return [self.target_object, self.relative_object, self.offset]

    def update(self):
        if self.world_interface.is_inserted(self.target_object):
            return pt.common.Status.SUCCESS
        return pt.common.Status.FAILURE


class Insert(Behavior, PlannedBehavior):
    """
    Insert target object
    """
    def __init__(self, name, parameters, world_interface, verbose=False):
        name = "atomic"  # Non-planner behavior to replace with
        preconditions = []
        postconditions = []
        if len(parameters) > 0:
            self.target_object = parameters[0]
            name += " " + self.target_object
        if len(parameters) > 1:
            self.relative_object = parameters[1]
        if len(parameters) > 2:
            self.offset = parameters[2]
            preconditions = [Aligned('atomic', [self.target_object], world_interface)]
            postconditions = [Inserted('at ', [self.target_object, self.relative_object, self.offset], world_interface)]
        Behavior.__init__(self, name, world_interface, verbose, max_ticks=1)
        PlannedBehavior.__init__(self, preconditions, postconditions)

    def cost(self) -> int:
        """Define the cost of the action."""
        return 1

    def check_for_success(self):
        """Check if object is at target position."""
        if self.world_interface.is_inserted(self.target_object):
            self.success()

    def update(self):
        """Executes behavior """
        self.check_for_success()
        Behavior.update(self)

        if self.state is pt.common.Status.RUNNING:
            self.world_interface.insert(self.target_object)
        return self.state


def get_action_nodes():
    """ Returns a list of all action nodes available for planning """
    return [Grasp, Reach, Place, Push, Open, OpenDoor, TurnHandle, Wipe, Align, Insert]
