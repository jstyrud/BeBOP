"""Simple world simulation for planning."""

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


class WorldInterface:
    """
    Class for handling the simple planning world
    """
    def __init__(self, scenario, movable_objects, graspable_objects=None):
        self.grasped_object = "none"
        self.object_positions = {}
        self.handle_down = {}
        self.wiped = {}
        self.aligned = {}
        self.inserted = {}
        for movable_object in movable_objects:
            self.object_positions[movable_object] = movable_object + "_INITIAL"
            self.handle_down[movable_object] = False
            self.wiped[movable_object] = False
            self.aligned[movable_object] = False
            self.inserted[movable_object] = False
        if graspable_objects is None:
            graspable_objects = movable_objects
        self.graspable_objects = graspable_objects
        self.door_open = False
        self.scenario = scenario

    def get_feedback(self):
        # pylint: disable=no-self-use
        """ Dummy to fit template """
        return True

    def send_references(self):
        # pylint: disable=no-self-use
        """ Dummy to fit template """
        return

    def is_graspable(self, target_object):
        """ True if object is graspable """
        return target_object in self.graspable_objects

    def get_grasped_object(self):
        """ Returns grasped object"""
        return self.grasped_object

    def grasp(self, target_object):
        """ Grasp an object"""
        self.grasped_object = target_object

    def object_at(self, target_object, target_position):
        """ Checks if object is at target_position"""
        if self.object_positions[target_object] == target_position:
            return True
        return False

    def is_door_open(self):
        """ True if door is open """
        return self.door_open

    def open_door(self):
        """ Opens door """
        self.door_open = True

    def is_handle_down(self, handle):
        """ True if handle is down """
        return self.handle_down[handle]

    def turn_handle(self, handle):
        """ Turn handle down """
        self.handle_down[handle] = True

    def is_wiped(self, target_object):
        """ True if object is wiped """
        return self.wiped[target_object]

    def wipe(self, target_object):
        """ Wipe target object """
        self.wiped[target_object] = True

    def is_aligned(self, target_object):
        """ True if object is aligned """
        return self.aligned[target_object]

    def align(self, target_object):
        """ Align target object """
        self.aligned[target_object] = True

    def is_inserted(self, target_object):
        """ True if object is inserted """
        return self.inserted[target_object]

    def insert(self, target_object):
        """ Insert target object """
        self.inserted[target_object] = True

    def reach(self, target_position):
        """ Reaches for target_position """
        if self.grasped_object != "none":
            self.move(self.grasped_object, target_position)

    def move(self, target_object, target_position):
        """ Moves target object to target position """
        self.object_positions[target_object] = target_position
