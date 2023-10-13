"""Hash table with linked list for entries with same hash."""

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

import hashlib
from typing import Any

from bt_learning.gp import logplot


# pylint: disable=too-few-public-methods
class Node:
    """Node data structure - essentially a LinkedList node."""

    def __init__(self, key: str, value: Any, value2=None):
        self.key = key
        self.value = [value]
        self.value2 = [value2]
        self.next = None

    def __eq__(self, other: 'Node') -> bool:
        if not isinstance(other, Node):
            return False
        equal = self.key == other.key and self.value == other.value
        if equal:
            if self.next is not None or other.next is not None:
                if self.next is None or other.next is None:
                    equal = False
                else:
                    equal = self.next == other.next
        return equal


class HashTable:
    """Main hash table / database class."""

    def __init__(self, size: int = 100000, log_name: str = 'tests/1', file_name: str = '/hash_log.txt'):
        """Initialize hash table to fixed size."""
        self.size = size
        self.buckets = [None]*self.size
        self.n_values = 0
        self.n_steps = 0
        self.log_name = log_name
        self.file_name = file_name

    def __eq__(self, other: 'HashTable') -> bool:
        if not isinstance(other, HashTable):
            return False

        equal = True
        for i in range(self.size):
            if self.buckets[i] != other.buckets[i]:
                equal = False
                break
        return equal

    def __hash(self, key: str) -> int:
        """
        Generate a hash for a given key.

        Args
        ----
            key: the string key

        Returns
        -------
            hash: hashcode generated from the key

        """
        new_hash = hashlib.md5()
        new_hash.update(key.encode('utf-8'))
        hashcode = new_hash.hexdigest()
        hashcode = int(hashcode, 16)
        return hashcode % self.size

    def insert(self, key: list, value: Any) -> None:
        """
        Insert a key - value pair to the hash table.

        Args:
        ----
            key: list
            value: anything

        """
        if isinstance(value, tuple):
            if len(value) >= 3:
                self.n_steps += value[2]

            if len(value) >= 2:
                value2 = value[1]
            value = value[0]
        else:
            value2 = value

        string_key = to_string(key)
        index = self.__hash(string_key)
        node = self.buckets[index]
        if node is None:
            self.buckets[index] = Node(string_key, value, value2)
        else:
            done = False
            while not done:
                if node.key == string_key:
                    node.value.append(value)
                    node.value2.append(value2)
                    done = True
                elif node.next is None:
                    node.next = Node(string_key, value, value2)
                    done = True
                else:
                    node = node.next
        self.n_values += 1

    def find(self, key: list) -> Any or None:
        """
        Find a data value based on key.

        Args
        ----
            key: key in the hash-table

        Returns
        -------
            value, value2: values stored under "key" or None if not found
        """
        string_key = to_string(key)
        index = self.__hash(string_key)
        node = self.buckets[index]
        while node is not None and node.key != string_key:
            node = node.next

        if node is None:
            return None

        return node.value, node.value2

    def load(self) -> None:
        """Load hash table information."""
        with open(
                logplot.get_log_folder(self.log_name) + self.file_name,
                'r',
                encoding='utf-8'
             ) as f:
            lines = f.read().splitlines()

            for i in range(0, len(lines)):
                individual = lines[i]
                individual = individual[5:].split(', value: ')
                key = individual[0]
                individual = individual[1].split(', value2: ')
                individual2 = individual[1].split(', count: ')
                values = individual[0][1:-1].split(', ')  # Remove brackets and split multiples
                values2 = individual2[0][1:-1].split(', ')  # Remove brackets and split multiples
                for i, value in enumerate(values):
                    self.insert(key, (float(value), float(values2[i])))

    def write_table(self):
        """Write table contents to a file."""
        with open(
                logplot.get_log_folder(self.log_name) + self.file_name,
                'w',
                encoding='utf-8'
             ) as f:
            for node in filter(lambda x: x is not None, self.buckets):
                while node is not None:
                    f.writelines(
                        'key: ' + str(node.key) +
                        ', value: ' + str(node.value) +
                        ', value2: ' + str(node.value2) +
                        ', count: ' + str(len(node.value)) + '\n'
                    )
                    node = node.next
        f.close()

    def set_n_steps(self, n_steps):
        """ Sets n_steps from outside """
        self.n_steps = n_steps


def to_string(key: Any) -> str:
    """Convert a key to string."""
    if isinstance(key, str):
        return key
    try:
        string = ', '.join(str(e) for e in key)
    except TypeError:
        string = str(key)
    return string
