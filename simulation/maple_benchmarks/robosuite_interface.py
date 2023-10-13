"""Interface to Robosuite simulations"""
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
import os.path as osp
import copy
from enum import IntEnum
import random
import math
from typing import Any
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
import imageio

import maple.util.hyperparameter as hyp
import maple.launchers.visualization as visualization
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper


IN_POS_DIST = 0.03  # Distance to consider in position for reach targets
AT_ROT_DIFF = 0.2  # Acceptable angle difference to be considered correct rotation

base_variant = dict(
    layer_size=256,
    replay_buffer_size=int(1E6),
    rollout_fn_kwargs=dict(
        terminals_all_false=True,
    ),
    algorithm_kwargs=dict(
        num_epochs=1000,
        num_expl_steps_per_train_loop=3000,
        num_eval_steps_per_epoch=3000,
        num_trains_per_train_loop=1000,
        min_num_steps_before_training=30000,
        max_path_length=150,
        batch_size=1024,
        eval_epoch_freq=10,
    ),
    env_variant=dict(
        robot_keys=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel'],
        obj_keys=['object-state'],
        controller_type='OSC_POSITION_YAW',
        controller_config_update=dict(
            position_limits=[
                [-0.30, -0.30, 0.75],
                [0.15, 0.30, 1.15]
            ],
        ),
        env_kwargs=dict(
            ignore_done=True,
            reward_shaping=True,
            hard_reset=False,
            control_freq=10,
            camera_heights=512,
            camera_widths=512,
            table_offset=[-0.075, 0, 0.8],
            reward_scale=5.0,

            skill_config=dict(
                skills=['atomic', 'open', 'reach', 'grasp', 'push'],
                aff_penalty_fac=15.0,

                base_config=dict(
                    global_xyz_bounds=[
                        [-0.30, -0.30, 0.80],
                        [0.15, 0.30, 0.95]
                    ],
                    lift_height=0.95,
                    binary_gripper=True,

                    aff_threshold=0.06,
                    aff_type='dense',
                    aff_tanh_scaling=10.0,
                ),
                atomic_config=dict(
                    use_ori_params=True,
                ),
                reach_config=dict(
                    use_gripper_params=False,
                    local_xyz_scale=[0.0, 0.0, 0.06],
                    use_ori_params=False,
                    max_ac_calls=15,
                ),
                grasp_config=dict(
                    global_xyz_bounds=[
                        [-0.30, -0.30, 0.80],
                        [0.15, 0.30, 0.85]
                    ],
                    aff_threshold=0.03,

                    local_xyz_scale=[0.0, 0.0, 0.0],
                    use_ori_params=True,
                    max_ac_calls=20,
                    num_reach_steps=2,
                    num_grasp_steps=3,
                ),
                push_config=dict(
                    global_xyz_bounds=[
                        [-0.30, -0.30, 0.80],
                        [0.15, 0.30, 0.85]
                    ],
                    delta_xyz_scale=[0.25, 0.25, 0.05],

                    max_ac_calls=20,
                    use_ori_params=True,

                    aff_threshold=[0.12, 0.12, 0.04],
                ),
            ),
        ),
    ),
    save_video=True,
    save_video_period=100,
    dump_video_kwargs=dict(
        rows=1,
        columns=6,
        pad_length=5,
        pad_color=0,
    ),
)

env_params = dict(
    lift={
        'env_variant.env_type': ['Lift'],
    },
    door={
        'env_variant.env_type': ['Door'],
        'env_variant.controller_config_update.position_limits': [[[-0.25, -0.25, 0.90], [0.05, 0.05, 1.20]]],
        'env_variant.env_kwargs.skill_config.base_config.global_xyz_bounds': [
            [[-0.25, -0.25, 0.90], [0.05, 0.05, 1.20]]],
        'env_variant.env_kwargs.skill_config.grasp_config.global_xyz_bounds': [
            [[-0.25, -0.25, 0.90], [0.05, 0.05, 1.20]]],
        'env_variant.env_kwargs.skill_config.push_config.global_xyz_bounds': [
            [[-0.25, -0.25, 0.90], [0.05, 0.05, 1.20]]],
        'env_variant.env_kwargs.skill_config.base_config.lift_height': [1.15],
        'env_variant.env_kwargs.skill_config.grasp_config.aff_threshold': [0.06],
        'env_variant.env_kwargs.skill_config.skills': [['atomic', 'grasp', 'reach_osc', 'push', 'open']],
    },
    pnp={
        'env_variant.env_type': ['PickPlaceCan'],
        'env_variant.env_kwargs.bin1_pos': [[0.0, -0.25, 0.8]],
        'env_variant.env_kwargs.bin2_pos': [[0.0, 0.28, 0.8]],
        'env_variant.controller_config_update.position_limits': [[[-0.15, -0.50, 0.75], [0.15, 0.50, 1.15]]],
        'env_variant.env_kwargs.skill_config.base_config.global_xyz_bounds': [
            [[-0.15, -0.50, 0.82], [0.15, 0.50, 1.02]]],
        'env_variant.env_kwargs.skill_config.grasp_config.global_xyz_bounds': [
            [[-0.15, -0.50, 0.82], [0.15, 0.50, 0.88]]],
        'env_variant.env_kwargs.skill_config.push_config.global_xyz_bounds': [
            [[-0.15, -0.50, 0.82], [0.15, 0.50, 0.88]]],
        'env_variant.env_kwargs.skill_config.base_config.lift_height': [1.0],
        'env_variant.env_kwargs.skill_config.reach_config.aff_threshold': [[0.15, 0.25, 0.06]],
    },
    wipe={
        'env_variant.env_type': ['Wipe'],
        'env_variant.obj_keys': [['robot0_contact-obs', 'object-state']],
        'algorithm_kwargs.max_path_length': [300],
        'env_variant.controller_type': ['OSC_POSITION'],
        'env_variant.controller_config_update.position_limits': [[[-0.10, -0.30, 0.75], [0.20, 0.30, 1.00]]],
        'env_variant.env_kwargs.table_offset': [[0.05, 0, 0.8]],
        'env_variant.env_kwargs.skill_config.base_config.global_xyz_bounds': [
            [[-0.10, -0.30, 0.75], [0.20, 0.30, 1.00]]],
        'env_variant.env_kwargs.skill_config.grasp_config.global_xyz_bounds': [
            [[-0.10, -0.30, 0.80], [0.20, 0.30, 0.85]]],
        'env_variant.env_kwargs.skill_config.push_config.global_xyz_bounds': [
            [[-0.10, -0.30, 0.80], [0.20, 0.30, 0.85]]],
        'env_variant.env_kwargs.skill_config.base_config.aff_threshold': [[0.15, 0.25, 0.03]],
        'env_variant.env_kwargs.skill_config.reach_config.aff_threshold': [[0.15, 0.25, 0.03]],
        'env_variant.env_kwargs.skill_config.grasp_config.aff_threshold': [[0.15, 0.25, 0.03]],
        'env_variant.env_kwargs.skill_config.push_config.aff_threshold': [[0.15, 0.25, 0.03]],
        'env_variant.env_kwargs.skill_config.skills': [['atomic', 'reach', 'push']],
    },
    stack={
        'env_variant.env_type': ['Stack'],
    },
    nut={
        'env_variant.env_type': ['NutAssemblyRound'],
        'env_variant.env_kwargs.skill_config.grasp_config.aff_threshold': [0.06],
    },
    cleanup={
        'env_variant.env_type': ['Cleanup'],
    },
    peg_ins={
        'env_variant.env_type': ['PegInHole'],
        'env_variant.controller_config_update.position_limits': [[[-0.30, -0.30, 0.75], [0.15, 0.30, 1.00]]],
        'env_variant.env_kwargs.skill_config.reach_config.aff_threshold': [0.06],
        'pamdp_variant.one_hot_factor': [0.375],
    },
)


class ActionTypes(IntEnum):
    """Define the action types."""
    ATOMIC = 0  # Atomic action
    REACH = 1  # Reach for target position
    GRASP = 2  # Grasp target object
    PUSH = 3  # Push in delta offset direction
    OPEN = 4  # Open gripper


def quat_to_euler(quat):
    """
    Converts from quat to euler
    input quaternion should be in order x, y, z, w
    """
    r = Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]]).as_euler('xyz')
    return r


def rotate_frame_z(frame, angle):
    """ Rotate frame around z """
    r = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return np.dot(r, frame)


def get_normalized_value(pos, low, high):
    """ Input an unnormalized position and output normalized between -1 and 1 in the working range"""
    pos = (pos - low) / (high - low)
    pos = 2 * pos - 1
    pos = np.clip(pos, -1, 1)
    return pos


def get_normalized_angle(angle):
    """
    Ensures angle in the interval [-pi/2, pi/2]
    and normalizes that to [-1, 1]
    Finally, for some reason the object angle is flipped (at least for lift)
    so we need to flip to translate to the robot reference)
    """
    if angle < -np.pi / 2:
        angle += np.pi
    elif angle > np.pi / 2:
        angle -= np.pi
    # Normalize
    angle /= -np.pi / 2
    return angle


@dataclass
class RobosuiteParameters:
    """Data class for parameters for the Robosuite simulations."""
    type: str = ""                    # Benchmark type/name
    use_addl_info_func: bool = False  # Add info for making video
    image_obs_in_info: bool = False   # Include image observation in info


class RobosuiteInterface():
    """Interface to Robosuite simulations"""
    def __init__(self, parameters: Any = None, seed: int = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.ready_for_action = False  # At most one action each tick
        self.type = parameters.type
        self.env = self.make_env()
        self.observation = self.env.reset()
        self.action_type = 0
        self.action = np.zeros(self.env.action_space.low.size)
        self.actions = []
        self.use_addl_info_func = parameters.use_addl_info_func
        self.addl_infos = []
        self.image_obs_in_info = parameters.image_obs_in_info
        self.path_length = 0
        self.rewards = []
        self.env_infos = []
        self.done = False
        self.grasped_object = "none"

    def set_make_video(self):
        """ Sets up saving of information for video """
        self.use_addl_info_func = True
        self.image_obs_in_info = True

    def make_env(self):
        """ Create robosuite environment """
        search_space = env_params[self.type]

        sweeper = hyp.DeterministicHyperparameterSweeper(
            search_space, default_parameters=base_variant,
        )
        # We always get just one, but the sweeper helps set up all the parameters
        variant = sweeper.iterate_hyperparameters()[0]

        env_variant = variant['env_variant']

        controller_config = suite.load_controller_config(default_controller=env_variant['controller_type'])
        controller_config_update = env_variant.get('controller_config_update', {})
        controller_config.update(controller_config_update)

        robot_type = env_variant.get('robot_type', 'Panda')

        obs_keys = env_variant['robot_keys'] + env_variant['obj_keys']

        self.max_path_length = variant['algorithm_kwargs']['max_path_length']

        env = suite.make(
            env_name=env_variant['env_type'],
            robots=robot_type,
            has_renderer=False,  # Renderer doesn't work with wsl
            has_offscreen_renderer=True,
            use_camera_obs=False,
            controller_configs=controller_config,
            **env_variant['env_kwargs']
        )

        env = GymWrapper(env, keys=obs_keys)

        return env

    def get_feedback(self) -> bool:
        """ Read feedback (dummy)"""
        self.action = np.zeros(self.env.action_space.low.size)
        return not self.done

    def send_references(self):
        """ Sends new references to simulation and takes step"""
        self.step()

    def step(self):
        """
        Take one step in simulation.
        Based on rollout in rollout_functions.py of maple but
        adapted to fit the BT architecture
        """

        if self.use_addl_info_func:
            self.addl_infos.append(self.addl_info_func())

        next_o, r, self.done, env_info = self.env.step(copy.deepcopy(self.action),
                                                       image_obs_in_info=self.image_obs_in_info)

        new_steps = env_info.get('num_ac_calls', 1)
        if self.path_length + new_steps <= self.max_path_length:
            self.path_length += new_steps

            self.rewards.append(r)
            self.actions.append(self.action)
            self.env_infos.append(env_info)
            self.observation = next_o
        else:
            self.done = True

    def addl_info_func(self):
        """ Adds info for video use """
        skill = self.env.skill_controller.get_skill_name_from_action(self.action)
        info = dict(
            skill=skill,
        )
        return info

    def get_frames(self,
                   pad_length=5,
                   pad_color=0,
                   imsize=512
                   ):
        """
        Returns frames for video of last run. Based on dump_video in visualization.py of maple but
        adapted to fit the BT architecture
        """
        skill_name_map = self.env.env.skill_controller.get_full_skill_name_map()

        frames = []
        for j in range(len(self.env_infos)):
            imgs = self.env_infos[j]['image_obs']
            skill = self.addl_infos[j]['skill']

            skill_name = skill_name_map[skill]
            ac_str = skill_name
            success = self.env_infos[max(j-1, 0)].get('success', False)

            for img in imgs:
                img = np.flip(img, axis=0)
                img[-80:, :, :] = 235
                img = visualization.get_image(
                    img,
                    pad_length=pad_length,
                    pad_color=(0, 225, 0) if success else pad_color,
                    imsize=imsize,
                )
                if success:
                    ac_str = 'Success'
                img = visualization.annotate_image(
                    img,
                    text=ac_str,
                    imsize=imsize,
                    color=(0, 175, 0) if success else (0, 0, 0,),
                    loc='ll',
                )
                frames.append(img)

            if success:
                break

        return frames

    def save_video(self,
                   frames,
                   logdir="filmtest",
                   rows=3,
                   columns=6,
                   pad_length=5,
                   horizon=150,
                   imsize=512,
                   num_channels=3
                   ):
        """
        Compiles and saves video of input frames. Based on dump_video in visualization.py of maple but
        adapted to fit the BT architecture
        """
        H = imsize
        W = imsize
        N = rows * columns

        for i in range(len(frames)):
            last_img = frames[i][-1]
            for _ in range(horizon - len(frames[i])):
                frames[i].append(last_img)

        frames = np.array(frames, dtype=np.uint8)
        path_length = frames.size // (
                N * (H + 2 * pad_length) * (W + 2 * pad_length) * num_channels
        )
        frames = np.array(frames, dtype=np.uint8).reshape(
            (N, path_length, H + 2 * pad_length, W + 2 * pad_length, num_channels)
        )
        f1 = []
        for k1 in range(columns):
            f2 = []
            for k2 in range(rows):
                k = k1 * rows + k2
                f2.append(frames[k:k + 1, :, :, :, :].reshape(
                    (path_length, H + 2 * pad_length, W + 2 * pad_length,
                     num_channels)
                ))
            f1.append(np.concatenate(f2, axis=1))
        output_data = np.concatenate(f1, axis=2)

        filename = osp.join(logdir, 'video.mp4')
        with imageio.get_writer(filename, fps=24) as writer:
            for frame in output_data:
                writer.append_data(frame)
        print("Saved video to ", filename)

    def make_video(self,
                   pad_length=5,
                   pad_color=0,
                   imsize=512,
                   logdir="filmtest",
                   horizon=150,
                   num_channels=3
                   ):
        """
        Make and saves video. Basically a combination of the function get_frame and save_video but can only
        put one run in the video, i.e. just one row and one column
        """
        frames = [self.get_frames(pad_length=pad_length, pad_color=pad_color, imsize=imsize)]

        self.save_video(frames,
                        logdir=logdir,
                        rows=1,
                        columns=1,
                        pad_length=pad_length,
                        horizon=horizon,
                        imsize=imsize,
                        num_channels=num_channels
                        )

    def get_robot_position(self):
        """ Returns robot position """
        return self.observation[0:3]

    def get_robot_yaw(self):
        """ Returns robot yaw angle """
        _, _, robot_yaw = quat_to_euler(self.observation[3:7])
        return robot_yaw
    
    def to_global_frame(self, position, input_frame=None):
        """ Switch from input frame to global frame """
        frame_position = np.array([0.0, 0.0, 0.0])
        if input_frame is not None:
            frame_position = self.get_object_position(input_frame)
            frame_yaw = self.get_object_yaw(input_frame)

            if frame_yaw != 0.0:
                position = rotate_frame_z(position, frame_yaw)

        return frame_position + position

    def at_pos(self, target_object, target_offset):
        """ Is robot currently at the target position"""
        target_pos = self.to_global_frame(target_offset, input_frame=target_object)

        if np.linalg.norm(self.get_robot_position() - target_pos) < IN_POS_DIST:
            return True
        return False

    def object_at_pos(self, target_object, pos, threshold=0.0):
        """ Is target object currently at the target position"""
        if threshold == 0.0:
            threshold = IN_POS_DIST
        object_position = self.get_object_position(target_object)
        if np.linalg.norm(object_position - pos) < threshold:
            return True
        return False

    def at_yaw(self, yaw, target_object=None):
        """ Is robot currently at the target yaw angle """
        if self.env.env.robot_configs[0]['controller_config']['type'] == 'OSC_POSITION':  # No angle control, so assume always ok
            return True
        robot_yaw = self.get_robot_yaw()

        object_yaw = 0.0
        if target_object is not None:
            object_yaw = self.get_object_yaw(target_object)
        target_yaw = object_yaw + yaw

        diff = np.minimum(
            (target_yaw - robot_yaw) % (np.pi),
            (robot_yaw - target_yaw) % (np.pi)
        )
        if diff <= AT_ROT_DIFF:
            return True
        return False

    def object_at_yaw(self, yaw, target_object):
        """ Is object currently at the target yaw angle """
        object_yaw = self.get_object_yaw(target_object)

        diff = np.minimum(
            (yaw - object_yaw) % (np.pi),
            (object_yaw - yaw) % (np.pi)
        )
        if diff <= AT_ROT_DIFF:
            return True
        return False

    def gripper_at_ref(self, gripper_ref):
        """ Is gripper at reference state """
        if gripper_ref == -1.0:
            if abs(self.observation[7] - self.observation[8]) >= 0.06 and \
               abs(self.observation[9]) <= 0.2 and abs(self.observation[10]) <= 0.2:
                return True
            return False
        elif gripper_ref == 1.0:  # Check for grasp
            if abs(self.observation[7] - self.observation[8]) <= 0.06 and \
               abs(self.observation[7] - self.observation[8]) >= 0.015 and \
               self.observation[9] <= 0.2 and self.observation[10] <= 0.2:
                return True
            return False
        return True

    def set_action_type(self, action_type):
        """ Set action type for next step """
        self.action_type = action_type
        if self.type == 'wipe':
            self.action[0:3] = np.zeros(3)
            if action_type in [ActionTypes.ATOMIC, ActionTypes.REACH]:
                self.action[action_type] = 1.0
            elif action_type == ActionTypes.PUSH:
                self.action[2] = 1.0
        else:
            self.action[0:5] = np.zeros(5)
            self.action[action_type] = 1.0

    def get_xyz_bounds(self):
        """ Gets the xyz bounds of the current action """
        xyz_bounds = None
        try:
            if self.action_type == ActionTypes.ATOMIC:
                xyz_bounds = self.env.skill_controller._config['atomic_config']['global_xyz_bounds']
            elif self.action_type == ActionTypes.REACH:
                xyz_bounds = self.env.skill_controller._config['reach_config']['global_xyz_bounds']
            elif self.action_type == ActionTypes.GRASP:
                xyz_bounds = self.env.skill_controller._config['grasp_config']['global_xyz_bounds']
            elif self.action_type == ActionTypes.PUSH:
                xyz_bounds = self.env.skill_controller._config['push_config']['global_xyz_bounds']
        except KeyError:
            pass  # No problem, we will take base_config instead
        if xyz_bounds is None:
            xyz_bounds = self.env.skill_controller._config['base_config']['global_xyz_bounds']

        return xyz_bounds

    def get_object_position(self, target_object):
        """ Returns object position from observation """
        if target_object == 'robot_ee':
            return self.get_robot_position()
        elif target_object == 'none':
            # Just set up an origin inside the workspace [0.0, 0.0, 0.0] is not
            if self.type == 'peg_ins':
                return np.array([-0.075, -0.15, 0.8])
            else:
                return np.array([-0.075, 0.0, 0.8])
        elif self.type == 'lift':
            if target_object == 'cube':
                return self.observation[11:14]
        elif self.type == 'door':
            if target_object == 'handle':
                return self.observation[14:17]
        elif self.type == 'wipe':
            if target_object == 'centroid':
                return self.observation[10:13]
        elif self.type == 'pnp':
            if target_object == 'can':
                return self.observation[11:14]
        elif self.type == 'stack':
            if target_object == 'red':
                return self.observation[11:14]
            elif target_object == 'green':
                return self.observation[18:21]
        elif self.type == 'nut':
            if target_object == 'nut':
                if self.action_type == ActionTypes.GRASP:
                    # Account for nut being dropped from high above in the start so z is not correct
                    pos = self.observation[11:14]
                    if pos[2] < 0.88:
                        pos[2] = 0.81
                    return pos
                else:
                    return self.observation[11:14]
            elif target_object == 'robot_to_nut':
                return self.observation[18:21]
        elif self.type == 'cleanup':
            if target_object == 'spam':
                return self.observation[11:14]
            elif target_object == 'jello':
                return self.observation[14:17]
        elif self.type == 'peg_ins':
            if target_object == 'peg':
                return self.observation[11:14]
            elif target_object == 'hole':
                return self.observation[21:24]

    def get_object_yaw(self, target_object):
        """ Returns object yaw angle from observation """
        object_yaw = 0.0
        if target_object == 'robot_ee':
            object_yaw = self.get_robot_yaw()
        elif target_object == 'none':
            object_yaw = 0.0
        elif self.type == 'lift':
            if target_object == 'cube':
                _, _, object_yaw = quat_to_euler(self.observation[14:18])
        elif self.type == 'door':
            if target_object == 'handle':
                # Angle between handle and door is 1.108
                object_yaw = math.atan2(self.observation[15] - self.observation[12], self.observation[14] - self.observation[11]) - 1.108
        elif self.type == 'pnp':
            if target_object == 'can':
                _, _, object_yaw = quat_to_euler(self.observation[14:18])
        elif self.type == 'stack':
            if target_object == 'red':
                _, _, object_yaw = quat_to_euler(self.observation[14:18])
            elif target_object == 'green':
                _, _, object_yaw = quat_to_euler(self.observation[21:25])
        elif self.type == 'nut':
            if target_object == 'nut':
                _, _, object_yaw = quat_to_euler(self.observation[14:18])
            elif target_object == 'robot_to_nut':
                _, _, object_yaw = quat_to_euler(self.observation[21:25])
        elif self.type == 'cleanup':
            if target_object == 'spam':
                _, _, object_yaw = quat_to_euler(self.observation[17:21])
            elif target_object == 'jello':
                _, _, object_yaw = quat_to_euler(self.observation[21:25])
        elif self.type == 'peg_ins':
            if target_object == 'peg':
                _, _, object_yaw = quat_to_euler(self.observation[14:18])
            elif target_object == 'hole':
                object_yaw = 0.0
        return object_yaw

    def is_object_aligned(self, target_object):
        """
        Check if target object is aligned by checking if cos of angle difference is large enough and
        perpendicular distance is small enough
        """
        aligned = False
        if self.type == 'peg_ins':
            if target_object == 'peg':
                if self.observation[24] > 0.97 and self.observation[26] < 0.03:
                    aligned = True
        return aligned

    def get_handle_angle(self):
        """ Returns angle of door handle """
        if self.type == 'door':
            return self.observation[24]
        return 0.0

    def get_door_angle(self):
        """ Returns angle of door hinge """
        if self.type == 'door':
            return self.observation[23]
        return 0.0

    def get_normalized_pos(self, pos, around_zero=False):
        """ Get normalized position """

        xyz_bounds = copy.deepcopy(self.get_xyz_bounds())

        if around_zero:
            x_mean = (xyz_bounds[0][0] + xyz_bounds[1][0]) / 2
            xyz_bounds[0][0] -= x_mean
            xyz_bounds[1][0] -= x_mean
            y_mean = (xyz_bounds[0][1] + xyz_bounds[1][1]) / 2
            xyz_bounds[0][1] -= y_mean
            xyz_bounds[1][1] -= y_mean
            z_mean = (xyz_bounds[0][2] + xyz_bounds[1][2]) / 2
            xyz_bounds[0][2] -= z_mean
            xyz_bounds[1][2] -= z_mean

        normalized_x = get_normalized_value(pos[0], xyz_bounds[0][0], xyz_bounds[1][0])
        normalized_y = get_normalized_value(pos[1], xyz_bounds[0][1], xyz_bounds[1][1])
        normalized_z = get_normalized_value(pos[2], xyz_bounds[0][2], xyz_bounds[1][2])
        return [normalized_x, normalized_y, normalized_z]

    def set_target_and_offset(self, target_object, offset, input_offset_is_global=False, normalize_around_zero=False):
        """ Set target object """
        if input_offset_is_global:
            if target_object is not None:
                target_pos = self.get_object_position(target_object) + offset
            else:
                target_pos = offset
        else:
            target_pos = self.to_global_frame(offset, input_frame=target_object)

        normalized_pos = self.get_normalized_pos(target_pos, normalize_around_zero)

        if self.type == 'wipe':
            self.action[3:6] = normalized_pos
        else:
            self.action[5:8] = normalized_pos

        return target_pos

    def set_yaw(self, yaw, target_object=None):
        """ Set yaw angle reference. If object is not None, the angle will be relative to the object yaw """
        if self.env.env.robot_configs[0]['controller_config']['type'] == 'OSC_POSITION':
            pass  # Yaw not used
        else:
            object_yaw = 0.0
            if target_object is not None:
                object_yaw = self.get_object_yaw(target_object)
            self.action[8] = get_normalized_angle(object_yaw + yaw)

    def set_gripper(self, gripper_ref):
        """ Set gripper reference 1 is closed, -1 is open """
        if self.type == 'wipe':
            pass  # Gripper not used
        elif self.env.env.robot_configs[0]['controller_config']['type'] == 'OSC_POSITION':
            self.action[8] = gripper_ref
        else:
            self.action[9] = gripper_ref

    def set_delta_offset(self, delta_offset):
        """ Sets the delta offset used in the push skill """
        normalized_offset = np.array(delta_offset)
        normalized_offset /= self.env.skill_controller._config['push_config']['delta_xyz_scale']

        if self.type == 'wipe':
            self.action[6:9] = normalized_offset
        elif self.env.env.robot_configs[0]['controller_config']['type'] == 'OSC_POSITION':
            self.action[9:12] = normalized_offset
        else:
            self.action[10:13] = normalized_offset

    def set_grasped_object(self, grasped_object):
        """ Sets grasped object """
        self.grasped_object = grasped_object

    def get_grasped_object(self):
        """ Gets grasped object"""
        return self.grasped_object

    def get_fitness(self):
        """ Gets the fitness score of the current logged run """
        fitness = np.sum([env_info.get('reward_actions', 0) for env_info in self.env_infos])
        affordance_penalty = 15 * sum((1-env_info['aff_reward']) * env_info['num_ac_calls']
                                      for env_info in self.env_infos)

        if self.path_length < self.max_path_length:
            fitness += self.env_infos[-1]['reward_skills'] * (self.max_path_length - self.path_length)
            affordance_penalty += (15 * (1-self.env_infos[-1]['aff_reward'])) * \
                                  (self.max_path_length - self.path_length)

        success = False
        if any([env_info.get('success', False) for env_info in self.env_infos]):
            success = True

        return fitness, affordance_penalty, self.path_length, success
