#!/usr/bin/env python2
# -----------------------------------------------------------------------------
#   @brief:
#       The Centipede environments.
#   @author:
#       Tingwu (Wilson) Wang, Aug. 30nd, 2017
# -----------------------------------------------------------------------------

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import environments.init_path as init_path
import os
import num2words


class CentipedeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    '''
        @brief:
            In the CentipedeEnv, we define a task that is designed to test the
            power of transfer learning for gated graph neural network.
        @children:
            @CentipedeFourEnv
            @CentipedeEightEnv
            @CentipedeTenEnv
            @CentipedeTwelveEnv
    '''

    def __init__(self, CentipedeLegNum=4, is_crippled=False):

        # get the path of the environments
        if is_crippled:
            xml_name = 'CpCentipede' + self.get_env_num_str(CentipedeLegNum) + \
                '.xml'
        else:
            xml_name = 'Centipede' + self.get_env_num_str(CentipedeLegNum) + \
                '.xml'
        xml_path = os.path.join(init_path.get_base_dir(),
                                'environments', 'assets',
                                xml_name)
        xml_path = str(os.path.abspath(xml_path))
        self.num_body = int(np.ceil(CentipedeLegNum / 2.0))
        self._control_cost_coeff = .5 * 4 / CentipedeLegNum
        self._contact_cost_coeff = 0.5 * 1e-3 * 4 / CentipedeLegNum

        self.torso_geom_id = 1 + np.array(range(self.num_body)) * 5
        # make sure the centipede is not born to be end of episode
        self.body_qpos_id = 6 + 6 + np.array(range(self.num_body)) * 6
        self.body_qpos_id[-1] = 5

        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)

        utils.EzPickle.__init__(self)

    def get_env_num_str(self, number):
        num_str = num2words.num2words(number)
        return num_str[0].upper() + num_str[1:]

    def step(self, a):
        xposbefore = self.get_body_com("torso_" + str(self.num_body - 1))[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso_" + str(self.num_body - 1))[0]

        # calculate reward
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = self._control_cost_coeff * np.square(a).sum()
        contact_cost = self._contact_cost_coeff * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        # check if finished
        state = self.state_vector()
        notdone = np.isfinite(state).all() and \
            self._check_height() and self._check_direction()
        done = not notdone
        # done = False

        obs = self._get_obs()

        return obs, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward
        )

    def _get_obs(self):
        """ Original NerveNet Obs """
        # print(self.sim.data.qpos.shape)
        # print(self.sim.data.qvel.shape)
        # print(self.sim.data.cfrc_ext.shape)
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def __get_obs(self):
        """
        My implementation of Obs including `Root`

        Description of Obs
            - body_xpos: Cartesian position of body frame     (nbody x 3)
            - body_xquat: Cartesian orientation of body frame (nbody x 4)
            - cvel: com-based velocity [3D rot; 3D tran]      (nbody x 6)
            - cinert: com-based body inertia and mass         (nbody x 10)

        :return
            - flat_obs = flattened observation
            - graph_obs = obs are organised by associating data with each node

        """
        # print(self.sim.data.body_xpos.shape) # 11 x 3
        # print(self.sim.data.body_xquat.shape) # 11 x 4
        # print(self.sim.data.cvel.shape) # 11 x 6
        # print(self.sim.data.cinert.shape) # 11 x 10
        flat_obs = np.concatenate([
            self.sim.data.body_xpos[1:, :].flat,
            self.sim.data.body_xquat[1:, :].flat,
            self.sim.data.cvel[1:, :].flat,
            self.sim.data.cinert[1:, :].flat
        ])
        # remove the "root"
        graph_obs = np.concatenate([
            self.sim.data.body_xpos[1:, :],
            self.sim.data.body_xquat[1:, :],
            self.sim.data.cvel[1:, :],
            self.sim.data.cinert[1:, :]
        ], axis=-1)
        return {"flat_obs": flat_obs,
                "graph_obs": graph_obs}

    def reset_model(self):
        while True:
            qpos = self.init_qpos + self.np_random.uniform(
                size=self.model.nq, low=-.1, high=.1
            )
            qpos[self.body_qpos_id] = self.np_random.uniform(
                size=len(self.body_qpos_id),
                low=-.1 / (self.num_body - 1),
                high=.1 / (self.num_body - 1)
            )

            qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
            self.set_state(qpos, qvel)
            if self._check_height() and self._check_direction():
                break
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.8
        body_name = 'torso_' + str(int(np.ceil(self.num_body / 2 - 1)))
        self.viewer.cam.trackbodyid = self.model.body_names.index(body_name)

    '''
    def _check_height(self):
        height = self.data.geom_xpos[self.torso_geom_id, 2]
        return (height < 1.5).all() and (height > 0.35).all()
    '''

    def _check_height(self):
        height = self.data.geom_xpos[self.torso_geom_id, 2]
        return (height < 1.15).all() and (height > 0.35).all()

    def _check_direction(self):
        y_pos_pre = self.data.geom_xpos[self.torso_geom_id[:-1], 1]
        y_pos_post = self.data.geom_xpos[self.torso_geom_id[1:], 1]
        y_diff = np.abs(y_pos_pre - y_pos_post)
        return (y_diff < 0.45).all()


'''
    the following environments are just models with even number legs
'''


class CpCentipedeFourEnv(CentipedeEnv):

    def __init__(self):
        super(CentipedeFourEnv, self).__init__(CentipedeLegNum=4, is_crippled=True)


class CpCentipedeSixEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=6, is_crippled=True)


class CpCentipedeEightEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=8, is_crippled=True)


class CpCentipedeTenEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=10, is_crippled=True)


class CpCentipedeTwelveEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=12, is_crippled=True)


class CpCentipedeFourteenEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=14, is_crippled=True)

# regular


class CentipedeFourEnv(CentipedeEnv):

    def __init__(self):
        super(CentipedeFourEnv, self).__init__(CentipedeLegNum=4)


class CentipedeSixEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=6)


class CentipedeEightEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=8)


class CentipedeTenEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=10)


class CentipedeTwelveEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=12)


class CentipedeFourteenEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=14)


class CentipedeTwentyEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=20)


class CentipedeThirtyEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=30)


class CentipedeFortyEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=40)


class CentipedeFiftyEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=50)


class CentipedeOnehundredEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=100)


'''
    the following environments are models with odd number legs
'''


class CentipedeThreeEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=3)


class CentipedeFiveEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=5)


class CentipedeSevenEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=7)
