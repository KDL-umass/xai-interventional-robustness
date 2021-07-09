###########
# Authors:Kaleigh Clary 2021.04.06 -- set up the template wrappers and abstract FeatureVecWrapper class
#
# Authors: Jack Kenney 2021.04.06
# _get_feature_vec(), ship_x(),
# ship_laser_mid_air(), num_enemies(), lowest_enemy_height(), ufo_on_screen(),
# ufo_sign_distance(), in_danger()
# set up the test case file to test these functions and added many test cases
#
# Authors: Erica Cai 2021.04.06 -- shield_i_xrange(), compute_shield_xranges(), ship_xrange(),
# partially_under_shield(), completely_under_shield(), sign_distance_closest_shield_partial(),
# sign_distance_closest_shield_complete(), sign_distance_closest_UN_shield_complete(), added some test cases
###########

import random

import gym
import numpy as np
from toybox import Input
from toybox.envs.atari import SpaceInvadersEnv
from gym.spaces.box import Box

from space_invaders_wrapper.wrappers import FeatureVecWrapper


# constants from toybox code
ENEMY_SIZE = 16, 10
NUM_ENEMIES = 36
NUM_SHIELDS = 3

idx = 0
INDEX_OF_observation_count = idx; idx += 1
INDEX_OF_score = idx; idx += 1
INDEX_OF_num_lives = idx; idx += 1
INDEX_OF_level_num = idx; idx += 1
INDEX_OF_ship_x = idx; idx += 1
INDEX_OF_ship_laser_mid_air = idx; idx += 1
INDEX_OF_num_enemies = idx; idx += 1
INDEX_OF_enemy_direction = idx; idx += 1
INDEX_OF_lowest_enemy_height = idx; idx += 1
INDEX_OF_below_enemy = idx; idx += 1
INDEX_OF_in_danger = idx; idx += 1
INDEX_OF_ufo_on_screen = idx; idx += 1
INDEX_OF_ufo_sign_distance = idx; idx += 1
INDEX_OF_partially_under_shield = idx; idx += 1
INDEX_OF_completely_under_shield = idx; idx += 1
INDEX_OF_sign_distance_closest_shield_partial = idx; idx += 1
INDEX_OF_sign_distance_closest_shield_complete = idx; idx += 1
INDEX_OF_sign_distance_closest_UN_shield_complete = idx; idx += 1
INDEX_OF_num_shields = idx; idx += 1
INDEX_OF_ship_laser_pos_x = idx; idx += 1
INDEX_OF_ship_laser_pos_y = idx; idx += 1
INDEX_OF_shield_y = idx; idx += 1
INDEX_OF_shield_1x = idx; idx += 1
INDEX_OF_shield_2x = idx; idx += 1
INDEX_OF_shield_3x = idx; idx += 1
INDEX_OF_enemy_xs_start = idx; idx += NUM_ENEMIES - 1
INDEX_OF_enemy_xs_end = idx; idx += 1


class SpaceInvadersFeatureVecWrapper(FeatureVecWrapper):
    def __init__(self, tbenv):
        assert type(tbenv) == SpaceInvadersEnv
        tbenv.observation_space = Box(low=-float("inf"), high=float("inf"), shape=(INDEX_OF_enemy_xs_end+1,))
        self.timestep = 0
        super().__init__(tbenv)

    def observation(self, observation):
        """
        FeatureVecWrapper subclasses gym.ObservationWrapper
        we must implement the abstract class gym.ObservationWrapper.observation.
        """
        self.timestep += 1
        return self.interpret_state()

    def interpret_state(self):
        return self._get_feature_vec()

    def _get_feature_vec(self):
        """
        Return a set of executed python functions that act as feature oracles.
        """
        state = self.toybox.state_to_json()
        feature_fns = [
            self.observation_count,
            self.score,
            self.num_lives,
            self.level_num,
            self.ship_x,
            self.ship_laser_mid_air,
            self.num_enemies,
            self.enemy_direction,
            self.lowest_enemy_height,
            self.below_enemy,
            self.in_danger,
            self.ufo_on_screen,
            self.ufo_sign_distance,
            self.partially_under_shield,
            self.completely_under_shield,
            self.sign_distance_closest_shield_partial,
            self.sign_distance_closest_shield_complete,
            self.sign_distance_closest_UN_shield_complete,
            self.num_shields,
            self.ship_laser_pos_x,
            self.ship_laser_pos_y,
            self.shield_y,
            self.shield_1x,
            self.shield_2x,
            self.shield_3x,
        ]
        features = [int(f(state)) for f in feature_fns]
        # dynamic [ure
        features.extend(self.enemy_xs(state))
        return np.array(features)

    def observation_count(self, state):
        """
        Returns number of observations requested as proxy for timestep (not available in state).
        """
        return self.timestep

    def score(self, state):
        """
        Returns current game score.
        """
        return state["score"]

    def num_lives(self, state):
        """
        Returns number of agent lives left.
        """
        return state["lives"]

    def level_num(self, state):
        if state["level"] is not None:
            return state["level"]
        else:
            return 0

    def ship_x(self, state):
        """
        Returns the agent ship's horizontal location in the frame.
        """
        return state["ship"]["x"]

    def ship_laser_mid_air(self, state):
        """
        Returns binary output of whether or not the agent's laser has been fired.
        """
        ship_laser = state["ship_laser"]
        return ship_laser is not None

    def num_enemies(self, state):
        """
        Returns the number of enemies that are alive on screen.
        """
        enemies = state["enemies"]
        alive = list(map(lambda e: int(e["alive"]), enemies))
        return sum(alive)

    def num_enemy_moves(self, state):
        """
        Returns number of time steps in enemy movements from start of game.
        """
        return state["enemies_movement"]["move_counter"]

    def enemy_direction(self, state):
        """
        Returns -1 if enemies moving left, +1 if moving right, 0 if unknown.
        """
        dir = 0
        if state["enemies_movement"]["move_dir"] == "Right":
            dir = 1
        elif state["enemies_movement"]["move_dir"] == "Left":
            dir = -1
        return dir

    def lowest_enemy_height(self, state):
        """
        Returns the distance to the lowest enemy on the screen,
        relative to the ship's `y` position.
        """
        ship = state["ship"]
        enemies = state["enemies"]
        return ship["y"] - max(map(lambda e: e["y"] if e["alive"] else 0, enemies))

    def below_enemy(self, state):
        """
        Returns whether ship is below enemy.
        """
        enemy_xs = np.array(list(map(lambda e: e["x"], state["enemies"])))
        closest_enemy = np.argmin(enemy_xs - state["ship"]["x"])
        enemy_x_center = state["enemies"][closest_enemy]["x"]
        left_edge = enemy_x_center - ENEMY_SIZE[0] / 2
        right_edge = enemy_x_center + ENEMY_SIZE[0] / 2
        return left_edge <= state["ship"]["x"] and state["ship"]["x"] <= right_edge

    def ufo_on_screen(self, state):
        """
        Returns a binary variable of whether or not the ufo is on the screen.
        """
        ufo = state["ufo"]
        return 0 < ufo["x"] and ufo["x"] < 320

    def ufo_sign_distance(self, state):
        """
        Returns the signed horizontal distance from the agent to the ufo.
        Positive values to the right, negative values to the left.
        """
        return state["ufo"]["x"] - state["ship"]["x"]

    def in_danger(self, state):
        """
        Returns whether the agent is below an enemy laser.
        """
        ship_x, ship_xw = self.ship_xrange(state)
        under_laser = map(
            lambda l: ship_x <= l["x"] and l["x"] <= ship_xw, state["enemy_lasers"]
        )
        return any(under_laser)

    def shield_i_info(self, state, idx):
        """
        Returns the x,y location for the shield at `idx`.
        """
        x = state["shields"][idx]["x"]
        y = state["shields"][idx]["y"]
        return x, y

    def shield_i_xrange(self, state, idx):
        """
        Return the leftmost x coordinate of the shield and the
        width of the shield from the pixel data
        """
        x = state["shields"][idx]["x"]
        width = np.shape(state["shields"][0]["data"])[1]
        return x, x + width

    def shield_i_yrange(self, state, idx):
        """
        extract the uppermost y coordinate of the shield
        and the height of the shield from the pixel data going down is positive
        """
        y = state["shields"][idx]["y"]
        height = np.shape(state["shields"][0]["data"])[0]
        return y, y + height

    def compute_shield_xranges(self, state):
        """
        Return the x ranges of all of the existing shields
        """
        n_shields = len(state["shields"])
        shield_pos_list = [self.shield_i_xrange(state, j) for j in range(n_shields)]
        return shield_pos_list

    def ship_xrange(self, state):
        """
        Compute the x range for the ship, assuming that x is the leftmost point of the ship
        """
        x = state["ship"]["x"]
        width = state["ship"]["w"]
        return x, x + width

    def partially_under_shield(self, state, offset=0):
        """
        Returns whether the ship is partially under the shield
        or, determine whether, if you moved the ship left or right,
        if it would be partially under the shield.
        """
        # check for ship position in range covered by shield
        if len(state["shields"]) == 0:
            return False
        ship_x_range = self.ship_xrange(state)
        left = ship_x_range[0] + offset
        right = ship_x_range[1] + offset
        if left < 0 or right > 320:
            return False
        shield_xranges = self.compute_shield_xranges(state)
        for ship_x in range(left, right + 1):
            if any(
                [
                    (shield_xs[0] <= ship_x <= shield_xs[1])
                    for shield_xs in shield_xranges
                ]
            ):
                return True
        return False

    def completely_under_shield(self, state):
        """
        Returns whether the ship is entirely under the shield.
        """
        if len(state["shields"]) == 0:
            return False
        ship_x_range = self.ship_xrange(state)
        shield_xranges = self.compute_shield_xranges(state)
        for ship_x in range(ship_x_range[0], ship_x_range[1] + 1):
            if (
                any(
                    [
                        (shield_xs[0] <= ship_x <= shield_xs[1])
                        for shield_xs in shield_xranges
                    ]
                )
                != True
            ):
                return False
        return True

    def sign_distance_closest_shield_partial(self, state):
        """
        Return the minimum distance to the closest shield not the distance to go under a shield,
        but just the distance to touch the border of a shield four ways to match -
        for the shield and ship, both left most sides match OR both right most sides match OR
        the left most of the ship and the right most side of the shield matches OR
        the right most of the ship and the left most side of the shield matches.
        """
        if len(state["shields"]) == 0:
            return 320
        ship_x_range = self.ship_xrange(state)
        shield_xranges = self.compute_shield_xranges(state)
        min = 320
        if self.partially_under_shield(state) == True:
            return 0
        for r in shield_xranges:
            d1 = abs(r[0] - ship_x_range[0])
            d2 = abs(r[1] - ship_x_range[0])
            d3 = abs(r[0] - ship_x_range[1])
            d4 = abs(r[1] - ship_x_range[1])
            if d1 < abs(min):
                min = r[0] - ship_x_range[0]
            if d2 < abs(min):
                min = r[1] - ship_x_range[0]
            if d3 < abs(min):
                min = r[0] - ship_x_range[1]
            if d4 < abs(min):
                min = r[1] - ship_x_range[1]
        return min

    def sign_distance_closest_shield_complete(self, state):
        """
        Determine the minimum distance to go completely under the shield
        minimum distance from the first point where both the left most sides
        (of the shield and ship) match OR
        minimum distance from the first point where both the right most sides
        (of the shield and ship) match
        """
        if len(state["shields"]) == 0:
            return 320
        ship_x_range = self.ship_xrange(state)
        shield_xranges = self.compute_shield_xranges(state)
        min = 320
        if self.completely_under_shield(state) == True:
            return 0
        for r in shield_xranges:
            d1 = abs(r[0] - ship_x_range[0])
            d4 = abs(r[1] - ship_x_range[1])
            if d1 < abs(min):
                min = r[0] - ship_x_range[0]
            if d4 < abs(min):
                min = r[1] - ship_x_range[1]
        return min

    def sign_distance_closest_UN_shield_complete(self, state):
        """
        Return the minimum distance to go completely ***NOT*** under the shield
        minimum distance where, for the shield and ship,
        (the first time that the left most side of the ship and
        right most side of the shield don't match AND
        the ship is not under another shield!)
        OR (the first time that the right most side of the ship and left
        most side of the shield don't match AND the ship is not under another shield).
        """
        if len(state["shields"]) == 0:
            return 0
        ship_x_range = self.ship_xrange(state)
        shield_xranges = self.compute_shield_xranges(state)
        min_y = 320
        if self.partially_under_shield(state) == False:
            return 0
        for r in shield_xranges:
            d1 = abs(r[0] - ship_x_range[1])
            d4 = abs(r[1] - ship_x_range[0])
            if (
                d1 < abs(min_y)
                and self.partially_under_shield(state, r[0] - ship_x_range[1] - 1)
                == False
            ):
                min_y = r[0] - ship_x_range[1] - 1
            if (
                d4 < abs(min_y)
                and self.partially_under_shield(state, r[1] - ship_x_range[0] + 1)
                == False
            ):
                min_y = r[1] - ship_x_range[0] + 1
        return min_y

    def num_shields(self, state):
        """
        Returns the number of shields on the screen.
        """
        return len(state["shields"])

    def ship_laser_pos_x(self, state):
        """
        Returns the x position of the player's laser.
        """
        if state["ship_laser"] is not None:
            return state["ship_laser"]["x"]
        else:
            return 0

    def ship_laser_pos_y(self, state):
        """
        Returns the y position of the player's laser.
        """
        if state["ship_laser"] is not None:
            return state["ship_laser"]["y"]
        else:
            return 0

    def shield_y(self, state):
        """
        Returns y position of shields.
        """
        if 1 <= len(state["shields"]):
            return state["shields"][0]["y"]
        else:
            return 210

    def shield_1x(self, state):
        """
        Returns x position of first shield.
        """
        if 1 <= len(state["shields"]):
            return state["shields"][0]["x"]
        else:
            return 0

    def shield_2x(self, state):
        """
        Returns x position of second shield.
        """
        if 2 <= len(state["shields"]):
            return state["shields"][1]["x"]
        else:
            return 0

    def shield_3x(self, state):
        """
        Returns x position of third shield.
        """
        if 3 <= len(state["shields"]):
            return state["shields"][2]["x"]
        else:
            return 0

    def enemy_xs(self, state):
        """
        Returns list of enemy x values. 0 used if they are not alive.
        """
        return list(map(lambda e: e["x"] if e["alive"] else 0, state["enemies"]))
