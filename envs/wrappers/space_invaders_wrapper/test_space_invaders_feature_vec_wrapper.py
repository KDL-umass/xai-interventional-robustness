import unittest

import gym
from toybox.interventions.space_invaders import SpaceInvadersIntervention

from .space_invaders_feature_vec_wrapper import *


class TestSpaceInvadersFeatureVecWrapper(unittest.TestCase):
    def setUp(self):
        self.env = {
            "start": gym.make("SpaceInvadersToyboxNoFrameskip-v4"),
            "fewer_enemies": gym.make("SpaceInvadersToyboxNoFrameskip-v4"),
            "below_enemy": gym.make("SpaceInvadersToyboxNoFrameskip-v4"),
            "ufo_on_screen": gym.make("SpaceInvadersToyboxNoFrameskip-v4"),
            "not_under_laser": gym.make("SpaceInvadersToyboxNoFrameskip-v4"),
            "under_laser": gym.make("SpaceInvadersToyboxNoFrameskip-v4"),
            "no_shields": gym.make("SpaceInvadersToyboxNoFrameskip-v4"),
            "shield_right": gym.make("SpaceInvadersToyboxNoFrameskip-v4"),
            "shield_left": gym.make("SpaceInvadersToyboxNoFrameskip-v4"),
            "shield_special": gym.make("SpaceInvadersToyboxNoFrameskip-v4"),
            "fewer_shields": gym.make("SpaceInvadersToyboxNoFrameskip-v4"),
        }
        self.wrapper = {
            "start": SpaceInvadersFeatureVecWrapper(self.env["start"]),
            "fewer_enemies": SpaceInvadersFeatureVecWrapper(self.env["fewer_enemies"]),
            "below_enemy": SpaceInvadersFeatureVecWrapper(self.env["below_enemy"]),
            "ufo_on_screen": SpaceInvadersFeatureVecWrapper(self.env["ufo_on_screen"]),
            "not_under_laser": SpaceInvadersFeatureVecWrapper(
                self.env["not_under_laser"]
            ),
            "under_laser": SpaceInvadersFeatureVecWrapper(self.env["under_laser"]),
            "no_shields": SpaceInvadersFeatureVecWrapper(self.env["no_shields"]),
            "shield_right": SpaceInvadersFeatureVecWrapper(self.env["shield_right"]),
            "shield_left": SpaceInvadersFeatureVecWrapper(self.env["shield_left"]),
            "shield_special": SpaceInvadersFeatureVecWrapper(
                self.env["shield_special"]
            ),
            "fewer_shields": SpaceInvadersFeatureVecWrapper(self.env["fewer_shields"]),
        }

        with SpaceInvadersIntervention(
            self.env["fewer_enemies"].toybox
        ) as intervention:
            enemies = [e for e in intervention.game.enemies]
            for i, e in enumerate(enemies):
                if i % 5 == 0 or i % 7 == 0 or i >= 30:
                    e.alive = False

        with SpaceInvadersIntervention(self.env["below_enemy"].toybox) as intervention:
            enemies = [e for e in intervention.game.enemies]
            for i, e in enumerate(enemies):
                e.x += 16
        with SpaceInvadersIntervention(
            self.env["ufo_on_screen"].toybox
        ) as intervention:
            intervention.game.life_display_timer = 11
            intervention.game.ufo.x = 150
            intervention.game.ufo.appearance_counter = 499

        example_enemy_laser = {
            "x": 150,
            "y": 113,
            "w": 2,
            "h": 11,
            "t": 0,
            "movement": "Down",
            "speed": 3,
            "color": {"r": 144, "b": 144, "g": 144, "a": 255},
        }
        with SpaceInvadersIntervention(
            self.env["not_under_laser"].toybox
        ) as intervention:
            not_under_laser = example_enemy_laser
            not_under_laser["x"] = 150
            intervention.game.enemy_lasers = [not_under_laser]

        with SpaceInvadersIntervention(self.env["under_laser"].toybox) as intervention:
            under_laser = example_enemy_laser
            under_laser["x"] = intervention.game.ship.x
            intervention.game.enemy_lasers = [under_laser]

        with SpaceInvadersIntervention(self.env["no_shields"].toybox) as intervention:
            intervention.shields = []

        with SpaceInvadersIntervention(self.env["shield_right"].toybox) as intervention:
            for shield in intervention.game.shields:
                shield.x += 25

        with SpaceInvadersIntervention(self.env["shield_left"].toybox) as intervention:
            for shield in intervention.game.shields:
                shield.x -= 25

        with SpaceInvadersIntervention(
            self.env["shield_special"].toybox
        ) as intervention:
            i = 0
            for shield in intervention.game.shields:
                if i == 0:
                    shield.x -= 40
                if i == 1:
                    shield.x -= 80
                i = i + 1
        pass

        with SpaceInvadersIntervention(
            self.env["fewer_shields"].toybox
        ) as intervention:
            intervention.game.shields = []

    def test_feature_length_is_static(self):
        expected = len(self.wrapper["start"].observation(None))
        for env_key in self.wrapper.keys():
            actual = len(self.wrapper[env_key].observation(None))
            self.assertEqual(
                expected, actual, f"{env_key} environment doesn't have correct length"
            )
        pass

    def test_num_lives_start(self):
        expected = 3
        actual = self.wrapper["start"].observation(None)[INDEX_OF_num_lives]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_level_num(self):
        expected = 1
        actual = self.wrapper["start"].observation(None)[INDEX_OF_level_num]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_observation_count_start(self):
        expected = 1
        actual = self.wrapper["start"].observation(None)[INDEX_OF_observation_count]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_score_start(self):
        expected = 0
        actual = self.wrapper["start"].observation(None)[INDEX_OF_score]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ship_x(self):
        state = self.env["start"].toybox.state_to_json()
        expected = state["ship"]["x"]
        actual = self.wrapper["start"].observation(None)[INDEX_OF_ship_x]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ship_laser_mid_air(self):
        expected = 0
        actual = self.wrapper["start"].observation(None)[INDEX_OF_ship_laser_mid_air]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_num_enemies(self):
        expected = 36
        actual = self.wrapper["start"].observation(None)[INDEX_OF_num_enemies]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_num_fewer_enemies(self):
        expected = 20
        actual = self.wrapper["fewer_enemies"].observation(None)[INDEX_OF_num_enemies]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_enemy_direction(self):
        expected = 1
        actual = self.wrapper["start"].observation(None)[INDEX_OF_enemy_direction]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_lowest_enemy_height(self):
        expected = 62
        actual = self.wrapper["start"].observation(None)[INDEX_OF_lowest_enemy_height]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_lowest_enemy_height_fewer_enemies(self):
        expected = 80
        actual = self.wrapper["fewer_enemies"].observation(None)[
            INDEX_OF_lowest_enemy_height
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_not_below_enemy(self):
        expected = False
        actual = self.wrapper["start"].observation(None)[INDEX_OF_below_enemy]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_below_enemy(self):
        expected = True
        actual = self.wrapper["below_enemy"].observation(None)[INDEX_OF_below_enemy]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_in_danger_vacuous(self):
        expected = False
        actual = self.wrapper["start"].observation(None)[INDEX_OF_in_danger]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_in_danger_false(self):
        expected = False
        actual = self.wrapper["not_under_laser"].observation(None)[INDEX_OF_in_danger]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_in_danger_true(self):
        expected = True
        actual = self.wrapper["under_laser"].observation(None)[INDEX_OF_in_danger]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ufo_on_screen_false(self):
        expected = False
        actual = self.wrapper["start"].observation(None)[INDEX_OF_ufo_on_screen]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ufo_on_screen_true(self):
        expected = True
        actual = self.wrapper["ufo_on_screen"].observation(None)[INDEX_OF_ufo_on_screen]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ufo_sign_distance_neg(self):
        expected = -70
        actual = self.wrapper["start"].observation(None)[INDEX_OF_ufo_sign_distance]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ufo_sign_distance_pos(self):
        state = self.env["ufo_on_screen"].toybox.state_to_json()
        expected = state["ufo"]["x"] - state["ship"]["x"]
        actual = self.wrapper["ufo_on_screen"].observation(None)[
            INDEX_OF_ufo_sign_distance
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_no_shields(self):
        obs = self.wrapper["no_shields"].observation(None)
        pass

    def test_ship_partially_under(self):
        expected = True
        actual = self.wrapper["start"].observation(None)[
            INDEX_OF_partially_under_shield
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ship_completely_under_envshield_right(self):
        expected = False
        actual = self.wrapper["shield_right"].observation(None)[
            INDEX_OF_completely_under_shield
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ship_partially_under_envshield_right(self):
        expected = False
        actual = self.wrapper["shield_right"].observation(None)[
            INDEX_OF_partially_under_shield
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_shield_partial_envshield_right(self):
        expected = 25
        actual = self.wrapper["shield_right"].observation(None)[
            INDEX_OF_sign_distance_closest_shield_partial
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_shield_complete_envshield_right(self):
        expected = 41
        actual = self.wrapper["shield_right"].observation(None)[
            INDEX_OF_sign_distance_closest_shield_complete
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_UN_shield_complete_envshield_right(self):
        expected = 0
        actual = self.wrapper["shield_right"].observation(None)[
            INDEX_OF_sign_distance_closest_UN_shield_complete
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ship_completely_under_envshield_left(self):
        expected = False
        actual = self.wrapper["shield_left"].observation(None)[
            INDEX_OF_completely_under_shield
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ship_partially_under_envshield_left(self):
        expected = True
        actual = self.wrapper["shield_left"].observation(None)[
            INDEX_OF_partially_under_shield
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_shield_partial_envshield_left(self):
        expected = 0
        actual = self.wrapper["shield_left"].observation(None)[
            INDEX_OF_sign_distance_closest_shield_partial
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_shield_complete_envshield_left(self):
        expected = -9
        actual = self.wrapper["shield_left"].observation(None)[
            INDEX_OF_sign_distance_closest_shield_complete
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_UN_shield_complete_envshield_left(self):
        expected = 8
        actual = self.wrapper["shield_left"].observation(None)[
            INDEX_OF_sign_distance_closest_UN_shield_complete
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ship_completely_under_envshield_special(self):
        expected = True
        actual = self.wrapper["shield_special"].observation(None)[
            INDEX_OF_completely_under_shield
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ship_partially_under_envshield_special(self):
        expected = True
        actual = self.wrapper["shield_special"].observation(None)[
            INDEX_OF_partially_under_shield
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_shield_partial_envshield_special(self):
        expected = 0
        actual = self.wrapper["shield_special"].observation(None)[
            INDEX_OF_sign_distance_closest_shield_partial
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_shield_complete_envshield_special(self):
        expected = 0
        actual = self.wrapper["shield_special"].observation(None)[
            INDEX_OF_sign_distance_closest_shield_complete
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_UN_shield_complete_envshield_special(self):
        expected = 17
        actual = self.wrapper["shield_special"].observation(None)[
            INDEX_OF_sign_distance_closest_UN_shield_complete
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ship_completely_under_envfewershields(self):
        expected = False
        actual = self.wrapper["fewer_shields"].observation(None)[
            INDEX_OF_completely_under_shield
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ship_partially_under_envfewershields(self):
        expected = False
        actual = self.wrapper["fewer_shields"].observation(None)[
            INDEX_OF_partially_under_shield
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_shield_partial_envfewershields(self):
        expected = 320
        actual = self.wrapper["fewer_shields"].observation(None)[
            INDEX_OF_sign_distance_closest_shield_partial
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_shield_complete_envfewershields(self):
        expected = 320
        actual = self.wrapper["fewer_shields"].observation(None)[
            INDEX_OF_sign_distance_closest_shield_complete
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_UN_shield_complete_envfewershields(self):
        expected = 0
        actual = self.wrapper["fewer_shields"].observation(None)[
            INDEX_OF_sign_distance_closest_UN_shield_complete
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_enemy_xs_all_present(self):
        feats = self.wrapper["start"].observation(None)
        enemy_xs = feats[INDEX_OF_enemy_xs_start : INDEX_OF_enemy_xs_end + 1]
        self.assertAlmostEqual(36, len(enemy_xs))
        self.assertTrue(all(enemy_xs > 0))
        pass

    def test_enemy_xs_some_missing(self):
        feats = self.wrapper["fewer_enemies"].observation(None)
        enemy_xs = feats[INDEX_OF_enemy_xs_start : INDEX_OF_enemy_xs_end + 1]
        self.assertAlmostEqual(36, len(enemy_xs))
        self.assertAlmostEqual(feats[INDEX_OF_num_enemies], np.sum(enemy_xs > 0))
        pass

    def test_num_shields_no_shields(self):
        expected = 0
        actual = self.wrapper["fewer_shields"].observation(None)[INDEX_OF_num_shields]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_num_shields(self):
        expected = 3
        actual = self.wrapper["start"].observation(None)[INDEX_OF_num_shields]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_shield1x(self):
        expected = 84
        actual = self.wrapper["start"].observation(None)[INDEX_OF_shield_1x]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_shieldy(self):
        expected = 157
        actual = self.wrapper["start"].observation(None)[INDEX_OF_shield_y]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_shield2x(self):
        expected = 148
        actual = self.wrapper["start"].observation(None)[INDEX_OF_shield_2x]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_shield3x(self):
        expected = 212
        actual = self.wrapper["start"].observation(None)[INDEX_OF_shield_3x]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_shieldy_noshields(self):
        expected = 210
        actual = self.wrapper["fewer_shields"].observation(None)[INDEX_OF_shield_y]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_shield1x_noshields(self):
        expected = 0
        actual = self.wrapper["fewer_shields"].observation(None)[INDEX_OF_shield_1x]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_shield2x_noshields(self):
        expected = 0
        actual = self.wrapper["fewer_shields"].observation(None)[INDEX_OF_shield_2x]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_shield3x_noshields(self):
        expected = 0
        actual = self.wrapper["fewer_shields"].observation(None)[INDEX_OF_shield_3x]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ship_laser_pos_x_no_laser(self):
        expected = 0
        actual = self.wrapper["start"].observation(None)[INDEX_OF_ship_laser_pos_x]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ship_laser_pos_y_no_laser(self):
        expected = 0
        actual = self.wrapper["start"].observation(None)[INDEX_OF_ship_laser_pos_y]
        self.assertAlmostEqual(expected, actual)
        pass


if __name__ == "__main__":
    unittest.main()
