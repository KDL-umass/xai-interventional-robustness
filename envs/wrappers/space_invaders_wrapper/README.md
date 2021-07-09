# ToyBox Space Invaders Feature Vector Wrapper

## Setup
```
pip install wheel gym[atari] ctoybox
pip install git+https://github.com/toybox-rs/Toybox
```

## Run the tests
```
python -m unittest test_space_invaders_feature_vec_wrapper
```

## Description of files

* `wrappers.py`: This abstract class uses `gym.ObservationWrapper` to define class of wrapper methods for ToyBox games.
* `space_invaders_feature_vec_wrapper.py`: This class is an instantiation of `FeatureVecWrapper` class from `wrappers.py`. 
    The class to import from here is `SpaceInvadersFeatureVecWrapper`.
* `test_space_invaders_feature_vec_wrapper.py`: This file has tests for `SpaceInvadersFeatureVecWrapper` and can be used as a reference 
    for instantiating the wrapper and intervening on the Toybox Space Invaders environment.

## Feature Considerations

See `space_invaders_feature_vec_wrapper.py` and specifically the `_get_feature_vec` method for details about what the features are. 
Each feature function has a docstring describing the feature it returns, and they are concatenated in a deterministic order.

The `INDEX_OF_*` constants can be used to refer to the indices of specific features directly.

### Forward simulation

Many of the features that require forward simulation of the game to determine the outcome 
are not possible without extreme modification of the underlying rust code. 
For instance, the enemy movement speeds change as the game time progresses, 
so determining if a laser from the agent will hit an enemy is difficult if not impossible to include in the features.

### Range of Objects

Is an object's location one pixel or a range of pixels? We consider an object as taking up a range of pixels; therefore, the shield occupies a range of pixels and the ship also does. However, the json only stores an x and y location for the object. As a result, we (1) identify where in a range of pixels that x and y location is and (2) identify how to compute that range of pixels that the object takes up. 

To carry out (1), we retrieved the x and y location of a shield from the json, and also printed out the current game state in an image using the .get_rgb_frame() function. From the image, we concluded that the x and y location of the shield in the json indicates the top-left corner location of the shield. Therefore, to carry out (2), we computed the x range of the shield to be [x, x+width] where width is the width of the shield and moving to the right is a positive direction. To compute the y range of the shield, we do [y, y+height] where height is the height of the shield and moving to the south is a positive direction. 

We carry out the same steps above for the ship, identifying that the x and y location of the ship is its top-left corner pixel. Therefore, the x range of the ship is [x, x+width]. 

Because the shield and ship could occupy a range of pixels, the ship could be partially under the shield or completely under the shield. Further, the minimum distance to a shield could be the minimum distance for being adjacent to the shield or the minimum distance for being completely under the shield. Therefore, feature_vec_wrapper includes functions to retrieve whether the ship is partially under the shield, completely under the shield, the minimum distance for being adjacent to a shield, the minimum distance for being completely under a shield, and the minimum distance for being completely unshielded (as opposed to partially). 

### Assumptions

The functions that compute whether the ship is partially or completely under the shield and the minimum distance for being adjacent to or completely under a shield, or for being completely unshielded, assume that the ship is alive. They don't check whether the ship is not alive -- we think that this should be ok because the ship should always be alive when the game is happening. Further, these functions assume that the shield range is always the same, even if the enemy destroyed part of it. For example, if the shield belongs to a range [40, 56] and all the pixels for the shield between 46 and 48 are destroyed, the completely_under_shield function will still return True. 
