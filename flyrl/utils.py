import functools
import operator
from typing import Tuple
from flyrl.aircraft import cessna172P, a320, f15
from typing import Dict, Iterable
import numpy as np


class AttributeFormatter(object):
    """
    Replaces characters that would be illegal in an attribute name

    Used through its static method, translate()
    """
    ILLEGAL_CHARS = '\-/.'
    TRANSLATE_TO = '_' * len(ILLEGAL_CHARS)
    TRANSLATION_TABLE = str.maketrans(ILLEGAL_CHARS, TRANSLATE_TO)

    @staticmethod
    def translate(string: str):
        return string.translate(AttributeFormatter.TRANSLATION_TABLE)


def get_env_id(task_type, aircraft, shaping, enable_flightgear) -> str:
    """
    Creates an env ID from the environment's components

    :param task_type: Task class, the environment's task
    :param aircraft: Aircraft namedtuple, the aircraft to be flown
    :param shaping: HeadingControlTask.Shaping enum, the reward shaping setting
    :param enable_flightgear: True if FlightGear simulator is enabled for visualisation else False
     """
    if enable_flightgear:
        fg_setting = 'FG'
    else:
        fg_setting = 'NoFG'
    return f'JSBSim-{task_type.__name__}-{aircraft.name}-{shaping}-{fg_setting}-v0'


def get_env_id_kwargs_map() -> Dict[str, Tuple]:
    """ Returns all environment IDs mapped to tuple of (task, aircraft, shaping, flightgear) """
    # lazy import to avoid circular dependencies
    from flyrl.tasks import Shaping, HeadingControlTask, TurnHeadingControlTask

    map = {}
    for task_type in (HeadingControlTask, TurnHeadingControlTask):
        for plane in (cessna172P, a320, f15):
            for shaping in (Shaping.STANDARD, Shaping.EXTRA, Shaping.EXTRA_SEQUENTIAL):
                for enable_flightgear in (True, False):
                    id = get_env_id(task_type, plane, shaping, enable_flightgear)
                    assert id not in map
                    map[id] = (task_type, plane, shaping, enable_flightgear)
    return map


def product(iterable: Iterable):
    """
    Multiplies all elements of iterable and returns result

    ATTRIBUTION: code provided by Raymond Hettinger on SO
    https://stackoverflow.com/questions/595374/whats-the-function-like-sum-but-for-multiplication-product
    """
    return functools.reduce(operator.mul, iterable, 1)


def reduce_reflex_angle_deg(angle: float) -> float:
    """ Given an angle in degrees, normalises in [-179, 180] """
    # ATTRIBUTION: solution from James Polk on SO,
    # https://stackoverflow.com/questions/2320986/easy-way-to-keeping-angles-between-179-and-180-degrees#
    new_angle = angle % 360
    if new_angle > 180:
        new_angle -= 360
    return new_angle

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def mt2ft(mt):
    return mt*3.28

def ft2mt(ft):
    return ft/3.28