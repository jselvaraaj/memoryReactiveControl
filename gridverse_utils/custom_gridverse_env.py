from typing import Optional, Set

import numpy.random as rnd
from gym_gridverse.action import Action
from gym_gridverse.agent import Agent
from gym_gridverse.design import (
    draw_area,
    draw_line_horizontal,
    draw_line_vertical,
)
from gym_gridverse.envs.reset_functions import reset_function_registry
from gym_gridverse.envs.reward_functions import reward_function_registry
from gym_gridverse.geometry import Orientation, Position, Shape
from gym_gridverse.grid import Grid
from gym_gridverse.grid_object import (
    Color,
    Exit,
    Floor,
    Wall,
)
from gym_gridverse.rng import choices, get_gv_rng_if_none
from gym_gridverse.state import State


def memory_without_beacon(
        shape: Shape,
        colors: Set[Color],
        *,
        rng: Optional[rnd.Generator] = None,
) -> State:
    if shape.height < 5:
        raise ValueError(f'height ({shape.height}) must be >= 5')
    if shape.width < 5 or shape.width % 2 == 0:
        raise ValueError(f'width ({shape.width}) must be odd and >= 5')
    if Color.NONE in colors:
        raise ValueError(f'colors ({colors}) must not include Colors.NONE')
    if len(colors) < 2:
        raise ValueError(f'colors ({colors}) must have at least 2 colors')

    rng = get_gv_rng_if_none(rng)

    grid = Grid.from_shape((shape.height, shape.width))
    draw_area(grid, grid.area, Wall, fill=True)
    draw_line_horizontal(grid, 1, range(2, shape.width - 2), Floor)
    draw_line_horizontal(
        grid, shape.height - 2, range(2, shape.width - 2), Floor
    )
    draw_line_vertical(
        grid, range(2, shape.height - 2), shape.width // 2, Floor
    )

    color_good, color_bad = choices(rng, list(colors), size=2, replace=False)
    x_exit_good, x_exit_bad = choices(
        rng, [1, shape.width - 2], size=2, replace=False
    )
    grid[1, x_exit_good] = Exit(color_good)
    grid[1, x_exit_bad] = Exit(color_bad)
    grid[shape.height - 2, 1] = Floor()
    grid[shape.height - 2, shape.width - 2] = Floor()

    agent_position = Position(shape.height // 2, shape.width // 2)
    agent_orientation = Orientation.F
    agent = Agent(agent_position, agent_orientation)

    return State(grid, agent)


def reach_exit_random(
        state: State,
        action: Action,
        next_state: State,
        *,
        reward_good: float = 1.0,
        reward_bad: float = -1.0,
        rng: Optional[rnd.Generator] = None,
) -> float:
    rng = get_gv_rng_if_none(rng)
    rand_num, = choices(rng, [0, 1], size=1, replace=False)

    agent_grid_object = next_state.grid[next_state.agent.position]

    return (
        (reward_good if rand_num == 0 else reward_bad)
        if isinstance(agent_grid_object, Exit)
        else 0.0
    )


def register_custom_functions():
    reset_function_registry.register(memory_without_beacon)
    reward_function_registry.register(reach_exit_random)
