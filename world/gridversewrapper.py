from gym_gridverse.gym import GymEnvironment


class GridVerseWrapper(GymEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def render(self):
        rgb_arrays = super().render()
        return rgb_arrays[1] if len(rgb_arrays) == 2 else rgb_arrays[0]
