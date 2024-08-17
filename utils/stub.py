import pyglet


class StubWindow:
    def __init__(self, *args, **kwargs):
        self.width = kwargs.get('width', 600)
        self.height = kwargs.get('height', 400)
        self.isopen = True

    def switch_to(self):
        pass

    def dispatch_events(self):
        pass

    def clear(self):
        pass

    def flip(self):
        pass

    def close(self):
        self.isopen = False

    def on_close(self):
        return False

    def get_buffer_manager(self):
        return self

    def get_color_buffer(self):
        return pyglet.image.get_buffer_manager().get_color_buffer()

    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self
