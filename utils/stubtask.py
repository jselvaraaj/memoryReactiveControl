class StubTask:
    def __init__(self):
        self.id = '1'

    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        # Allow the object to be callable and return itself
        return self

    def __getitem__(self, key):
        # Handle indexing and return itself
        return self

    def __repr__(self):
        return "<NoErrorObject>"

    def __str__(self):
        return "<NoErrorObject>"
