import dataclasses


class Items:
    """
    Similar to dict_items view.
    """

    def __init__(self, dc):
        self.dc = dc

    def __iter__(self):
        for field in dataclasses.fields(self.dc):
            yield field.name, getattr(self.dc, field.name)
