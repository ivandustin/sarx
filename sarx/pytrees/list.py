class List(list):
    def tree_flatten(self):
        return (self, None)

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(children)
