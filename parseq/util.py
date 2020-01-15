
class DatasetSplitProxy(object):
    def __init__(self, data, **kw):
        super(DatasetSplitProxy, self).__init__(**kw)
        self.data = data

    def __getitem__(self, item):
        return self.data[item].make_copy()

    def __len__(self):
        return len(self.data)