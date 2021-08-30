from itertools import chain

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset


class DatasetSplitProxy(object):
    def __init__(self, data, **kw):
        super(DatasetSplitProxy, self).__init__(**kw)
        self.data = data

    def __getitem__(self, item):
        return self.data[item].make_copy()

    def __len__(self):
        return len(self.data)


class CatDataLoader(object):
    def __init__(self, *dls, **kw):
        super(CatDataLoader, self).__init__(**kw)
        self.dls = dls
        self.dliters = []

    def __iter__(self):
        self.dliters = []
        for dl in self.dls:
            self.dliters.append(iter(dl))
        return self

    def __next__(self):
        nexts = tuple()
        for dliter in self.dliters:
            nexts = nexts + next(dliter)
        return nexts

    def __len__(self):
        return len(self.dls[0])


class CatDataset(Dataset):

    @classmethod
    def create(cls, *dss, **kw):
        map_dss, iter_dss = [], []
        for ds in dss:
            if isinstance(ds, IterableDataset):
                iter_dss.append(ds)
            else:
                map_dss.append(ds)
        if len(map_dss) > 0:
            # assert equal lengths of map-style datasets
            assert all([len(ds) == len(map_dss[0]) for ds in map_dss])
            return CatDataset(*dss, **kw)
        else:
            raise NotImplementedError()
            return IterableCatDataset(*dss, **kw)

    def __init__(self, *dss, **kw):
        super(CatDataset, self).__init__(**kw)
        self.dss = dss
        map_dss, iter_dss = [], []
        for ds in dss:
            if isinstance(ds, IterableDataset):
                iter_dss.append(ds)
            else:
                map_dss.append(ds)
        # assert equal lengths of map-style datasets
        assert len(map_dss) > 0
        assert all([len(ds) == len(map_dss[0]) for ds in map_dss])
        self._length = len(map_dss[0])

        self.reset_iters()

    def __len__(self):
        return self._length

    def reset_iters(self):
        self._iter_ds_iters = None

    def init_iters(self):
        self._iter_ds_iters = [iter(ds) if isinstance(ds, IterableDataset) else None for ds in self.dss]

    def __getitem__(self, item):
        tuples = []
        if self._iter_ds_iters is None:
            self.init_iters()
        for i, ds in enumerate(self.dss):
            if isinstance(ds, IterableDataset):
                x = next(self._iter_ds_iters[i])
            else:
                x = ds[item]
            tuples.append(x)
        ret = tuple(chain(*tuples))
        return ret



