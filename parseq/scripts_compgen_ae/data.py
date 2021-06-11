import os
import random

from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import shelve
import numpy as np
import torch
import qelos as q


from parseq.datasets import Dataset, SCANDatasetLoader, CFQDatasetLoader, autocollate
from parseq.vocab import Vocab

from transformers import AutoTokenizer


class Tokenizer(object):
    def __init__(self, toktype="vanilla", inpvocab:Vocab=None, outvocab:Vocab=None, **kw):
        super(Tokenizer, self).__init__(**kw)
        if toktype.startswith("bert"):
            self.berttok = AutoTokenizer.from_pretrained(toktype)
            assert inpvocab == None
            self._inpvocab = self.berttok.vocab
        else:
            self.berttok = None
            self._inpvocab = inpvocab
        self.outvocab = outvocab

    @property
    def inpvocab(self):
        return self._inpvocab

    @inpvocab.setter
    def inpvocab(self, value):
        if self.berttok is not None:
            pass
        else:
            self._inpvocab = value

    def tokenize(self, inps, outs):
        if self.berttok is not None:
            inptoks = self.berttok.tokenize(inps)
        else:
            inptoks = ["@START@"] + self.get_toks(inps) + ["@END@"]
        outtoks = self.get_toks(outs)
        if self.berttok is not None:
            inptensor = self.berttok.encode(inps, return_tensors="pt")[0]
        else:
            inptensor = self.tensorize_output(inptoks, self._inpvocab)
        ret = {"inps": inps, "outs":outs, "inptoks": inptoks, "outtoks": outtoks,
               "inptensor": inptensor, "outtensor": self.tensorize_output(outtoks, self.outvocab)}
        ret = (ret["inptensor"], ret["outtensor"])
        return ret

    def get_toks(self, x):
        return x.strip().split(" ")

    def tensorize_output(self, x, vocab):
        ret = [vocab[xe] for xe in x]
        ret = torch.tensor(ret)
        return ret


ORDERLESS = {"@WHERE", "@OR", "@AND", "@QUERY", "(@WHERE", "(@OR", "(@AND", "(@QUERY"}


def load_ds(dataset="scan/random", validfrac=0.1, splitseed=42, recompute=False):
    tt = q.ticktock("data")
    tt.tick(f"loading '{dataset}'")
    if dataset.startswith("cfq/") or dataset.startswith("scan/mcd"):
        key = f"{dataset}|tokenizer=vanilla"
        print(f"validfrac is ineffective with dataset '{dataset}'")
    else:
        key = f"{dataset}|validfrac={validfrac}(seed={splitseed})|tokenizer=vanilla"

    shelfname = os.path.basename(__file__) + ".cache.shelve"
    if not recompute:
        tt.tick(f"loading from shelf (key '{key}')")
        with shelve.open(shelfname) as shelf:
            if key not in shelf:
                recompute = True
                tt.tock("couldn't load from shelf")
            else:
                shelved = shelf[key]
                trainex, validex, testex, fldic = shelved["trainex"], shelved["validex"], shelved["testex"], shelved["fldic"]
                inpdic = shelved["inpdic"] if "inpdic" in shelved else None
                trainds, validds, testds = Dataset(trainex), Dataset(validex), Dataset(testex)
                tt.tock("loaded from shelf")

    if recompute:
        tt.tick("loading data")
        splits = dataset.split("/")
        dataset, splits = splits[0], splits[1:]
        split = "/".join(splits)
        if dataset == "scan":
            ds = SCANDatasetLoader().load(split, validfrac=validfrac, seed=splitseed)
        elif dataset == "cfq":
            ds = CFQDatasetLoader().load(split + "/modent")
        else:
            raise Exception(f"Unknown dataset: '{dataset}'")
        tt.tock("loaded data")

        tt.tick("creating tokenizer")
        tokenizer = Tokenizer(toktype="vanilla")
        tt.tock("created tokenizer")

        print(len(ds))

        tt.tick("dictionaries")
        inpdic = Vocab()
        inplens, outlens = [0], []
        fldic = Vocab()
        for x in ds:
            outtoks = tokenizer.get_toks(x[1])
            outlens.append(len(outtoks))
            for tok in outtoks:
                fldic.add_token(tok, seen=x[2] == "train")
            inptoks = tokenizer.get_toks(x[0])
            for tok in inptoks:
                inpdic.add_token(tok, seen=x[2] == "train")
        inpdic.finalize(min_freq=0, top_k=np.infty)
        fldic.finalize(min_freq=0, top_k=np.infty)
        print(
            f"input avg/max length is {np.mean(inplens):.1f}/{max(inplens)}, output avg/max length is {np.mean(outlens):.1f}/{max(outlens)}")
        print(f"output vocabulary size: {len(fldic.D)} at output, {len(inpdic.D)} at input")
        tt.tock()

        tt.tick("tensorizing")
        if tokenizer.berttok is None:
            tokenizer._inpvocab = inpdic
        tokenizer.outvocab = fldic
        trainds = ds.filter(lambda x: x[-1] == "train").map(lambda x: x[:-1]).map(lambda x: tokenizer.tokenize(x[0], x[1])).cache(True)
        validds = ds.filter(lambda x: x[-1] == "valid").map(lambda x: x[:-1]).map(lambda x: tokenizer.tokenize(x[0], x[1])).cache(True)
        testds = ds.filter(lambda x: x[-1] == "test").map(lambda x: x[:-1]).map(lambda x: tokenizer.tokenize(x[0], x[1])).cache(True)
        # ds = ds.map(lambda x: tokenizer.tokenize(x[0], x[1]) + (x[2],)).cache(True)
        tt.tock("tensorized")

        tt.tick("shelving")
        with shelve.open(shelfname) as shelf:
            shelved = {
                "trainex": trainds.examples,
                "validex": validds.examples,
                "testex": testds.examples,
                "fldic": fldic,
                "inpdic": inpdic,
            }
            shelf[key] = shelved
        tt.tock("shelved")

    tt.tock(f"loaded '{dataset}'")
    tt.msg(f"#train={len(trainds)}, #valid={len(validds)}, #test={len(testds)}")

    tt.msg("Overlap of validation with train:")
    overlaps = compute_overlaps(trainds, validds)
    print(json.dumps(overlaps, indent=4))

    tt.msg("Overlap of test with train:")
    overlaps = compute_overlaps(trainds, testds)
    print(json.dumps(overlaps, indent=4))

    return trainds, validds, testds, fldic, inpdic


def compute_overlaps(train, test):
    inp_overlap = []
    out_overlap = []
    both_overlap = []
    traininps, trainouts, trainboths = set(), set(), set()
    for i in tqdm(range(len(train))):
        ex = train[i]
        inpstr = list(ex[0].cpu().numpy())
        inpstr = " ".join([str(exe) for exe in inpstr])
        outstr = list(ex[1].cpu().numpy())
        outstr = " ".join([str(exe) for exe in outstr])
        traininps.add(inpstr)
        trainouts.add(outstr)
        trainboths.add(inpstr+"|"+outstr)

    for i in tqdm(range(len(test))):
        ex = test[i]
        inpstr = list(ex[0].cpu().numpy())
        inpstr = " ".join([str(exe) for exe in inpstr])
        outstr = list(ex[1].cpu().numpy())
        outstr = " ".join([str(exe) for exe in outstr])
        if inpstr in traininps:
            inp_overlap.append(inpstr)
        if outstr in trainouts:
            out_overlap.append(outstr)
        if inpstr + "|" + outstr in trainboths:
            both_overlap.append(inpstr + "|" + outstr)

    ret = {"inps": len(inp_overlap)/len(test),
           "outs": len(out_overlap)/len(test),
           "both": len(both_overlap)/len(test),}
    return ret


def collate_fn(x, pad_value=0, numtokens=5000):
    lens = [len(xe[1]) for xe in x]
    a = list(zip(lens, x))
    a = sorted(a, key=lambda xe: xe[0], reverse=True)
    maxnum = int(numtokens/max(lens))
    b = a[:maxnum]
    b = [be[1] for be in b]
    ret = autocollate(b, pad_value=pad_value)
    return ret


class RandomAugData(object):



def get_dataloaders(dataset="scan/mcd1", augmode="none", batsize=10, validfrac=0.1, recompute=False, auglen=100):
    tt = q.ticktock("data")
    tt.tick("getting dataloaders")
    tt.tick("loading data")
    trainds, validds, testds, fldic, inpdic = load_ds(dataset=dataset,
                                                      validfrac=validfrac,
                                                      recompute=recompute)

    if augmode == "none" or augmode is None:
        auginpdl = None
        augoutdl = None
    elif augmode == "random":
        auginpds = RandomAugData(inpdic, maxlen=auglen)
        augoutds = RandomAugData(fldic, maxlen=auglen)
        auginpdl = DataLoader(auginpds, batch_size=batsize, shuffle=True, collate_fn=autocollate)
        augoutdl = DataLoader(augoutds, batch_size=batsize, shuffle=True, collate_fn=autocollate)

    tt.tock("data loaded")
    tt.msg(f"TRAIN DATA: {len(trainds)}")
    tt.msg(f"DEV DATA: {len(validds)}")
    tt.msg(f"TEST DATA: {len(testds)}")

    tt.tick("dataloaders")
    traindl = DataLoader(trainds, batch_size=batsize, shuffle=True, collate_fn=autocollate)
    validdl = DataLoader(validds, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    testdl = DataLoader(testds, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    tt.tock()
    tt.tock("got dataloaders")

    return traindl, validdl, testdl, auginpdl, augoutdl, fldic, inpdic


def tst_tokenizer():
    tok = Tokenizer("bert-base-uncased")
    print(tok)


def tst_scan_loader():
    trainds, validds, testds, fldic, inpdic = load_ds(dataset="scan/mcd1",
                                                      validfrac=0.1,
                                                      recompute=True)
    trainds, validds, testds, fldic, inpdic = load_ds(dataset="scan/mcd1",
                                                      validfrac=0.1,
                                                      recompute=False)

def tst_data_loader(dataset="scan/mcd1", batsize=10):
    traindl, validdl, testdl, fldic, inpdic = get_dataloaders(dataset, batsize=batsize)


if __name__ == '__main__':
    # tst_tokenizer()
    # tst_scan_loader()
    tst_data_loader()