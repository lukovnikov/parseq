import shelve
from collections import Callable
from copy import deepcopy

import torch
from functools import partial

import os
import tqdm
import random

from parseq.datasets import Dataset
from transformers import T5Model, T5TokenizerFast


class MetaQADatasetLoader(object):
    def __init__(self,
                 p="../../datasets/metaqa/",
                 ):
        super(MetaQADatasetLoader, self).__init__()
        self.p = p

    def load_qa(self, hops="all", kbds=None, tok=None, recompute=False, subset=None, mode="set"):
        """
        :param which:  "all" or "1", or "2" or "3" or "1+2" etc
        :return:
        """
        if hops == "all":
            hops = "1+2+3"
        hops = hops.split("+")

        with shelve.open(os.path.basename(__file__) + ".cache") as s:

            if f"qads-{mode}" not in s or recompute:
                data = []
                whichhops = "1+2+3"
                for numhops in whichhops.split("+"):
                    print(f"loading {numhops}-hop")
                    path = os.path.join(self.p, f"{numhops}-hop", "vanilla")
                    # load train data
                    print("loading train")
                    with open(os.path.join(path, "qa_train.txt"), encoding="utf-8") as f:
                        for line in tqdm.tqdm(f.readlines()):
                            question, answers = self.process_qa_line(line)
                            data.append((question, answers, numhops, "train"))
                    print("loading test")
                    with open(os.path.join(path, "qa_test.txt"), encoding="utf-8") as f:
                        for line in tqdm.tqdm(f.readlines()):
                            question, answers = self.process_qa_line(line)
                            data.append((question, answers, numhops, "test"))
                    print("loading valid")
                    with open(os.path.join(path, "qa_dev.txt"), encoding="utf-8") as f:
                        for line in tqdm.tqdm(f.readlines()):
                            question, answers = self.process_qa_line(line)
                            data.append((question, answers, numhops, "valid"))

                assert tok is not None and kbds is not None
                ds = QADataset(data, tok=tok, kbds=kbds)
                s[f"qads-{mode}"] = ds
                print("shelved")
            print("loading from shelve")
            ds = s[f"qads-{mode}"]
            _ds = ds.filter(lambda x: str(x[-2]) in hops)

        evaltrainds = _ds["train"]
        random.shuffle(evaltrainds._examples)
        evaltrainds = Dataset(evaltrainds.examples[:len(evaltrainds)//10])

        trainds = _ds["train"]
        validds = _ds["valid"]
        testds = _ds["test"]
        if subset is not None:
            print(f"using only subset: {subset}")
            random.shuffle(trainds._examples)
            trainds = Dataset(trainds.examples[:subset])
            evaltrainds = trainds
            random.shuffle(validds._examples)
            validds = Dataset(validds.examples[:subset])
            random.shuffle(testds._examples)
            testds = Dataset(testds.examples[:subset])

        trainds = trainds.map(partial(ds.item_mapper, return_mode=mode))
        validds = validds.map(partial(ds.item_mapper, return_mode=mode))
        testds = testds.map(partial(ds.item_mapper, return_mode=mode))

        evaltrainds = evaltrainds.map(partial(ds.item_mapper, return_mode=mode))
        return trainds, evaltrainds, validds, testds


    # def qa_mapper(self, x, tok=None, kbds=None):
    #     question, answers, hops, split = x
    #     question_tokenized = tok(question, return_tensors="pt")["input_ids"]
    #     answerset = set()
    #     for answer in answers:
    #         answerset.add(kbds.elemdic[answer])
    #     return question_tokenized, answerset, hops, split

    def process_qa_line(self, line):
        line = line.strip()
        question, answers = line.split("\t")
        answers = answers.split("|")
        question = question.replace("[", "").replace("]", "")
        question = question + " [ANS]"
        question = question.replace("\s+", " ").strip()
        return question, answers

    def load_kb(self, tok, validfrac=0.1, recompute=False, subset=None, mode="set"):
        with shelve.open(os.path.basename(__file__)+".cache") as s:
            if f"kbds-{mode}" not in s or recompute:
                print("loading KB dataset")
                triples = []
                with open(os.path.join(self.p, "kb.txt"), encoding="utf-8") as f:
                    for line in tqdm.tqdm(f.readlines()):
                        newtriples = self.process_kb_line(line)
                        triples.extend(newtriples)
                _ds = KBDataset(triples, tok)
                s[f"kbds-{mode}"] = _ds
                print("shelved")
            print("loading from shelve")
            _ds = s[f"kbds-{mode}"]
        # INFO: validate only on portion of train data, optionally implement splitting
        # random.shuffle(_ds.examples)
        # indexes = list(range(len(_ds)))
        # random.shuffle(indexes)
        # validindexes = set(indexes[:round(validfrac * len(_ds))])
        # _ds[lambda x: ]

        if subset is not None:
            print(f"using only subset: {subset}")
            _ds = deepcopy(_ds)
            random.shuffle(_ds._examples)
            _ds._examples = _ds._examples[:subset]

        trainds = _ds.map(partial(_ds.item_mapper, return_mode=mode))
        trainvalidds = _ds.map(partial(_ds.item_mapper, return_mode=mode))
        return trainds, trainvalidds

    def process_kb_line(self, line:str):
        line = line.strip()
        subj, rel, obj = line.split("|")
        rel = rel.replace("_", " ")
        ret = [(subj, rel, obj)]
        # ret = [(subj, f"is {rel}", obj), (obj, f"is {rel} of", subj)]
        return ret


class QADataset(Dataset):
    getitemtype = "seq"
    ansmaxlen = 200
    maxitemnr = 100
    numsamples = 1

    def __init__(self, examples, tok=None, kbds=None):
        super(QADataset, self).__init__()
        self.entities = kbds.entities
        self.elemdic = kbds.elemdic
        self.elems_pretokenized = kbds.elems_pretokenized
        mappedexamples = [self.init_mapper(example, tok) for example in tqdm.tqdm(examples)]
        self._examples = mappedexamples
        self.tok = tok

        print("copying dictionary")
        self.D = self.tok.vocab
        self.D = {k: v for k, v in self.D.items()}

    def init_mapper(self, x, tok):
        question, answers, hops, split = x
        question_tokenized = tok(question, return_tensors="pt")["input_ids"][0]
        # answerstr = " [SEPITEM] ".join(answers[:min(len(answers), self.maxans)])
        # answertensor = tok(answerstr, return_tensors="pt")["input_ids"][0]
        answerset = set(answers)
        return question_tokenized, answerset, hops, split

    def item_mapper(self, example, return_mode=None):
        return_mode = return_mode if return_mode is not None else self.getitemtype
        question_pretokenized, values, hops = example

        if return_mode == "seq":
            retids = [self.elemdic[value] for value in sorted(values)]
            # retid = retids[0]
            # # retid = random.choice(retids)
            # rettensor = self.elems_pretokenized[retid]

            rettensor = torch.ones(self.ansmaxlen, dtype=torch.long) * -1000
            rettensor[0] = self.D["[BOS]"]
            pos = 1
            for retid in retids:
                elemtensor = self.elems_pretokenized[retid]
                newpos = pos + elemtensor.size(0) - 1
                if newpos + 1 < self.ansmaxlen:
                    rettensor[pos:newpos] = elemtensor[0:newpos-pos]
                    pos = newpos
                else:
                    newpos = pos
            rettensor[newpos] = 1       # EOS
            rettensor = rettensor[:newpos+1]

            if not torch.all(rettensor >= 0):
                assert torch.all(rettensor >= 0)

            return question_pretokenized, rettensor

        elif return_mode == "seqset":   # set of sequences
            values = sorted(values)
            if len(values) > self.maxitemnr:
                values = values[:self.maxitemnr]
            rets = [(self.D[f"[ITEM-{i}]"], self.elemdic[value]) for i, value in enumerate(values)]
            # lastret = rets[-1]
            # rets[-1] = (lastret)
            # rets.append((self.D[f"[ITEM-{len(rets)}]"], None))
            choices = random.sample(range(len(rets)), min(len(rets), self.numsamples))
            # choices = random.sample(rets, min(len(rets), self.numsamples))
            rettensors = []
            for choice in choices:
                itemnr, retid = rets[choice]
                # if retid is None:
                #     rettensor = torch.tensor([itemnr, self.D["[ENDOFSET]"], self.D[self.tok.eos_token]], dtype=torch.long)
                # else:
                last = choice == len(rets) - 1
                rettensor_ = self.elems_pretokenized[retid]
                if last:
                    rettensor = torch.zeros(rettensor_.size(0) + 2, device=rettensor_.device, dtype=rettensor_.dtype)
                    rettensor[0] = itemnr
                    rettensor[1:-2] = rettensor_[:-1]
                    rettensor[-2] = self.D["[ENDOFSET]"]    # end of set
                    rettensor[-1] = rettensor_[-1]          # EOS token
                else:
                    rettensor = torch.zeros(rettensor_.size(0)+1, device=rettensor_.device, dtype=rettensor_.dtype)
                    rettensor[0] = itemnr
                    rettensor[1:] = rettensor_[:]
                rettensors.append(rettensor)

            return [question_pretokenized]*len(choices), rettensors


class KBDataset(Dataset):
    getitemtype = "pair"       # "pair" or "set"
    ansmaxlen = 200
    maxitemnr = 100
    numsamples = 2

    def __init__(self, triples, tok=None):
        super(KBDataset, self).__init__()
        # self.tripleset = set(triples)
        entities, rels = elems_from_triples(triples)
        self.entities = entities
        self.rels = rels
        allelems = self.rels + self.entities
        self.elemdic = {k: v for k, v in zip(allelems, range(len(allelems)))}
        # self.relids = [self.elemdic[rel] for rel in self.rels]
        # self.entids = [self.elemdic[ent] for ent in self.entities]

        self.tok = tok
        print("Pretokenizing elements")
        self.elems_pretokenized = []
        for elem in tqdm.tqdm(self.rels):
            self.elems_pretokenized.append(self.tok("[REL] " + elem, return_tensors="pt")["input_ids"][0])
        for elem in tqdm.tqdm(self.entities):
            self.elems_pretokenized.append(self.tok("[ENT] " + elem, return_tensors="pt")["input_ids"][0])

        # group triples
        print("Grouping triples")
        tripledict = {}
        for triple in tqdm.tqdm(triples):
            _triples = [((triple[0], triple[1], "[ANS]"), 2),
                        # ((triple[0], "[ANS]", triple[1]), 1),
                        (("[ANS]", triple[1], triple[2]), 0)]
            for _triple in _triples:
                if not _triple in tripledict:
                    tripledict[_triple] = set()
                tripledict[_triple].add(triple[_triple[1]])

        outtriples = []
        print("Creating triple tensors")
        for _triple in tqdm.tqdm(tripledict.keys()):
            triple, _ = _triple
            triplestr = f"{triple[0]} [SEP1] {triple[1]} [SEP2] {triple[2]}"
            tripletensor = self.tok(triplestr, return_tensors="pt")["input_ids"][0]
            outtriples.append((tripletensor, _triple[1], tripledict[_triple]))

        self._examples = outtriples

        print("copying dictionary")
        self.D = self.tok.vocab
        self.D = {k: v for k, v in self.D.items()}

    def item_mapper(self, example, return_mode=None):
        return_mode = return_mode if return_mode is not None else self.getitemtype
        triple_pretokenized, negwhich, values = example

        if return_mode == "seq":
            retids = [self.elemdic[value] for value in sorted(values)]
            # retid = random.choice(retids)
            # rettensor = self.elems_pretokenized[retid]
            rettensor = torch.ones(self.ansmaxlen, dtype=torch.long) * -1000
            rettensor[0] = self.D["[BOS]"]
            pos = 1
            for retid in retids:
                elemtensor = self.elems_pretokenized[retid]
                newpos = pos + elemtensor.size(0) - 1
                if newpos + 1 < self.ansmaxlen:
                    rettensor[pos:newpos] = elemtensor[0:newpos - pos]
                    pos = newpos
                else:
                    newpos = pos
            rettensor[newpos] = 1  # EOS
            rettensor = rettensor[:newpos + 1]
            return triple_pretokenized, rettensor

        elif return_mode == "seqset":   # set of sequences
            values = sorted(values)
            if len(values) > self.maxitemnr:
                values = values[:self.maxitemnr]
            rets = [(self.D[f"[ITEM-{i}]"], self.elemdic[value]) for i, value in enumerate(values)]
            rets.append((self.D[f"[ITEM-{len(rets)}]"], None))
            choices = random.sample(rets, min(len(rets), self.numsamples))
            rettensors = []
            for choice in choices:
                itemnr, retid = choice
                if retid is None:
                    rettensor = torch.tensor([itemnr, self.D["[ENDOFSET]"], self.D[self.tok.eos_token]], dtype=torch.long)
                else:
                    rettensor_ = self.elems_pretokenized[retid]
                    rettensor = torch.zeros(rettensor_.size(0)+1, device=rettensor_.device, dtype=rettensor_.dtype)
                    rettensor[0] = itemnr
                    rettensor[1:] = rettensor_[:]
                rettensors.append(rettensor)

            return [triple_pretokenized]*len(choices), rettensors

            # if len(retids) > 20:
            #     retids = retids[:20]
            # rettensor = []
            # for retid in retids:
            #     rettensor_i = self.elems_pretokenized[retid]
            #     # rettensor_i[-1] = self.tok.vocab["[SEPITEM]"]
            #     rettensor.append(rettensor_i)
            # rettensor = torch.cat(rettensor, 0)
            # rettensor[rettensor == self.tok.vocab["</s>"]] = self.tok.vocab["[SEPITEM]"]
            # rettensor[-1] = self.tok.vocab["</s>"]
            # return triple_pretokenized, rettensor

        #
        #
        # triple = super(KBDataset, self).__getitem__(item)
        # subj, rel, obj = triple
        # if random.random() > 0.8:
        #     posans = rel
        #     negans = rel
        #     while (subj, negans, obj) in self.tripleset:
        #         negans = random.choice(self.rels)
        #     whichreplaced = 1
        # else:
        #     if random.random() > 0.5:
        #         posans = obj
        #         negans = obj
        #         while (subj, rel, negans) in self.tripleset:
        #             negans = random.choice(self.entities)
        #         whichreplaced = 2
        #     else:
        #         posans = subj
        #         negans = subj
        #         while (negans, rel, obj) in self.tripleset:
        #             negans = random.choice(self.entities)
        #         whichreplaced = 0
        #
        # triple = [subj, rel, obj]
        # triple[whichreplaced] = "[ANS]"
        #
        # triplestr = f"{triple[0]} [SEP1] {triple[1]} [SEP2] {triple[2]}"
        # return triplestr, posans, negans
    #
    #
    # def __getitem__(self, item):
    #     triple = super(KBDataset, self).__getitem__(item)
    #     postriple = triple
    #     subj, rel, obj = triple
    #     if random.random() > 0.8:
    #         randrel = rel
    #         while (subj, randrel, obj) in self.tripleset:
    #             randrel = random.choice(self.rels)
    #         negtriple = (subj, randrel, obj)
    #         whichreplaced = 2
    #     else:
    #         if random.random() > 0.5:
    #             randobj = obj
    #             while (subj, rel, randobj) in self.tripleset:
    #                 randobj = random.choice(self.entities)
    #             negtriple = (subj, rel, randobj)
    #             whichreplaced = 3
    #         else:
    #             randsubj = subj
    #             while (randsubj, rel, obj) in self.tripleset:
    #                 randsubj = random.choice(self.entities)
    #             negtriple = (randsubj, rel, obj)
    #             whichreplaced = 1
    #
    #     triplestr = f"[ANS]"
    #     posstr = f"{postriple[0]} [SEP1] {postriple[1]} [SEP2] {postriple[2]}"
    #     negstr = f"{negtriple[0]} [SEP1] {negtriple[1]} [SEP2] {negtriple[2]}"
    #     return posstr, negstr, whichreplaced


def elems_from_triples(triples):
    entities = set()
    rels = set()
    for triple in triples:
        entities.add(triple[0])
        entities.add(triple[2])
        rels.add(triple[1])
    entities = sorted(list(entities))
    rels = sorted(list(rels))
    return entities, rels


def try_metaqa(recompute = True):
    print("loading tokenizer")
    extra_tokens = ["[SEP1]", "[SEP2]", "[ANS]", "[ENT]", "[REL]", "[SEPITEM]", "[BOS]", "[ENDOFSET]"] # + [f"extra_id_{i}" for i in range(0)]
    extra_tokens = extra_tokens + [f"[ITEM-{i}]" for i in range(1000)]
    tok = T5TokenizerFast.from_pretrained("google/t5-v1_1-base", additional_special_tokens=extra_tokens, extra_ids=0)
    print(len(tok.vocab))

    kbds, validkbds = MetaQADatasetLoader().load_kb(tok, recompute=recompute, mode="seqset")
    # kbds = kbds.map(kbds.item_mapper)
    print(len(kbds))
    print("\n".join([str(kbds[i]) for i in range(15)]))

    trainqads, evaltrainds, validqads, testqads = MetaQADatasetLoader().load_qa("1", kbds=kbds.baseds, tok=tok, recompute=recompute,
                                                    mode="seqset")
    print("\n".join([str(trainqads[i]) for i in range(15)]))
    print("\n".join([str(validqads[i]) for i in range(15)]))
    # tok.add_tokens(["[SEP1]", "[SEP2]"])

    t5 = T5Model.from_pretrained("google/t5-v1_1-base")
    emb = t5.encoder.embed_tokens
    # from parseq.scripts_qa.bert import adapt_embeddings
    from parseq.scripts_cbqa.adapter_t5 import adapt_embeddings
    emb = adapt_embeddings(emb, tok)
    #
    tokids = tok("[SEP1] [SEP2] <unk> </s> <pad>", return_tensors="pt")["input_ids"]
    embs = emb(tokids)
    t5(tokids)


if __name__ == '__main__':
    try_metaqa()
