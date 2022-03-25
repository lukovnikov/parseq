from collections import Callable

from functools import partial

import os
import tqdm
import random

from parseq.datasets import Dataset
from transformers import BertTokenizer, BertModel, AutoTokenizer, BertTokenizerFast


class MetaQADatasetLoader(object):
    def __init__(self,
                 p="../../datasets/metaqa/",
                 ):
        super(MetaQADatasetLoader, self).__init__()
        self.p = p

    def load_qa(self, hops="all", kbds=None, tok=None):
        """
        :param which:  "all" or "1", or "2" or "3" or "1+2" etc
        :return:
        """
        if hops == "all":
            hops = "1+2+3"
        whichhops = hops.split("+")

        data = []
        for numhops in whichhops:
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

        if tok is not None and kbds is not None:
            ds = QADataset(data, tok=tok, kbds=kbds)
            # ds = Dataset(data)
            # ds = ds.map(partial(self.qa_mapper, tok=tok, kbds=kbds)).cache(compute_now=True, showprogress=True)
        else:
            raise NotImplementedError()
            ds = Dataset(data)
        return ds

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

    def load_kb(self, tok):
        print("loading KB dataset")
        triples = []
        with open(os.path.join(self.p, "kb.txt"), encoding="utf-8") as f:
            for line in tqdm.tqdm(f.readlines()):
                newtriples = self.process_kb_line(line)
                triples.extend(newtriples)
        ds = KBDataset(triples, tok)
        return ds

    def process_kb_line(self, line:str):
        line = line.strip()
        subj, rel, obj = line.split("|")
        rel = rel.replace("_", " ")
        ret = [(subj, rel, obj)]
        # ret = [(subj, f"is {rel}", obj), (obj, f"is {rel} of", subj)]
        return ret


class QADataset(Dataset):
    getitemtype = "pair"       # "pair" or "positives"
    def __init__(self, examples, tok=None, kbds=None):
        super(QADataset, self).__init__()
        self.entities = kbds.entities
        self.elemdic = kbds.elemdic
        self.elems_pretokenized = kbds.elems_pretokenized
        mappedexamples = [self.mapper(example, tok) for example in tqdm.tqdm(examples)]
        self._examples = mappedexamples

    def mapper(self, x, tok):
        question, answers, hops, split = x
        question_tokenized = tok(question, return_tensors="pt")["input_ids"]
        answerset = set(answers)
        return question_tokenized, answerset, hops, split

    def __getitem__(self, item):
        if isinstance(item, (Callable, tuple, dict)):
            ret = self.filter(item)
            return ret
        elif isinstance(item, str):     # interpreted as split, then last column is assumed to be the split
            ret = self.filter(lambda x: x[-1] == item).map(lambda x: x[:-1])
            return ret
        else:
            if isinstance(item, slice):
                rets = []
                items = list(range(item.stop)[item])
                for iteme in items:
                    rets.append(self.__getitem__(iteme))
                ret = zip(*rets)
                return ret
            else:
                question_pretokenized, values, hops, split = self._examples[item]

                if self.getitemtype == "pair":
                    posans = random.choice(list(values))
                    negans = random.choice(self.entities)
                    while negans in values:
                        negans = random.choice(self.entities)

                    posans_pretokenized = self.elems_pretokenized[self.elemdic[posans]]
                    negans_pretokenized = self.elems_pretokenized[self.elemdic[negans]]
                    return question_pretokenized, posans_pretokenized, negans_pretokenized

                elif self.getitemtype == "positives":
                    valids = set([self.elemdic[value] for value in values])
                    return question_pretokenized, valids


class KBDataset(Dataset):
    getitemtype = "pair"       # "pair" or "positives"

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
            self.elems_pretokenized.append(self.tok("[REL] " + elem, return_tensors="pt")["input_ids"])
        for elem in tqdm.tqdm(self.entities):
            self.elems_pretokenized.append(self.tok("[ENT] " + elem, return_tensors="pt")["input_ids"])

        # group triples
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
        for triple in tqdm.tqdm(tripledict.keys()):
            outtriples.append((triple[0], triple[1], tripledict[triple]))

        print("Pretokenizing triples")
        self.triples_pretokenized = []
        for _triple in tqdm.tqdm(outtriples):
            triple, negwhich, values = _triple
            triplestr = f"{triple[0]} [SEP1] {triple[1]} [SEP2] {triple[2]}"
            self.triples_pretokenized.append(self.tok(triplestr, return_tensors="pt")["input_ids"])

        self._examples = outtriples

    def __getitem__(self, item):
        if isinstance(item, (Callable, tuple, dict)):
            ret = self.filter(item)
            return ret
        elif isinstance(item, str):     # interpreted as split, then last column is assumed to be the split
            ret = self.filter(lambda x: x[-1] == item).map(lambda x: x[:-1])
            return ret
        else:
            if isinstance(item, slice):
                rets = []
                items = list(range(item.stop)[item])
                for iteme in items:
                    rets.append(self.__getitem__(iteme))
                ret = zip(*rets)
                return ret
            else:
                triple, negwhich, values = self._examples[item]
                triple_pretokenized = self.triples_pretokenized[item]

                if self.getitemtype == "pair":
                    posans = random.choice(list(values))
                    if negwhich == 1:
                        choices = list(set(self.rels) - values)
                        negans = random.choice(choices)
                    else:
                        negans = random.choice(self.entities)
                        while negans in values:
                            negans = random.choice(self.entities)

                    posans_pretokenized = self.elems_pretokenized[self.elemdic[posans]]
                    negans_pretokenized = self.elems_pretokenized[self.elemdic[negans]]
                    return triple_pretokenized, posans_pretokenized, negans_pretokenized

                elif self.getitemtype == "positives":
                    valids = [self.elemdic[value] for value in values]
                    return triple_pretokenized, valids
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


def try_metaqa():
    print("loading tokenizer")
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased", additional_special_tokens=["[SEP1]", "[SEP2]", "[ANS]", "[ENT]", "[REL]"])
    print(len(tok.vocab))

    kbds = MetaQADatasetLoader().load_kb(tok)
    print(len(kbds))
    print(kbds[:15])

    qads = MetaQADatasetLoader().load_qa("1", kbds=kbds, tok=tok)
    print(qads[:15])
    # tok.add_tokens(["[SEP1]", "[SEP2]"])

    print(tok.tokenize("zelensky [SEP1] president [SEP2] ukraine"))
    print(tok.__call__("zelensky [SEP1] president [SEP2] ukraine"))
    print(tok("zelensky [SEP1] president [SEP2] ukraine"))

    bert = BertModel.from_pretrained("bert-base-uncased")
    bertemb = bert.embeddings
    from parseq.scripts_qa.bert import adapt_embeddings
    bertemb = adapt_embeddings(bertemb, tok)

    tokids = tok("[SEP1] [SEP2] [SEP] [UNK]", return_tensors="pt")["input_ids"][None, :]
    embs = bertemb(tokids)

    ds = MetaQADatasetLoader().load_qa("1+2")
    print(len(ds))
    print(ds[:5])



if __name__ == '__main__':
    try_metaqa()
