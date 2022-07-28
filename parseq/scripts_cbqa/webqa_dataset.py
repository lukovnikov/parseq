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


class WebQADatasetLoader(object):
    def __init__(self,
                 p="../../datasets/webquestions/",
                 ):
        super(WebQADatasetLoader, self).__init__()
        self.p = p

    def load_qa(self, tok=None, recompute=False, mode="set"):
        """
        :param which:  "all" or "1", or "2" or "3" or "1+2" etc
        :return:
        """
        with shelve.open(os.path.basename(__file__) + ".cache") as s:

            if f"qads-{mode}" not in s or recompute:
                data = []
                # load train data
                print("loading train")
                with open(os.path.join(self.p, "webq-train.csv"), encoding="utf-8") as f:
                    for line in tqdm.tqdm(f.readlines()):
                        question, answers = self.process_qa_line(line)
                        data.append((question, answers, "train"))
                print("loading test")
                with open(os.path.join(self.p, "webq-test.csv"), encoding="utf-8") as f:
                    for line in tqdm.tqdm(f.readlines()):
                        question, answers = self.process_qa_line(line)
                        data.append((question, answers, "test"))
                print("loading valid")
                with open(os.path.join(self.p, "webq-dev.csv"), encoding="utf-8") as f:
                    for line in tqdm.tqdm(f.readlines()):
                        question, answers = self.process_qa_line(line)
                        data.append((question, answers, "valid"))

                assert tok is not None
                ds = QADataset(data, tok=tok)
                s[f"qads-{mode}"] = ds
                print("shelved")
            print("loading from shelve")
            ds = s[f"qads-{mode}"]
            # _ds = ds.filter(lambda x: str(x[-2]) in hops)

        evaltrainds = ds["train"]
        random.shuffle(evaltrainds._examples)
        evaltrainds = Dataset(evaltrainds.examples[:len(evaltrainds)//10])

        trainds = ds["train"]
        validds = Dataset(ds["valid"].examples)
        testds = Dataset(ds["test"].examples)

        trainds = trainds.map(partial(ds.item_mapper, return_mode=mode))
        validds = validds.map(partial(ds.item_mapper, return_mode=mode))
        testds = testds.map(partial(ds.item_mapper, return_mode=mode))

        evaltrainds = evaltrainds.map(partial(ds.item_mapper, return_mode=mode))
        return trainds, evaltrainds, validds, testds

    def process_qa_line(self, line):
        line = line.strip()
        question, answers = line.split("\t")
        if answers[0] == '"':
            assert answers[-1] == '"'
            answers = answers[1:-1]
            answers = answers.replace('""', '"')
        answers = eval(answers)
        answers = [a.strip().lower() for a in answers]      # LOWER CASING !!
        question = question + " [ANS]"
        question = question.replace("\s+", " ").strip()
        return question, answers


class QADataset(Dataset):
    getitemtype = "seq"
    ansmaxlen = 200
    maxitemnr = 100
    numsamples = 1
    maxanswerlen = 30

    def __init__(self, examples, tok=None):
        super(QADataset, self).__init__()

        longestanswer = ""
        maxnumans = 0
        mappedexamples = []
        for example in tqdm.tqdm(examples):
            question, answers, split = example
            question_tokenized = tok(question, return_tensors="pt")["input_ids"][0]                 # tokenize question
            answers = sorted(list(answers), key=lambda x: x)
            _answers = [tok("[ENT] " + answer, return_tensors="pt", max_length=self.maxanswerlen)["input_ids"][0] for answer in answers]      # tokenize answers
            mappedexamples.append((question_tokenized, _answers, split))
            numans = len(answers)
            maxnumans = max(maxnumans, numans)
            for answer, _answer in zip(answers, _answers):
                if len(answer) > len(longestanswer):
                    longestanswer = answer
                    print("longer answer found: ", answer, _answer)

        print("maxnumans: ", maxnumans)

        self._examples = mappedexamples
        self.tok = tok

        print("copying dictionary")
        self.D = self.tok.vocab
        self.D = {k: v for k, v in self.D.items()}

    def item_mapper(self, example, return_mode=None):
        return_mode = return_mode if return_mode is not None else self.getitemtype
        question_pretokenized, answers = example

        if return_mode == "seq":
            rettensor = torch.ones(self.ansmaxlen, dtype=torch.long) * -1000
            rettensor[0] = self.D["[BOS]"]
            pos = 1
            for answertensor in answers:
                newpos = pos + answertensor.size(0) - 1
                if newpos + 1 < self.ansmaxlen:
                    rettensor[pos:newpos] = answertensor[0:newpos-pos]
                    pos = newpos
                else:
                    newpos = pos
            rettensor[newpos] = 1       # EOS
            rettensor = rettensor[:newpos+1]

            if not torch.all(rettensor >= 0):
                assert torch.all(rettensor >= 0)

            return [question_pretokenized], [rettensor]

        elif return_mode == "seqset":

            if len(answers) > self.maxitemnr:
                answers = answers[:self.maxitemnr]
            rets = [(self.D[f"[ITEM-{i}]"], answertensor) for i, answertensor in enumerate(answers)]
            choices = random.sample(range(len(rets)), min(len(rets), self.numsamples))
            rettensors = []
            for choice in choices:
                itemnr, answertensor = rets[choice]
                rettensor = [itemnr, self.D[f"[TOTAL-{len(rets)}]"]] + [0] * answertensor.size(0)
                rettensor = torch.tensor(rettensor, device=answertensor.device, dtype=answertensor.dtype)
                rettensor[2:] = answertensor[:]
                rettensors.append(rettensor)

            return [question_pretokenized] * len(choices), rettensors


def try_webqa(recompute = True):
    print("loading tokenizer")
    extra_tokens = ["[SEP1]", "[SEP2]", "[ANS]", "[ENT]", "[REL]", "[SEPITEM]", "[BOS]", "[ENDOFSET]", "[LASTITEM]"] # + [f"extra_id_{i}" for i in range(0)]
    extra_tokens = extra_tokens + [f"[ITEM-{i}]" for i in range(1000)] + [f"[TOTAL-{i}]" for i in range(1000)]
    tok = T5TokenizerFast.from_pretrained("google/t5-v1_1-base", additional_special_tokens=extra_tokens, extra_ids=0)
    print(len(tok.vocab))

    # kbds, validkbds = WebQADatasetLoader().load_kb(tok, recompute=recompute, mode="seqset")
    # # kbds = kbds.map(kbds.item_mapper)
    # print(len(kbds))
    # print("\n".join([str(kbds[i]) for i in range(15)]))

    trainqads, evaltrainds, validqads, testqads = WebQADatasetLoader().load_qa(tok=tok, recompute=recompute,
                                                    mode="seqset")
    print("\n".join([str(trainqads[i]) for i in range(15)]))
    print("\n".join([str(validqads[i]) for i in range(15)]))
    # tok.add_tokens(["[SEP1]", "[SEP2]"])
    #
    # t5 = T5Model.from_pretrained("google/t5-v1_1-base")
    # emb = t5.encoder.embed_tokens
    # # from parseq.scripts_qa.bert import adapt_embeddings
    # from parseq.scripts_cbqa.adapter_t5 import adapt_embeddings
    # emb = adapt_embeddings(emb, tok)
    # #
    # tokids = tok("[SEP1] [SEP2] <unk> </s> <pad>", return_tensors="pt")["input_ids"]
    # embs = emb(tokids)
    # t5(tokids)


if __name__ == '__main__':
    try_webqa()
