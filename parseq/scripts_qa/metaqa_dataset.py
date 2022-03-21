import os
import tqdm

from parseq.datasets import Dataset


class MetaQADatasetLoader(object):
    def __init__(self,
                 p="../../datasets/metaqa/",
                 ):
        super(MetaQADatasetLoader, self).__init__()
        self.p = p

    def load_qa(self, hops="all"):
        """
        :param which:  "all" or "1", or "2" or "3" or "1+2" etc
        :return:
        """
        if hops == "all":
            hops = "1+2+3"
        which = hops.split("+")

        data = []
        for which_elem in which:
            print(f"loading {which_elem}-hop")
            path = os.path.join(self.p, f"{which_elem}-hop", "vanilla")
            # load train data
            print("loading train")
            with open(os.path.join(path, "qa_train.txt"), encoding="utf-8") as f:
                for line in tqdm.tqdm(f.readlines()):
                    question, answers = self.process_qa_line(line)
                    data.append((question, answers, which_elem, "train"))
            print("loading test")
            with open(os.path.join(path, "qa_test.txt"), encoding="utf-8") as f:
                for line in tqdm.tqdm(f.readlines()):
                    question, answers = self.process_qa_line(line)
                    data.append((question, answers, which_elem, "test"))
            print("loading valid")
            with open(os.path.join(path, "qa_dev.txt"), encoding="utf-8") as f:
                for line in tqdm.tqdm(f.readlines()):
                    question, answers = self.process_qa_line(line)
                    data.append((question, answers, which_elem, "valid"))

        ds = Dataset(data)
        return ds

    def process_qa_line(self, line):
        line = line.strip()
        question, answers = line.split("\t")
        answers = answers.split("|")
        question = question.replace("[", "").replace("]", "")
        return question, answers

    def load_kb(self):
        print("loading KB")
        triples = []
        with open(os.path.join(self.p, "kb.txt"), encoding="utf-8") as f:
            for line in tqdm.tqdm(f.readlines()):
                newtriples = self.process_kb_line(line)
                triples.extend(newtriples)
        ds = Dataset(triples)
        return ds

    def process_kb_line(self, line:str):
        line = line.strip()
        subj, rel, obj = line.split("|")
        rel = rel.replace("_", " ")
        ret = [(subj, rel, obj)]
        # ret = [(subj, f"is {rel}", obj), (obj, f"is {rel} of", subj)]
        return ret


def try_metaqa():
    ds = MetaQADatasetLoader().load_qa("1+2")
    print(len(ds))
    print(ds[:5])

    kbds = MetaQADatasetLoader().load_kb()
    print(len(kbds))
    print(kbds[:15])


if __name__ == '__main__':
    try_metaqa()
