import fire
import re
from transformers import T5ForConditionalGeneration

import qelos as q
import torch
from parseq.datasets import autocollate
from parseq.scripts_cbqa.adapter_t5 import adapt_t5_to_tok

from parseq.scripts_cbqa.traint5_ftbase_setdec import Main, decode, Model

# uses decoder to generate answer string

class NewModel(Model):

    def test_forward(self, x, y):
        xmask = x != 0
        maxlen = min(self.maxlen, y.size(1) + 1)

        preds = self.model.generate(
            input_ids = x,
            decoder_input_ids=y[:, 0:1],
            attention_mask = xmask,
            max_length=maxlen,
            num_beams=1,
            do_sample=False,
            )

        if preds.size(-1) < y.size(-1):
            # acc = torch.zeros(len(x), device=x.device)
            app = torch.zeros(preds.size(0), y.size(1) - preds.size(1), device=preds.device, dtype=preds.dtype)
            preds = torch.cat([preds, app], 1)
            same = preds == y
        else:
            same = preds[..., :y.size(-1)] == y
        same |= y == 0
        acc = torch.all(same, -1).float()

        inpstring = [decode(xe, self.tok) for xe in x.unbind(0)]
        predstring = [decode(g, self.tok) for g in preds.unbind(0)]
        targetstring = [decode(t, self.tok) for t in y.unbind(0)]

        predsets = [re.split(r"(\[ENT\]|\[REL\]|\[SEPITEM\])", a.replace("[BOS]", "")) for a in predstring]
        targetsets = [re.split(r"(\[ENT\]|\[REL\]|\[SEPITEM\])", a.replace("[BOS]", "")) for a in targetstring]
        predsets = [set([re.sub(r"\[[^\]]+\]", "", ae).strip() for ae in a]) for a in predsets]
        targetsets = [set([re.sub(r"\[[^\]]+\]", "", ae).strip() for ae in a]) for a in targetsets]

        rets = []
        for pred, exp in zip(predsets, targetsets):
            pred, exp = pred - {""}, exp - {""}
            tp = len(pred & exp)
            recall = tp / max(1e-6, len(exp))
            precision = tp / max(1e-6, len(pred))
            fscore = 2 * recall * precision / max(1e-6, (recall + precision))
            rets.append((fscore, precision, recall))

        fscores, precisions, recalls = zip(*rets)

        # stracc = [float(a == b) for a, b in zip(predstring, targetstring)]
        # stracc = torch.tensor(stracc, device=x.device)
        fscores = torch.tensor(fscores, device=x.device)

        ret = list(zip(inpstring, predstring, targetstring))
        return {"fscore": fscores}, ret


class NewMain(Main):
    DATAMODE = "seq"
    TESTMETRIC = "fscore"
    VARIANT = "adapter-setdec"
    MAXLEN = 200

    def get_task_model(self):
        return NewModel

    def create_model(self, tok=None, modelsize=None, maxlen=None, dropout=0., loadfrom=None, tt=None,
                     adapterdim=-1, adapterlayers=-1, **kwargs):
        tt.tick("model")
        modelname = f"google/t5-v1_1-{modelsize}"
        t5model = T5ForConditionalGeneration.from_pretrained(modelname)
        t5model = adapt_t5_to_tok(t5model, tok)
        t5model = add_ff_adapters(t5model, adapterdim=adapterdim, adapterlayers=adapterlayers)
        m = self.get_task_model()(t5model, t5model.config.d_model, tok=tok)

        if len(loadfrom) > 0:
            tt.tick(f"Loading model from runs/{self.VARIANT}/{loadfrom}/model.ckpt")
            m.load_state_dict(torch.load(f"runs/{self.VARIANT}/{loadfrom}/model.ckpt"))
            tt.tock()

        m.maxlen = maxlen

        def _set_dropout(m, _p=0.):
            if isinstance(m, torch.nn.Dropout):
                m.p = _p
        m.apply(partial(_set_dropout, _p=dropout))
        tt.tock()


def run_experiment(
        lr=-1.,
        kbpretrainepochs=0,
        kbpretrainvalidinter=-1,
        useadafactor=False,
        gradnorm=2,
        gradacc=1,
        batsize=-1,
        maxlen=-1,
        testbatsize=-1,
        epochs=-1,
        validinter=-1.,
        validfrac=0.1,
        warmup=0.1,
        cosinecycles=0,
        modelsize="small",
        dataset="default",
        seed=-1,
        dropout=-1.,
        testcode=False,
        debugcode=False,
        gpu=-1,
        recomputedata=False,
        dosave=False,
        loadfrom="",
        usesavedconfig=False,
        adapterdim=-1,
):
    settings = locals().copy()

    main = NewMain()
    maxlen = main.MAXLEN if maxlen == -1 else maxlen
    settings["maxlen"] = maxlen

    ranges = {
        "dataset": ["metaqa/1"],
        "dropout": [0.],
        "seed": [42, 87646464, 456852],
        "epochs": [50],
        "batsize": [60],
        "lr": [0.0005],
        "modelsize": ["base"],
        "validinter": [5],
    }

    for k in ranges:
        if k in settings:
            if isinstance(settings[k], str) and settings[k] != "default":
                ranges[k] = settings[k].split(",")
            elif isinstance(settings[k], (int, float)) and settings[k] >= 0:
                ranges[k] = [settings[k]]
            else:
                pass
                # raise Exception(f"something wrong with setting '{k}'")
            del settings[k]

    def checkconfig(spec):
        return True

    q.run_experiments_random(
        main.run, ranges, path_prefix=None, check_config=checkconfig, **settings)


if __name__ == '__main__':
    q.argprun(run_experiment)