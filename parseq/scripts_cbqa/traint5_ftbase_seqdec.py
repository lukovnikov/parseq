import fire
import re
import qelos as q
import torch
from parseq.datasets import autocollate

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
    VARIANT = "seqdec"
    MAXLEN = 200

    def get_model(self):
        return NewModel

    def get_predictions(self, m:Model, ds, tok=None, batsize=10, device=torch.device("cpu"), maxnumans=100):
        tto = q.ticktock("pred")
        tto.tick("predicting")
        tt = q.ticktock("-")

        m.eval()
        m.to(device)

        ret = {}
        with torch.no_grad():
            batch = []
            i = 0
            done = False
            while not done:
                while len(batch) < batsize:
                    inpstr = ds[i][0][0]
                    inpstr = decode(inpstr, tok)
                    batch.append((inpstr, ds[i][0][0], ds[i][1][0]))
                    i += 1
                    if i == len(ds):
                        done = True
                        break

                # tt.msg(f"{i}/{len(ds)}")
                tt.live(f"{i}/{len(ds)}")
                packedbatch = autocollate(batch)
                packedbatch = q.recmap(packedbatch, lambda x: x.to(device) if hasattr(x, "to") else x)

                _, outs = m(*packedbatch[1:])
                for j, out in enumerate(outs):
                    inpstr = batch[j][0]
                    _inpstr, predstr, _ = out

                    predset = re.split(r"(\[ENT\]|\[REL\]|\[SEPITEM\])", predstr.replace("[BOS]", ""))
                    predset = set([re.sub(r"\[[^\]]+\]", "", ae).strip() for ae in predset]) - {""}
                    ret[inpstr] = predset

                batch = []

        return ret


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
        nosave=False,
        loadfrom="",
        usesavedconfig=False,
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