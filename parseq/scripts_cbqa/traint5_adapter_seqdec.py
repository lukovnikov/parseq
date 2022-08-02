from functools import partial

import fire
import re
from transformers import T5ForConditionalGeneration

import qelos as q
import torch
from parseq.datasets import autocollate
from parseq.scripts_cbqa.adapter_t5 import adapt_t5_to_tok, add_ff_adapters

from parseq.scripts_cbqa.traint5_adapter_setdec import Main as OldMain
from parseq.scripts_cbqa.traint5_ftbase_seqdec import Model as Model, Main as OldMain2

# uses decoder to generate answer string


class Main(OldMain, OldMain2):
    DATAMODE = "seq"
    TESTMETRIC = "fscore"
    VARIANT = "adapter-seqdec"
    MAXLEN = 200

    def create_model(self, tok=None, modelsize=None, maxlen=None, dropout=0., loadfrom=None, tt=None,
                     adapterdim=-1, **kwargs):
        tt.tick("model")
        modelname = f"google/t5-v1_1-{modelsize}"
        t5model = T5ForConditionalGeneration.from_pretrained(modelname)
        t5model = adapt_t5_to_tok(t5model, tok)
        t5model = add_ff_adapters(t5model, adapterdim=adapterdim, adaptmode="ff")
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
        return m


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

    main = Main()
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