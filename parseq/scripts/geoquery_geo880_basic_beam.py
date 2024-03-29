import os
import re
import sys
from functools import partial
from typing import *

import torch

import qelos as q
from nltk import PorterStemmer

from parseq.decoding import SeqDecoder, TFTransition, FreerunningTransition, BeamDecoder, BeamTransition
from parseq.eval import CELoss, SeqAccuracies, BeamSeqAccuracies, make_array_of_metrics
from parseq.scripts.geoquery_geo880_basic import GeoQueryDatasetSub as GeoQueryDataset, do_rare_stats, create_model
from parseq.vocab import SequenceEncoder


def run(lr=0.01,
        batsize=20,
        epochs=101,
        embdim=100,
        encdim=200,
        numlayers=1,
        dropout=.25,
        wreg=1e-6,
        cuda=False,
        gpu=0,
        minfreq=2,
        gradnorm=3.,
        beamsize=5,
        smoothing=0.,
        fulltest=False,
        cosine_restarts=1.,
        nocopy=True,
        validinter=5,
        ):
    print(locals().copy())
    tt = q.ticktock("script")
    device = torch.device("cpu") if not cuda else torch.device("cuda", gpu)
    tt.tick("loading data")
    # stemmer = PorterStemmer()
    # tokenizer = lambda x: [stemmer.stem(xe) for xe in x.split()]
    tokenizer = lambda x: x.split()
    ds = GeoQueryDataset(sentence_encoder=SequenceEncoder(tokenizer=tokenizer), min_freq=minfreq)
    dls = ds.dataloader(batsize=batsize)
    train_dl = ds.dataloader("train", batsize=batsize)
    test_dl = ds.dataloader("test", batsize=batsize)
    tt.tock("data loaded")

    do_rare_stats(ds)

    # batch = next(iter(train_dl))
    # print(batch)
    # print("input graph")
    # print(batch.batched_states)

    model = create_model(embdim=embdim, hdim=encdim, dropout=dropout, numlayers=numlayers,
                             sentence_encoder=ds.sentence_encoder, query_encoder=ds.query_encoder, feedatt=True, nocopy=nocopy)

    tfdecoder = SeqDecoder(TFTransition(model),
                           [CELoss(ignore_index=0, mode="logprobs", smoothing=smoothing),
                            SeqAccuracies()])
    # beamdecoder = BeamActionSeqDecoder(tfdecoder.model, beamsize=beamsize, maxsteps=50)
    freedecoder = BeamDecoder(model, beamsize=beamsize, maxtime=60,
                              eval_beam=[BeamSeqAccuracies()])

    losses = make_array_of_metrics("loss", "elem_acc", "seq_acc")
    vlosses = make_array_of_metrics(*([f"beam_seq_recall_at{i}" for i in range(1, min(beamsize, 5))] + ["beam_recall"]))

    # 4. define optim
    optim = torch.optim.RMSprop(tfdecoder.parameters(), lr=lr, weight_decay=wreg)

    # optim = torch.optim.SGD(tfdecoder.parameters(), lr=lr, weight_decay=wreg)

    # lr schedule
    if cosine_restarts >= 0:
        t_max = epochs # * len(train_dl)
        print(f"Total number of updates: {t_max} ({epochs} * {len(train_dl)})")
        lr_schedule = q.WarmupCosineWithHardRestartsSchedule(optim, 0, t_max, cycles=cosine_restarts)
        reduce_lr = [lambda: lr_schedule.step()]
    else:
        reduce_lr = []

    # 6. define training function (using partial)
    clipgradnorm = lambda: torch.nn.utils.clip_grad_norm_(tfdecoder.parameters(), gradnorm)
    trainbatch = partial(q.train_batch, on_before_optim_step=[clipgradnorm])
    trainepoch = partial(q.train_epoch, model=tfdecoder, dataloader=train_dl, optim=optim, losses=losses,
                         _train_batch=trainbatch, device=device, on_end=reduce_lr)

    # 7. define validation function (using partial)
    validepoch = partial(q.test_epoch, model=freedecoder, dataloader=test_dl, losses=vlosses, device=device)
    # validepoch = partial(q.test_epoch, model=tfdecoder, dataloader=test_dl, losses=vlosses, device=device)

    # 7. run training
    tt.tick("training")
    q.run_training(run_train_epoch=trainepoch, run_valid_epoch=validepoch, max_epochs=epochs, validinter=validinter)
    tt.tock("done training")



if __name__ == '__main__':
    # try_build_grammar()
    # try_dataset()
    q.argprun(run)