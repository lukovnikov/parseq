import random
from functools import partial
from typing import Callable, Set

import qelos as q
import numpy as np
import torch
from torch.utils.data import DataLoader

from parseq.datasets import OvernightDatasetLoader, pad_and_default_collate, autocollate
from parseq.decoding import merge_metric_dicts
from parseq.eval import SeqAccuracies, TreeAccuracy, make_loss_array, CELoss
from parseq.grammar import tree_to_lisp_tokens, lisp_to_tree
from parseq.vocab import SequenceEncoder, Vocab
from transformers import AutoTokenizer, AutoModel, BartConfig, BartModel, BartForConditionalGeneration


UNKID = 3


def load_ds(domain="restaurants", min_freq=0, top_k=np.infty, nl_mode="bart-large"):
    ds = OvernightDatasetLoader().load(domain=domain)

    seqenc_vocab = Vocab(padid=1, startid=0, endid=2, unkid=UNKID)
    seqenc = SequenceEncoder(vocab=seqenc_vocab, tokenizer=tree_to_lisp_tokens,
                             add_start_token=True, add_end_token=True)
    for example in ds.examples:
        query = example[1]
        seqenc.inc_build_vocab(query, seen=example[2] == "train")
    seqenc.finalize_vocab(min_freq=min_freq, top_k=top_k)

    nl_tokenizer = AutoTokenizer.from_pretrained(nl_mode)
    def tokenize(x):
        ret = (nl_tokenizer.encode(x[0], return_tensors="pt")[0],
               seqenc.convert(x[1], return_what="tensor"),
               x[2],
               x[0], x[1])
        return ret
    tds, vds, xds = ds[(None, None, "train")].map(tokenize), \
                    ds[(None, None, "valid")].map(tokenize), \
                    ds[(None, None, "test")].map(tokenize)
    return tds, vds, xds, nl_tokenizer, seqenc


class BartGenerator(BartForConditionalGeneration):
    def __init__(self, config:BartConfig):
        super(BartGenerator, self).__init__(config)
        self.outlin = torch.nn.Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        use_cache=False,
        **unused
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )
        lm_logits = self.outlin(outputs[0])
        outputs = (lm_logits,) + outputs[1:]  # Add hidden states and attention if they are here
        return outputs


class BartGeneratorTrain(torch.nn.Module):
    def __init__(self, model:BartGenerator, smoothing=0., tensor2tree:Callable=None, orderless:Set[str]=set(), **kw):
        super(BartGeneratorTrain, self).__init__(**kw)
        self.model = model

        # CE loss
        self.ce = CELoss(ignore_index=model.config.pad_token_id, smoothing=smoothing)

        # accuracies
        self.accs = SeqAccuracies()
        self.accs.padid = model.config.pad_token_id
        self.accs.unkid = UNKID

        self.treeacc = TreeAccuracy(tensor2tree=tensor2tree,
                                    orderless=orderless)

        self.metrics = [self.ce, self.accs, self.treeacc]

    def forward(self, input_ids, output_ids, *args, **kwargs):
        ret = self.model(input_ids, decoder_input_ids=output_ids)
        probs = ret[0]
        _, predactions = probs.max(-1)
        outputs = [metric(probs, predactions, output_ids[:, 1:]) for metric in self.metrics]
        outputs = merge_metric_dicts(*outputs)
        return outputs, ret


class BartGeneratorTest(BartGeneratorTrain):
    def __init__(self, model:BartGenerator, maxlen:int=5, numbeam:int=None,
                 tensor2tree:Callable=None, orderless:Set[str]=set(), **kw):
        super(BartGeneratorTest, self).__init__(model, **kw)
        self.maxlen, self.numbeam = maxlen, numbeam
        # accuracies
        self.accs = SeqAccuracies()
        self.accs.padid = model.config.pad_token_id
        self.accs.unkid = UNKID

        self.treeacc = TreeAccuracy(tensor2tree=tensor2tree,
                                    orderless=orderless)

        self.metrics = [self.accs, self.treeacc]

    def forward(self, input_ids, output_ids, *args, **kwargs):
        ret = self.model.generate(input_ids, max_length=self.maxlen, num_beams=self.numbeam)
        outputs = [metric(None, ret, output_ids) for metric in self.metrics]
        outputs = merge_metric_dicts(*outputs)
        return outputs, ret


def create_model(encoder_name="bart-large",
                 dec_vocabsize=None, dec_layers=6, dec_dim=640, dec_heads=8, dropout=0.,
                 maxlen=20, smoothing=0., tensor2tree=None):
    layerdrop = 0
    if encoder_name != "bart-large":
        raise NotImplemented(f"encoder '{encoder_name}' not supported yet.")
    pretrained = AutoModel.from_pretrained(encoder_name)
    encoder = pretrained.encoder

    if pretrained.config.d_model != dec_dim:
        class BartEncoderWrapper(torch.nn.Module):
            def __init__(self, model, **kw):
                super(BartEncoderWrapper, self).__init__(**kw)
                self.model = model
                self.proj = torch.nn.Linear(pretrained.config.d_model, dec_dim, bias=False)

            def forward(self, input_ids, attention_mask):
                ret = self.model(input_ids, attention_mask=attention_mask)
                ret = (self.proj(ret[0]), ret[1], ret[2])
                return ret

        encoder = BartEncoderWrapper(encoder)

    decoder_config = BartConfig(d_model=dec_dim,
                                pad_token_id=1,
                                vocab_size=dec_vocabsize,
                                decoder_attention_heads=dec_heads,
                                decoder_layers=dec_layers,
                                dropout=dropout,
                                decoder_layerdrop=layerdrop)
    model = BartGenerator(decoder_config)
    model.model.encoder = encoder

    orderless = {"op:and", "SW:concat"}

    trainmodel = BartGeneratorTrain(model, smoothing=smoothing, tensor2tree=tensor2tree, orderless=orderless)
    testmodel = BartGeneratorTest(model, maxlen=maxlen, numbeam=None, tensor2tree=tensor2tree, orderless=orderless)
    return trainmodel, testmodel


def _tensor2tree(x, D:Vocab=None):
    # x: 1D int tensor
    x = list(x.detach().cpu().numpy())
    x = [D(xe) for xe in x]
    x = [xe for xe in x if xe != D.padtoken]

    # find first @END@ and cut off
    parentheses_balance = 0
    for i in range(len(x)):
        if x[i] ==D.endtoken:
            x = x[:i]
            break
        elif x[i] == "(" or x[i][-1] == "(":
            parentheses_balance += 1
        elif x[i] == ")":
            parentheses_balance -= 1
        else:
            pass

    # balance parentheses
    while parentheses_balance > 0:
        x.append(")")
        parentheses_balance -= 1
    i = len(x) - 1
    while parentheses_balance < 0 and i > 0:
        if x[i] == ")":
            x.pop(i)
            parentheses_balance += 1
        i -= 1

    # convert to nltk.Tree
    try:
        tree, parsestate = lisp_to_tree(" ".join(x), None)
    except Exception as e:
        tree = None
    return tree


def run(domain="restaurants",
        lr=0.001,
        enclr=0.0001,
        cosinelr=False,
        warmup=0.,
        batsize=20,
        epochs=100,
        dropout=0.1,
        wreg=1e-6,
        gradnorm=3,
        smoothing=0.,
        gpu=-1,
        seed=123456789,
        encoder="bart-large",
        numlayers=6,
        hdim=640,
        numheads=8,
        maxlen=50,
        ):
    localargs = locals().copy()
    print(locals())
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tt = q.ticktock("script")
    device = torch.device("cpu") if gpu < 0 else torch.device(gpu)

    tt.tick("loading data")
    tds, vds, xds, nltok, flenc = load_ds(domain=domain, nl_mode=encoder)
    tdl = DataLoader(tds, batch_size=batsize, shuffle=True, collate_fn=partial(autocollate, pad_value=1))
    vdl = DataLoader(vds, batch_size=batsize, shuffle=False, collate_fn=partial(autocollate, pad_value=1))
    xdl = DataLoader(xds, batch_size=batsize, shuffle=False, collate_fn=partial(autocollate, pad_value=1))
    tt.tock("data loaded")

    tt.tick("creating model")
    trainm, testm = create_model(encoder_name=encoder,
                                 dec_vocabsize=flenc.vocab.number_of_ids(),
                                 dec_layers=numlayers,
                                 dec_dim=hdim,
                                 dec_heads=numheads,
                                 dropout=dropout,
                                 smoothing=smoothing,
                                 maxlen=maxlen,
                                 tensor2tree=partial(_tensor2tree, D=flenc.vocab)
                                 )
    tt.tock("model created")

    # run a batch of data through the model
    if False:
        batch = next(iter(tdl))
        out = trainm(*batch)
        print(out)
        out = testm(*batch)
        print(out)

    losses = make_loss_array("loss", "seq_acc", "tree_acc")
    vlosses = make_loss_array("seq_acc", "tree_acc")

    trainable_params = trainm.named_parameters()
    exclude_params = set()
    # exclude_params.add("model.model.inp_emb.emb.weight")  # don't train input embeddings if doing glove
    if len(exclude_params) > 0:
        trainable_params = [(k, v) for k, v in trainable_params if k not in exclude_params]

    encparams = [v for k, v in trainable_params if k.startswith("model.model.encoder")]
    otherparams = [v for k, v in trainable_params if not k.startswith("model.model.encoder")]
    if len(encparams) == 0:
        raise Exception("No encoder parameters found!")
    paramgroups = [{"params": encparams, "lr": enclr},
                   {"params": otherparams, "lr": lr}]

    optim = torch.optim.Adam(paramgroups, lr=lr, weight_decay=wreg)

    clipgradnorm = lambda: torch.nn.utils.clip_grad_norm_(trainm.parameters(), gradnorm)

    t_max = epochs
    print(f"Total number of updates: {t_max} .")
    if cosinelr:
        lr_schedule = q.sched.Linear(steps=warmup) >> q.sched.Cosine(nsteps=t_max-warmup) >> 0.
    else:
        lr_schedule = q.sched.Linear(steps=warmup) >> 1.

    trainbatch = partial(q.train_batch, on_before_optim_step=[clipgradnorm])
    trainepoch = partial(q.train_epoch, model=trainm, dataloader=tdl, optim=optim, losses=losses,
                         _train_batch=trainbatch, device=device, on_end=[lambda: lr_schedule.step()])
    validepoch = partial(q.test_epoch, model=testm, dataloader=vdl, losses=vlosses, device=device)

    tt.tick("training")
    q.run_training(run_train_epoch=trainepoch, run_valid_epoch=validepoch, max_epochs=epochs)
    tt.tock("done training")

    tt.tick("testing")
    testresults = q.test_epoch(model=testm, dataloader=xdl, losses=vlosses, device=device)
    print(testresults)
    tt.tock("tested")

    print("done")



if __name__ == '__main__':
    q.argprun(run)