from copy import deepcopy

import torch
from parseq.vocab import Vocab
from parseq.scripts_compgen_new.transformer import TransformerConfig, TransformerStack
from parseq.scripts_compgen_new.transformerdecoder import TransformerStack as TransformerStackDecoder
from transformers import AutoTokenizer, BertModel


class TransformerEmbeddings(torch.nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, vocab_size, hidden_size, dropout=0., pad_token_id=0, max_position_embeddings=512,
                 layer_norm_eps=1e-12, useabspos=True):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = torch.nn.Embedding(max_position_embeddings, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = torch.nn.Dropout(dropout)

        self.useabspos = useabspos

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        ret = inputs_embeds

        if self.useabspos:
            if position_ids is None:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(input_shape)
            position_embeddings = self.position_embeddings(position_ids)
            ret = ret + position_embeddings

        ret = self.LayerNorm(ret)
        ret = self.dropout(ret)
        return ret


class BasicRelPosEmb(torch.nn.Module):
    ## Note: Even if this is shared across layers, keep the execution separate between layers because attention weights are different
    def __init__(self, dim, rng=10, **kw):
        super(BasicRelPosEmb, self).__init__(**kw)

        self.D = ["@PAD@"] + [str(i-rng) for i in range(rng)] + [str(i) for i in range(rng+1)]
        self.D = dict(zip(self.D, range(len(self.D))))
        self.emb = torch.nn.Embedding(len(self.D), dim, padding_idx=0)
        self.embv = torch.nn.Embedding(len(self.D), dim, padding_idx=0)

    def get_vectors(self, query, relpos, keyorvalue="key"):
        """
        :param q:       (batsize, numheads, qlen, dimperhead)
        :param relpos:  (batsize, qlen, klen, n)
        :return:        (batsize, numheads, qlen, klen, dimperhead)
        """
        ret = None
        for n in range(relpos.size(-1)):
            indexes = torch.arange(0, self.emb.num_embeddings, device=query.device).long()
            if keyorvalue.startswith("k"):
                embs = self.emb(indexes)# (numindexes, dim)
            elif keyorvalue.startswith("v"):
                embs = self.embv(indexes)
            embs = embs.view(embs.size(0), query.size(1), query.size(-1))        # (numindexes, numheads, dimperhead)
            vectors = relpos[:, :, :, n][embs]      # (batsize, qlen, klen, numheads, dimperhead)
            vectors = vectors.permute(0, 3, 1, 2, 4)
            if ret is None:
                ret = torch.zeros_like(vectors)
            ret = ret + vectors
        return ret

    def compute_scores(self, query, relpos):
        """
        :param q:       (batsize, numheads, qlen, dimperhead)
        :param relpos:  (batsize, qlen, klen, n)
        :return:
        """
        retscores = None
        for n in range(relpos.size(-1)):
            indexes = torch.arange(0, self.emb.num_embeddings, device=query.device).long()
            embs = self.emb(indexes)# (numindexes, dim)
            embs = embs.view(embs.size(0), query.size(1), query.size(-1))        # (numindexes, numheads, dimperhead)
            relpos_ = relpos[:, :, :, n]
            scores = torch.einsum("bhqd,nhd->bhqn", query, embs)  # (batsize, numheads, qlen, numindexes)
            relpos_ = relpos_[:, None, :, :].repeat(scores.size(0), scores.size(1), 1, 1)  # (batsize, numheads, qlen, klen)
            # print(scores.size(), relpos_.size())
            scores_ = torch.gather(scores, 3, relpos_)  # (batsize, numheads, qlen, klen)
            if retscores is None:
                retscores = torch.zeros_like(scores_)
            retscores = retscores + scores_
        return retscores        # (batsize, numheads, qlen, klen)

    def compute_context(self, weights, relpos):
        """
        :param weights: (batsize, numheads, qlen, klen)
        :param relpos:  (batsize, qlen, klen, 1)
        :return:    # weighted sum over klen (batsize, numheads, qlen, dimperhead)
        """
        ret = None
        batsize = weights.size(0)
        numheads = weights.size(1)
        qlen = weights.size(2)
        device = weights.device

        # Naive implementation builds matrices of (batsize, numheads, qlen, klen, dimperhead)
        # whereas normal transformer only (batsize, numheads, qlen, klen) and (batsize, numheads, klen, dimperhead)
        for n in range(relpos.size(-1)):
            relpos_ = relpos[:, :, :, n]

            # map relpos_ to compact integer space of unique relpos_ entries
            try:
                relpos_unique = relpos_.unique()
            except Exception as e:
                raise e
            mapper = torch.zeros(relpos_unique.max() + 1, device=device, dtype=torch.long)  # mapper is relpos_unique but the other way around
            mapper[relpos_unique] = torch.arange(0, relpos_unique.size(0), device=device).long()
            relpos_mapped = mapper[relpos_]     # (batsize, qlen, klen) but ids are from 0 to number of unique relposes

            # sum up the attention weights which refer to the same relpos id
            # scatter: src is weights, index is relpos_mapped[:, None, :, :]
            # scatter: gathered[batch, head, qpos, relpos_mapped[batch, head, qpos, kpos]]
            #               += weights[batch, head, qpos, kpos]
            gathered = torch.zeros(batsize, numheads, qlen, relpos_unique.size(0), device=device)
            gathered = torch.scatter_add(gathered, -1, relpos_mapped[:, None, :, :].repeat(batsize, numheads, 1, 1), weights)
            # --> (batsize, numheads, qlen, numunique): summed attention weights

            # get embeddings and update ret
            embs = self.embv(relpos_unique).view(relpos_unique.size(0), numheads, -1)        # (numunique, numheads, dimperhead)
            relposemb = torch.einsum("bhqn,nhd->bhqd", gathered, embs)
            if ret is None:
                ret = torch.zeros_like(relposemb)
            ret  = ret + relposemb
        return ret


class WordDropout(torch.nn.Module):
    def __init__(self, p=0., maskid=1, keepids=None, **kw):
        super(WordDropout, self).__init__(**kw)
        self.dropout = torch.nn.Dropout(p)
        self.maskid = maskid
        self.keepids = [] if keepids is None else keepids

    def forward(self, x):
        if self.training and self.dropout.p > 0:
            worddropoutmask = self.dropout(torch.ones_like(x).float()) > 0
            for keepid in self.keepids:
                worddropoutmask = worddropoutmask | (x == keepid)
            x = torch.where(worddropoutmask, x, torch.ones_like(x) * self.maskid)
        return x


class TransformerDecoderCell(torch.nn.Module):
    def __init__(self, dim, vocab:Vocab=None, inpvocab:Vocab=None, numlayers:int=6, numheads:int=6, userelpos=False, useabspos=True,
                 relposmode="basic", relposrng=10,
                 dropout:float=0., worddropout:float=0., maxpos=512, bertname="bert-base-uncased", **kw):
        super(TransformerDecoderCell, self).__init__(**kw)
        self.vocab = vocab
        self.inpvocab = inpvocab
        self.vocabsize = vocab.number_of_ids()
        self.dim = dim
        self.userelpos = userelpos
        self.relposrng = relposrng
        self.useabspos = useabspos

        decconfig = TransformerConfig(vocab_size=self.vocabsize, d_model=self.dim, d_ff=self.dim * 4,
                                      d_kv=int(self.dim/numheads),
                                      num_layers=numlayers, num_heads=numheads, dropout_rate=dropout)

        self.dec_emb = torch.nn.Embedding(self.vocabsize, decconfig.d_model)
        self.slot_emb = torch.nn.Embedding(1, decconfig.d_model)

        self.relposemb = None
        if self.userelpos is True:
            if relposmode == "basic":
                self.relposemb = BasicRelPosEmb(self.dim, relposrng)
            # elif relposmode == "mod":
            #     self.relposemb = ModRelPosEmb(self.dim, relposrng, levels=4)
            else:
                raise Exception(f"Unrecognized relposmode '{relposmode}'")

        self.absposemb = None
        if self.relposemb is None or self.useabspos is True:
            self.absposemb = torch.nn.Embedding(maxpos, decconfig.d_model)

        decoder_config = deepcopy(decconfig)
        decoder_config.is_decoder = True
        decoder_config.use_causal_mask = True
        self.decoder = TransformerStackDecoder(decoder_config, rel_emb=self.relposemb)

        self.out = torch.nn.Linear(self.dim, self.vocabsize)

        vocab_mask = torch.ones(self.vocabsize)
        # for excl_token in self.exclude:
        #     if excl_token in self.vocab:
        #         vocab_mask[self.vocab[excl_token]] = 0
        self.register_buffer("vocab_mask", vocab_mask)

        self.bertname = bertname
        self.encrelposemb = None
        if self.bertname.startswith("none") or self.bertname == "vanilla":
            if self.userelpos is True:
                if relposmode == "basic":
                    self.encrelposemb = BasicRelPosEmb(self.dim, relposrng)
                # elif relposmode == "mod":
                #     self.relposemb = ModRelPosEmb(self.dim, relposrng, levels=4)
                else:
                    raise Exception(f"Unrecognized relposmode '{relposmode}'")
            bname = "bert" + self.bertname[4:]
            if self.bertname == "vanilla":
                inpvocabsize = inpvocab.number_of_ids()
                self.inpworddropout = WordDropout(worddropout, self.inpvocab[self.inpvocab.masktoken], [self.inpvocab[self.inpvocab.padtoken]])
            else:
                tokenizer = AutoTokenizer.from_pretrained(bname)
                inpvocabsize = tokenizer.vocab_size
                self.inpworddropout = WordDropout(worddropout, self.inpvocab[self.inpvocab.masktoken],
                                                  [self.inpvocab["[CLS]"], self.inpvocab["[SEP]"], self.inpvocab[self.inpvocab.padtoken]])
            encconfig = TransformerConfig(vocab_size=inpvocabsize, d_model=self.dim, d_ff=self.dim * 4,
                                          d_kv=int(self.dim/numheads),
                                          num_layers=numlayers, num_heads=numheads, dropout_rate=dropout)
            encemb = TransformerEmbeddings(encconfig.vocab_size, encconfig.d_model, dropout=dropout, max_position_embeddings=maxpos, useabspos=useabspos)
            self.encoder_model = TransformerStack(encconfig, encemb, rel_emb=self.encrelposemb)
        else:
            self.encoder_model = BertModel.from_pretrained(self.bertname,
                                                           hidden_dropout_prob=min(dropout, 0.2),
                                                           attention_probs_dropout_prob=min(dropout, 0.1))
            tokenizer = AutoTokenizer.from_pretrained(self.bertname)
            inpvocabsize = tokenizer.vocab_size
            self.inpvocab = Vocab()
            for tok, id in tokenizer.vocab.items():
                self.inpvocab.D[tok] = id
            self.inpvocab.masktoken = "[MASK]"
            self.inpvocab.unktoken = "[UNK]"
            self.inpvocab.padtoken = "[PAD]"
            self.inpworddropout = WordDropout(worddropout, self.inpvocab[self.inpvocab.masktoken], [self.inpvocab["[CLS]"], self.inpvocab["[SEP]"], self.inpvocab[self.inpvocab.padtoken]])

        self.adapter = None
        if self.encoder_model.config.hidden_size != decoder_config.d_model:
            self.adapter = torch.nn.Linear(self.encoder_model.config.hidden_size, decoder_config.d_model, bias=False)

        self.worddropout = WordDropout(worddropout, self.vocab[self.vocab.masktoken], [self.vocab[self.vocab.padtoken]])

        self.reset_parameters()

    def encode_source(self, x):
        encmask = (x != 0)
        x = self.inpworddropout(x)
        relpos = None
        if self.encrelposemb is not None:      # compute relative positions
            positions = torch.arange(x.size(1), device=x.device)
            relpos = positions[None, :] - positions[:, None]
            relpos = relpos.clamp(-self.relposrng, self.relposrng) + self.relposrng + 1
            relpos = relpos[None, :, :, None]
        if relpos is not None:
            encs = self.encoder_model(x, attention_mask=encmask, relpos=relpos)[0]
        else:
            encs = self.encoder_model(x, attention_mask=encmask)[0]
        if self.adapter is not None:
            encs = self.adapter(encs)
        return encs, encmask

    def reset_parameters(self):
        pass
        # self.posemb.weight.fill_(0.)

    def forward(self, tokens:torch.Tensor=None, enc=None, encmask=None, cache=None, _full_sequence=False):
        padmask = (tokens != 0)
        tokens = self.worddropout(tokens)
        embs = self.dec_emb(tokens)     # (bs, seqlen, dim)
        slots = self.slot_emb.weight[0][None, None].repeat(embs.size(0), embs.size(1), 1)     # (1, seqlen, dim)
        if self.absposemb is not None:
            posembs = self.absposemb(torch.arange(tokens.size(1)+1, device=tokens.device))[None]
            embs = embs + posembs[:, :-1, :]
            slots = slots + posembs[:, 1:, :]
        relpos = None
        if self.relposemb is not None:      # compute relative positions
            positions = torch.arange(tokens.size(1)+1, device=tokens.device)
            positions = torch.cat([positions[:, None], positions[:, None]], 1).view(-1)
            relpos = positions[None, :] - positions[:, None]
            relpos = relpos[1:-1, 1:-1]
            relpos = relpos.clamp(-self.relposrng, self.relposrng) + self.relposrng + 1
            relpos = relpos[None, :, :, None]
        embs = torch.cat([embs[:, :, None, :], slots[:, :, None, :]], 2).view(embs.size(0), -1, embs.size(2))
        padmask = torch.cat([padmask[:, :, None], padmask[:, :, None]], 2).view(padmask.size(0), -1)

        if cache is not None:
            embs = embs[:, -2:, :]
            if relpos is not None:
                relpos = relpos[:, -2:, :, :]

        _ret = self.decoder(inputs_embeds=embs, attention_mask=padmask,
                     encoder_hidden_states=enc,
                     encoder_attention_mask=encmask, use_cache=True,
                     past_key_value_states=cache,
                     relpos=relpos)
        ret = _ret[0]
        c = ret.view(ret.size(0), int(ret.size(1)/2), 2, ret.size(2))[:, :, 1, :]
        cache = _ret[1]
        logits = self.out(c)
        return logits, cache

    # def forward(self, tokens:torch.Tensor=None, enc=None, encmask=None, cache=None):
    #     padmask = (tokens != 0)
    #     embs = self.dec_emb(tokens)
    #     if self.absposemb is not None:
    #         posembs = self.absposemb(torch.arange(tokens.size(1), device=tokens.device))[None]
    #         embs = embs + posembs
    #     relpos = None
    #     if self.relposemb is not None:      # compute relative positions
    #         positions = torch.arange(tokens.size(1), device=tokens.device)
    #         relpos = positions[None, :] - positions[:, None]
    #         relpos = relpos.clamp(-self.relposrng, self.relposrng) + self.relposrng + 1
    #         relpos = relpos[None, :, :, None]
    #     if cache is not None:
    #         embs = embs[:, -1:, :]
    #         if relpos is not None:
    #             relpos = relpos[:, -1:, :, :]
    #     _ret = self.decoder(inputs_embeds=embs, attention_mask=padmask,
    #                  encoder_hidden_states=enc,
    #                  encoder_attention_mask=encmask, use_cache=True,
    #                  past_key_value_states=cache,
    #                  relpos=relpos)
    #     ret = _ret[0]
    #     c = ret
    #     cache = _ret[1]
    #     logits = self.out(c)
    #     return logits, cache


