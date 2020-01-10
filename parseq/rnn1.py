"""
Seq2seq based: Neural Machine Translation by Jointly Learning to Align and Translate
https://arxiv.org/abs/1409.0473
"""
import random
from typing import List, Union, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import qelos as q

from parseq.decoding import merge_dicts
from parseq.eval import Metric, Loss
from parseq.states import TrainableDecodableState, State


class SeqDecoder(torch.nn.Module):
    def __init__(self, model, eval:List[Union[Metric, Loss]]=tuple(), mode="tf", out_vocab=None, **kw):
        super(SeqDecoder, self).__init__(**kw)
        self.model = model
        self._metrics = eval
        self.mode = mode
        self.out_vocab = out_vocab
        self.register_buffer("out_mapper", None)
        self.register_buffer("out_mask", None)

        id_mapper = torch.arange(out_vocab.number_of_ids())
        for id in out_vocab.rare_ids:
            id_mapper[id] = out_vocab[out_vocab.unktoken]
        self.register_buffer("out_mapper", id_mapper)

        out_mask = torch.ones(out_vocab.number_of_ids())
        for id in out_vocab.rare_ids:
            out_mask[id] = 0
        self.register_buffer("out_mask", out_mask)

    def forward(self, x:TrainableDecodableState) -> Tuple[Dict, State]:
        mask = x.inp_tensor != 0
        src_lengths = mask.sum(-1)
        inptensor = x.inp_tensor
        goldtensor = x.gold_tensor
        goldtensor = self.out_mapper[goldtensor]
        y = self.model(inptensor, src_lengths, goldtensor, teacher_forcing_ratio=1. if self.mode == "tf" else 0.)
        y = y.transpose(0, 1)
        y = y + torch.log(self.out_mask[None, None, :])
        y = y[:, 1:]

        outprobs = y
        _, predactions = outprobs.max(-1)

        golds = x.get_gold()
        golds = golds[:, 1:]

        if self.mode == "tf":
            traingold = self.out_mapper[golds]
            traingold = golds
            loss = self._metrics[0](outprobs, predactions, traingold, x)
            metrics = [metric(outprobs, predactions, golds, x) for metric in self._metrics[1:]]
            metrics += [loss]
        metrics = merge_dicts(*metrics)
        return metrics, x


class Seq2Seq(nn.Module):
    """
    Seq2seq class
    """
    def __init__(self, encoder, decoder, name):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.name = name

    def forward(self, src_tokens, src_lengths, trg_tokens, teacher_forcing_ratio=0.5):
        """
        Run the forward pass for an encoder-decoder model.

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(src_len, batch)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            trg_tokens (LongTensor): tokens in the target language of shape
                `(tgt_len, batch)`, for teacher forcing
            teacher_forcing_ratio (float): teacher forcing probability

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - attention scores of shape `(batch, trg_len, src_len)`
        """
        encoder_out = self.encoder(src_tokens, mask=src_tokens!=0)
        decoder_out = self.decoder(trg_tokens, encoder_out,
                                   src_tokens=src_tokens,
                                   teacher_forcing_ratio=teacher_forcing_ratio)
        return decoder_out


class Encoder(nn.Module):
    """Encoder"""
    def __init__(self, emb, enc, hdim, outdim, dropout=0):
        super().__init__()
        self.emb = emb
        self.enc = enc
        self.linear_out = nn.Linear(hdim, outdim)
        self.dropout = dropout

    def forward(self, src_tokens, mask, **kwargs):
        """
        Forward Encoder

        Args:
            src_tokens (LongTensor): (batch, src_len)
            src_lengths (LongTensor): (batch)

        Returns:
            x (LongTensor): (src_len, batch, hidden_size * num_directions)
            hidden (LongTensor): (batch, enc_hid_dim)
        """
        x = self.emb(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)  # (src_len, batch, embed_dim)

        y, hidden = self.enc(x, mask)

        y = F.dropout(y, p=self.dropout, training=self.training)

        hidden = torch.tanh(self.linear_out(hidden[-1][0]))  # (batch, enc_hid_dim)

        return y.transpose(0, 1), hidden        # (seqlen, batsize, encdim), (batsize, compressed_encdim)


class Attention(nn.Module):
    """Attention"""
    def __init__(self, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.att = q.Attention(q.SimpleFwdAttComp(dec_hid_dim, enc_hid_dim*2, dec_hid_dim), dropout=dropout)

    def forward(self, hidden, encoder_outputs, mask):
        alphas, summary, scores = self.att(hidden, encoder_outputs.transpose(0, 1), mask)
        return alphas, summary


class Decoder(nn.Module):
    """Decoder"""
    def __init__(self, outlin, vocabulary, device, embed_dim=256, hidden_size=512,
                 num_layers=2, dropout=0.5, max_positions=50):
        super().__init__()
        num_layers = 1      # TODO
        self.vocabulary = vocabulary
        self.hidden_size = hidden_size
        self.need_attn = True
        self.output_dim = max(vocabulary.stoi.values()) + 1
        self.pad_id = vocabulary.stoi[vocabulary.padtoken]
        self.sos_idx = vocabulary.stoi[vocabulary.starttoken]
        self.eos_idx = vocabulary.stoi[vocabulary.endtoken]
        self.dropout = dropout
        self.max_positions = max_positions
        self.device = device

        # suppose encoder and decoder have same hidden size
        self.attention = Attention(hidden_size, hidden_size, dropout=min(dropout, 0.1))
        self.embed_tokens = Embedding(self.output_dim, embed_dim, self.pad_id)

        self.rnn = GRU(
            input_size=(hidden_size * 2) + embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.linear_out = outlin

    def _decoder_step(self, input, hidden, encoder_outputs, mask):
        input = input.unsqueeze(0) # (1, batch)

        x = self.embed_tokens(input) # (1, batch, emb_dim)
        x = F.dropout(x, p=self.dropout, training=self.training)

        attn, summ = self.attention(hidden, encoder_outputs, mask) # (batch, src_len)
        summ = summ[None, :, :]

        rnn_input = torch.cat((x, summ), dim=2) # (1, batch, 2 * enc_hid_dim + embed_dim)

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output: (1, batch, dec_hid_dim)
        # hidden: (1, batch, dec_hid_dim)

        output = output.squeeze(0)
        weighted = summ.squeeze(0)

        x = torch.cat((output, weighted), dim=1)
        output = self.linear_out(x) # (batch, output_dim)

        return output, hidden.squeeze(0), attn.squeeze(1)

    def forward(self, trg_tokens, encoder_out, **kwargs):
        """
        Forward Decoder

        Args:
            trg_tokens (LongTensor): (trg_len, batch)
            Tuple (encoder_out):
                encoder_out (LongTensor): (src_len, batch, 2 * hidden_size)
                hidden (LongTensor): (batch, enc_hid_dim)
            src_tokens (LongTensor): (src_len, batch)

        Returns:
            outputs (LongTensor): (max_len, batch, output_dim)
            attentions (LongTensor): (max_len, batch, src_len)
        """
        encoder_out, hidden = encoder_out
        src_tokens = kwargs.get('src_tokens', '')
        teacher_ratio = kwargs.get('teacher_forcing_ratio', '')
        src_tokens = src_tokens.t()
        batch = src_tokens.shape[1]

        if trg_tokens is None:
            teacher_ratio = 0.
            inference = True
            trg_tokens = torch.zeros((self.max_positions, batch)).long().\
                                                                  fill_(self.sos_idx).\
                                                                  to(encoder_out.device)
        else:
            trg_tokens = trg_tokens.t()
            inference = False

        max_len = trg_tokens.shape[0]

        # initialize tensors to store the outputs and attentions
        outputs = torch.zeros(max_len, batch, self.output_dim).to(encoder_out.device)
        attentions = torch.zeros(max_len, batch, src_tokens.shape[0]).to(encoder_out.device)

        # prepare decoder input(<sos> token)
        input = trg_tokens[0, :]

        mask = (src_tokens != self.pad_id).permute(1, 0) # (batch, src_len)

        for i in range(1, max_len):

            # forward through decoder using inout, encoder hidden, encoder outputs and mask
            # get predictions, hidden state and attentions
            output, hidden, attention = self._decoder_step(input, hidden, encoder_out, mask)

            # save predictions for position i
            outputs[i] = output

            # save attention for position i
            attentions[i] = attention

            # if teacher forcing
            #   use actual next token as input for next position
            # else
            #   use highest predicted token
            input = trg_tokens[i] if random.random() < teacher_ratio else output.argmax(1)

            # if inference is enabled and highest predicted token is <eos> then stop
            # and return everything till position i
            if inference and input.item() == self.eos_idx:
                return outputs[:i] # , attentions[:i]

        return outputs # , attentions

def Embedding(num_embeddings, embedding_dim, padding_idx):
    """Embedding layer"""
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def Linear(in_features, out_features, bias=True):
    """Linear layer"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m

def GRU(input_size, hidden_size, **kwargs):
    """GRU layer"""
    m = nn.GRU(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m