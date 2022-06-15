import math
import torch


class MHSA(torch.nn.Module):
    DEBUG = False
    def __init__(self, dim, numheads=1):
        super(MHSA, self).__init__()
        self.dim = dim
        self.numheads = numheads
        self.q_proj = torch.nn.Linear(dim, dim)
        self.k_proj = torch.nn.Linear(dim, dim)
        self.v_proj = torch.nn.Linear(dim, dim)

    def forward(self, x):   # (batsize, seqlen, dim)
        q, k, v = self.q_proj(x).view(len(x), len(x[0]), self.numheads, -1), \
                  self.k_proj(x).view(len(x), len(x[0]), self.numheads, -1), \
                  self.v_proj(x).view(len(x), len(x[0]), self.numheads, -1)
        scores = torch.einsum("bshd,bzhd->bszh", q, k)
        scores = scores / math.sqrt(self.dim // self.numheads)
        attention = torch.softmax(scores, 2)
        output = torch.einsum("bzhd,bszh->bshd", v, attention)
        if self.DEBUG:
            return scores, attention, output, q, k, v
        return output


def tst_mhsa():
    dim = 6
    x = torch.randn(2, 3, dim)
    numheads = 2
    mhsa = MHSA(dim, numheads)
    mhsa.DEBUG = True
    scores, attention, output, q, k, v = mhsa(x)
    for b in range(x.size(0)):
        for s in range(x.size(1)):
            for z in range(x.size(1)):
                for h in range(numheads):
                    assert torch.allclose(scores[b, s, z, h],
                                          ((q[b, s, h] * k[b, z, h]).sum() / math.sqrt(dim // numheads)))
    print(scores[0, 0, 0, 0].item(), (q[0, 0, 0] * k[0,0,0]).sum().item()/ math.sqrt(dim // numheads))

    assert torch.allclose(torch.ones(x.size(0), x.size(1), numheads), attention.sum(2))

    for b in range(x.size(0)):
        for s in range(x.size(1)):
            for h in range(numheads):
                v_ref = torch.zeros(x.size(-1)//numheads)
                for z in range(x.size(1)):
                    v_ref += v[b, z, h] * attention[b, s, z, h].item()
                v_pred = output[b, s, h]
                assert torch.allclose(v_pred, v_ref)
    print("done")


class MHSA_Relpos(MHSA):
    def __init__(self, dim, numheads=1, clip=3):
        super(MHSA_Relpos, self).__init__(dim, numheads=numheads)
        self.clip = clip
        self.relposembs = torch.nn.Embedding(self.clip * 2 + 1, dim//numheads)

    def build_skew_matrix(self, seqlen, clip, device):
        m = torch.arange(seqlen, device=device, dtype=torch.long)[None, :]
        m2 = torch.arange(seqlen, device=device, dtype=torch.long)[:, None]
        o = m - m2 + clip
        o = o.clamp(0, clip*2)
        return o

    def build_skew_matrix_(self, seqlen, clip, device):
        # TODO: this could be more efficient
        skewer = torch.zeros(seqlen, seqlen, dtype=torch.long, device=device)
        for i in range(self.clip * 2):
            skewer += torch.triu(torch.ones_like(skewer), self.clip - i)
        return skewer

    def forward(self, x):   # (batsize, seqlen, dim)
        # compare queries to keys to compute content-based attention scores
        q, k, v = self.q_proj(x).view(len(x), len(x[0]), self.numheads, -1), \
                  self.k_proj(x).view(len(x), len(x[0]), self.numheads, -1), \
                  self.v_proj(x).view(len(x), len(x[0]), self.numheads, -1)
        content_scores = torch.einsum("bshd,bzhd->bszh", q, k)

        # compare queries to relative position embeddings to compute position-based attention scores
        k_pos = self.relposembs.weight[:, None, :].repeat(1, self.numheads, 1)
            # k_pos is (numpos, numheads, dim)
        pos_scores = torch.einsum("bshd,phd->bsph", q, k_pos)
        # build skew matrix
        skewer = self.build_skew_matrix(x.size(1), self.clip, x.device)
        skewer = skewer[None, :, :, None].repeat(x.size(0), 1, 1, self.numheads)
        # distribute pos_scores over z
        pos_scores = torch.gather(pos_scores, 2, skewer)
        # compute total score
        scores = content_scores + pos_scores
        scores = scores / math.sqrt(self.dim // self.numheads)

        attention = torch.softmax(scores, 2)
        output = torch.einsum("bzhd,bszh->bshd", v, attention)
        if self.DEBUG:
            return scores, attention, output, q, k, v
        return output


def tst_mhsa_relpos():
    dim = 6
    x = torch.randn(2, 5, dim)
    numheads = 2
    clip = 2
    mhsa = MHSA_Relpos(dim, numheads, clip=clip)
    mhsa.DEBUG = True
    scores, attention, output, q, k, v = mhsa(x)
    for b in range(x.size(0)):
        for s in range(x.size(1)):
            for z in range(x.size(1)):
                for h in range(numheads):
                    predscore = scores[b, s, z, h]
                    truescore = (q[b, s, h] * (k[b, z, h] + mhsa.relposembs.weight[clip+min(clip, max(-clip, (z-s)))])).sum()
                    truescore /= math.sqrt(dim // numheads)
                    assert torch.allclose(predscore, truescore)
    print(predscore.item(), truescore.item())


class MHSA_RelposMulvecFast(MHSA_Relpos):
    def forward(self, x):   # (batsize, seqlen, dim)
        q, k, v = self.q_proj(x).view(len(x), len(x[0]), self.numheads, -1), \
                  self.k_proj(x).view(len(x), len(x[0]), self.numheads, -1), \
                  self.v_proj(x).view(len(x), len(x[0]), self.numheads, -1)
        baseprod = torch.einsum("bshd,bzhd->bszhd", q, k)
        # build skew matrix
        skewer = self.build_skew_matrix(x.size(1), self.clip, x.device)
        # distribute relpos vectors  # (pd->szd)
        relposvecs = self.relposembs(skewer)
        relposvecs = relposvecs[None, :, :, None, :]
        scores = (baseprod * relposvecs).sum(-1)
        scores = scores / math.sqrt(self.dim // self.numheads)
        attention = torch.softmax(scores, 2)
        output = torch.einsum("bzhd,bszh->bshd", v, attention)
        if self.DEBUG:
            return scores, attention, output, q, k, v
        return output


def tst_mhsa_mulvec():
    dim = 6
    x = torch.randn(2, 5, dim)
    numheads = 2
    clip = 2
    mhsa = MHSA_RelposMulvecFast(dim, numheads, clip=clip)
    mhsa.DEBUG = True
    scores, attention, output, q, k, v = mhsa(x)
    for b in range(x.size(0)):
        for s in range(x.size(1)):
            for z in range(x.size(1)):
                for h in range(numheads):
                    predscore = scores[b, s, z, h]
                    truescore = (q[b, s, h] * k[b, z, h] * mhsa.relposembs.weight[clip+min(clip, max(-clip, (z-s)))]).sum()
                    truescore /= math.sqrt(dim // numheads)
                    assert torch.allclose(predscore, truescore)
    print(predscore.item(), truescore.item())
    print("Done")


if __name__ == '__main__':
    tst_mhsa()
    tst_mhsa_relpos()
    tst_mhsa_mulvec()