import torch


class VIB(torch.nn.Module):
    def __init__(self, indim, outdim=None, train_r=True, **kw):
        super(VIB, self).__init__(**kw)
        self.indim = indim
        self.outdim = outdim if outdim is not None else indim

        self.fc_mu = torch.nn.Linear(self.indim, self.outdim)
        self.fc_logvar = torch.nn.Linear(self.indim, self.outdim)

        self.r_mu = torch.zeros(1, self.outdim)
        self.r_logvar = torch.zeros(1, self.outdim)
        if train_r:
            self.r_mu = torch.nn.Parameter(self.r_mu)
            self.r_logvar = torch.nn.Parameter(self.r_logvar)

    def kl(self, mu, logvar):       # KL[N(mu, logvar) || N(self.r_mu, self.r_logvar)]
        mu_diff = self.r_mu - mu
        r = 0.5 * ((logvar - self.r_logvar).exp().sum(-1)
                   + (mu_diff.pow(2) * (-self.r_logvar).exp()).sum(-1)
                   - self.r_mu.size(1)
                   + self.r_logvar.sum(-1) - logvar.sum(-1))
        return r


    def forward(self, x):
        # compute sample
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # compute KL
        kl = self.kl(mu, logvar)

        # return
        return z, kl


class VIB_seq(VIB):
    """ VIB that is time-shared and supports mask. """
    def __init__(self, indim, outdim=None, train_r=True, **kw):
        super(VIB_seq, self).__init__(indim, outdim=outdim, train_r=train_r, **kw)

    def forward(self, x, mask=None):   # (batsize, seqlen, indim)
        z, kl = super(VIB_seq, self).forward(x)
        if mask is not None:
            z = z * mask.float()[:,:,None]
            kl = kl * mask.float()
            kl = kl.sum(-1) / mask.float().sum(-1)
        else:
            kl = kl.sum(-1) / kl.size(-1)

        return z, kl


def try_VIB():
    vib = VIB(10)
    mu = vib.r_mu.repeat(5, 1)
    mu[1, :] = 1
    logvar = vib.r_logvar.repeat(5, 1)
    kl = vib.kl(mu, logvar)
    print(kl)
    print(list(vib.parameters()))


if __name__ == '__main__':
    try_VIB()
