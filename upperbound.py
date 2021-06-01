import torch

from sac import PolicyNet

lr_pi = 0.0005
epoch = 1000
ckpt_path = 'checkpoints/ep-%d.pt' % epoch


if __name__ == '__main__':

    model = PolicyNet(lr_pi)
    ckpt = torch.load(ckpt_path)
    print('Load weights from %s' % ckpt_path)
    model.load_state_dict(ckpt['policy'])
