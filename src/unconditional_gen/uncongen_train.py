import sys
sys.path.insert(0,'..')
import numpy as np

import torch

from unconditional_gen import uncongen_utils as utils
import config as args

# Making it reproducible

np.random.seed(args.RANDOM_SEED)
torch.cuda.manual_seed_all(args.RANDOM_SEED)
torch.manual_seed(args.RANDOM_SEED)

class Uncon_gen_model(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.lstm = torch.nn.LSTM(in_channels,args.HIDDEN_LAYER_SIZE,args.N_LAYER,batch_first=True)
        self.mix_dens_net = torch.nn.Linear(args.HIDDEN_LAYER_SIZE,out_channels)

    def state_values_inits(self):

        cell = torch.zeros(args.N_LAYER, args.BATCH_SIZE, args.HIDDEN_LAYER_SIZE)
        hidden_layer = torch.zeros(args.N_LAYER, args.BATCH_SIZE, args.HIDDEN_LAYER_SIZE)

        return hidden_layer, cell


    def forward(self, ins, state_value_hidden):

        output, state_value_hid = self.lstm(ins, state_value_hidden)
        output = output.contiguous().view(-1,output.size(-1))
        output = self.mix_dens_net(output)

        m1, m2, log_s1, log_s2, rho, pred_pi, pred_e = output.split(args.GAUSSIAN_MIX_NUM,dim=1)
        rho = torch.tanh(rho)
        mix_dens_net_dict = {'m1':m1, 'm2':m2, 'log_s1':log_s1, 'log_s2':log_s2, 'rho':rho, 'pred_pi':pred_pi}

        return (state_value_hid, mix_dens_net_dict, pred_e)



def train_s(X,Y,uncon_gen_model,optimizer):

    state_value_hidden = uncon_gen_model.state_values_inits()
    (_ ,mix_dens_net_dict,pred_e) = uncon_gen_model(X,state_value_hidden)
    Y_true = Y.view(-1,3).contiguous()
    end_stroke, x1, x2 = Y_true.split(1, dim=1)

    x1 = x1.expand_as(mix_dens_net_dict['m1'])
    x2 = x2.expand_as(mix_dens_net_dict['m2'])

    loss = utils.mind_blowing_loss(end_stroke,x1,x2,mix_dens_net_dict,pred_e)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(uncon_gen_model.parameters(), args.GRAD_CLIP)

    optimizer.step()

    return loss


def main():
    loss = "..."

    strokes = utils.get_data()
    in_channels = strokes[0].shape[-1]
    out_channels = 1 + 6*args.GAUSSIAN_MIX_NUM

    uncon_gen_model = Uncon_gen_model(in_channels,out_channels)

    optimizer = torch.optim.Adam(uncon_gen_model.parameters(), lr=args.LR)

    print("Training, please wait ...")
    print()

    for epoch in range(args.EPOCH+1):

        for batch in range(args.BATCH_PER_EPOCH):
            X,Y = utils.get_batch(strokes)

            uncon_gen_model.train()
            optimizer.zero_grad()
            loss = train_s(X,Y,uncon_gen_model,optimizer)

        print("info : Epoch %s ; loss : %s"%(epoch,np.array(loss.data)))

        if epoch % args.VERBOSE_EVERY == 0:

            torch.save(uncon_gen_model,args.UNCON_GEN_MODEL_PATH)
            new_stroke = utils.compute_stroke(uncon_gen_model)
            utils.save_stroke(new_stroke,args.IMAGES_TRAIN_UNCON_PATH+'uncon_%s.png'%epoch)








