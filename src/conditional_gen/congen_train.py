import sys
sys.path.insert(0,'..')
import numpy as np

import torch

from conditional_gen import congen_utils as utils
import config as args

# Making it reproducible

np.random.seed(args.RANDOM_SEED)
torch.cuda.manual_seed_all(args.RANDOM_SEED)
torch.manual_seed(args.RANDOM_SEED)

class Con_gen_model(torch.nn.Module):

    def __init__(self, in_channels, out_channels, windows_out_channels, encode_dim):
        super().__init__()

        self.first_lstm = torch.nn.LSTM(in_channels,args.HIDDEN_LAYER_SIZE,batch_first=True)
        self.second_lstm = torch.nn.LSTM(in_channels+args.HIDDEN_LAYER_SIZE+encode_dim,args.HIDDEN_LAYER_SIZE,batch_first=True)
        self.mix_dens_net = torch.nn.Linear(args.HIDDEN_LAYER_SIZE,out_channels)
        self.window_net = torch.nn.Linear(args.HIDDEN_LAYER_SIZE,windows_out_channels)
        self.K = 0

    def state_values_inits(self):

        cell = torch.zeros(1, args.BATCH_SIZE, args.HIDDEN_LAYER_SIZE)
        hidden_layer = torch.zeros(1, args.BATCH_SIZE, args.HIDDEN_LAYER_SIZE)
        res = ((hidden_layer,cell),(hidden_layer,cell))

        return res


    def forward(self, ins,encoded, states_value_hidden,K_cumsum=0):

        (state_value_hidden_0, state_value_hidden_1) = states_value_hidden
        output_0, state_value_hid_0 = self.first_lstm(ins, state_value_hidden_0)
        output_window = self.window_net(output_0)
        alpha, beta, K = torch.exp(output_window).unsqueeze(-1).split(args.WINDOWS_NUM,-2)

        if K_cumsum is 0:
            full_K = K.cumsum(1)
        else :
            full_K = K_cumsum.unsqueeze(1) + K

        window_weight = (torch.exp(-beta * (full_K - torch.arange(0,encoded.size(1)).view(1,1,1,-1) ).pow(2)) * alpha).sum(-2)
        window_output = torch.matmul(window_weight,encoded)

        output = torch.cat([output_0,window_output,ins],-1)
        output, state_value_hid_1 = self.second_lstm(output,state_value_hidden_1)

        output = output.contiguous().view(-1,output.size(-1))
        output = self.mix_dens_net(output)

        m1, m2, log_s1, log_s2, rho, pred_pi, pred_e = output.split(args.GAUSSIAN_MIX_NUM,dim=1)
        rho = torch.tanh(rho)
        mix_dens_net_dict = {'m1':m1, 'm2':m2, 'log_s1':log_s1, 'log_s2':log_s2, 'rho':rho, 'pred_pi':pred_pi}

        self.K = full_K[:, -1,:,:]
        states_value_hid = (state_value_hid_0,state_value_hid_1)

        return (states_value_hid, mix_dens_net_dict, pred_e)



def train_s(X,Y,encoded,con_gen_model,optimizer):

    states_value_hidden = con_gen_model.state_values_inits()
    (_ ,mix_dens_net_dict,pred_e) = con_gen_model(X,encoded,states_value_hidden)
    Y_true = Y.view(-1,3).contiguous()
    end_stroke, x1, x2 = Y_true.split(1, dim=1)

    x1 = x1.expand_as(mix_dens_net_dict['m1'])
    x2 = x2.expand_as(mix_dens_net_dict['m2'])

    loss = utils.mind_blowing_loss(end_stroke,x1,x2,mix_dens_net_dict,pred_e)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(con_gen_model.parameters(), args.GRAD_CLIP)

    optimizer.step()

    return loss


def main():
    loss = "..."

    strokes,texts,encoded_texts,two_way_dict = utils.get_data()
    in_channels = strokes[0].shape[-1]
    encoded_channels = np.shape(encoded_texts[0])[-1]
    out_channels = 1 + 6*args.GAUSSIAN_MIX_NUM
    windows_out_channels = 3*args.WINDOWS_NUM

    con_gen_model = Con_gen_model(in_channels,out_channels,windows_out_channels, encoded_channels)

    optimizer = torch.optim.Adam(con_gen_model.parameters(), lr=args.LR)

    print("Training, please wait ...")
    print()

    for epoch in range(args.EPOCH+1):

        for batch in range(args.BATCH_PER_EPOCH):
            X,Y,encoded = utils.get_batch(strokes,encoded_texts,two_way_dict)

            con_gen_model.train()
            optimizer.zero_grad()
            loss = train_s(X,Y,encoded,con_gen_model,optimizer)

        print("info : Epoch %s ; loss : %s"%(epoch,np.array(loss.data)))

        if epoch % args.VERBOSE_EVERY == 0:

            torch.save(con_gen_model,args.CON_GEN_MODEL_PATH)
            new_stroke = utils.compute_stroke(con_gen_model,two_way_dict)
            utils.save_stroke(new_stroke,args.IMAGES_TRAIN_CON_PATH+'con_%s.png'%epoch)








