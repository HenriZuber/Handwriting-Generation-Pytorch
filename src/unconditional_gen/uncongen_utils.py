import sys
sys.path.insert(0,'..')
import numpy as np
import torch

import config as args
import utils as lb_uts


np.random.seed(args.RANDOM_SEED)

def get_data():

    strokes = np.load(args.STROKES_PATH,encoding='latin1')
    strk = []
    for stroke in strokes :
        if len(stroke) > args.SUBSTROKE_LENGTH +1 :
            strk.append(stroke)
    np.random.shuffle(strk)

    return strk

def mind_blowing_loss(end_stroke,x1,x2,mix_dens_net_dict,pred_e):

    # renaming for quality of life reasons
    m1 = mix_dens_net_dict['m1']
    m2 = mix_dens_net_dict['m2']
    log_s1 = mix_dens_net_dict['log_s1']
    log_s2 = mix_dens_net_dict['log_s2']
    rho = mix_dens_net_dict['rho']
    pred_pi = mix_dens_net_dict['pred_pi']

    # following that awesome paper (alex graves, arxiv 1308), the loss is gonna be two-fold
    # a 2D nll and a basic class loss
    # brace yourselves

    # Binary Cross Entropy on logits, easy to understand
    easy_class_loss = torch.nn.BCEWithLogitsLoss()(pred_e,end_stroke)

    # Ok, the fun begins now
    # Let's first get Z (again check the paper, don't ask)

    Z =(-2)*(x1-m1)*(x2-m2)*rho/(log_s1.exp()*log_s2.exp()) + ((x1-m1)/log_s1.exp()).pow(2) + ((x2-m2)/log_s2.exp()).pow(2)

    from_z = (-0.5)*Z/(1-rho.pow(2))
    from_sigmas =  -torch.log(2*float(np.pi)*log_s1.exp()*log_s2.exp()*torch.sqrt(1-rho.pow(2)))
    from_pi = torch.nn.functional.log_softmax(pred_pi)

    wtheck_loss = -torch.distributions.utils.log_sum_exp(from_z+from_sigmas+from_pi, keepdim=True).mean()
    #wtheck_loss = -torch.logsumexp(from_z+from_sigmas+from_pi, dim=-1).mean()
    loss = wtheck_loss + easy_class_loss

    return loss


def get_batch(strokes):

    size = len(strokes)

    X_batch = torch.zeros((args.BATCH_SIZE,args.SUBSTROKE_LENGTH,3))
    Y_batch = torch.zeros((args.BATCH_SIZE,args.SUBSTROKE_LENGTH,3))

    list_idx = list(range(size))
    np.random.shuffle(list_idx)

    for batch_id in range(args.BATCH_SIZE):
        stroke = strokes[list_idx[batch_id]]
        first = np.random.randint(0,len(stroke)-args.SUBSTROKE_LENGTH)
        sub_stroke_x = stroke[first: first+args.SUBSTROKE_LENGTH,:]
        sub_stroke_y = stroke[first+1: first+args.SUBSTROKE_LENGTH+1,:]
        sub_stroke_x = torch.Tensor(sub_stroke_x)
        sub_stroke_y = torch.Tensor(sub_stroke_y)
        X_batch[batch_id] = sub_stroke_x
        Y_batch[batch_id] = sub_stroke_y

    return X_batch, Y_batch


def compute_stroke(uncon_gen_model):

    state_value_hidden = uncon_gen_model.state_values_inits()
    start_new_stroke = torch.Tensor([[[1,0,0]]])
    (H,C) = state_value_hidden
    H = H[:,1,:]
    H = H.reshape(len(H),1,len(H[0]))
    C = C[:,1,:]
    C = C.reshape(len(C),1,len(C[0]))
    state_value_hidden = (H,C)
    new_stroke = torch.zeros(args.NEW_STROKE_LENGTH,1,3)

    # got 'inspired' by biased sampling part of the paper
    # even if it was for conditional generation

    for i in range(args.NEW_STROKE_LENGTH):

        (state_value_hidden, mix_dens_net_dict, pred_e) = uncon_gen_model(start_new_stroke,state_value_hidden)
        # Quality of life renaming
        m1 = mix_dens_net_dict['m1']
        m2 = mix_dens_net_dict['m2']
        log_s1 = mix_dens_net_dict['log_s1']
        log_s2 = mix_dens_net_dict['log_s2']
        rho = mix_dens_net_dict['rho']
        pred_pi = mix_dens_net_dict['pred_pi']

        idx = torch.multinomial(torch.nn.functional.softmax(pred_pi*(1+args.PROBABILITY_BIAS)),1)

        m1 = m1.gather(1,idx)
        m2 = m2.gather(1,idx)
        log_s1 = log_s1.gather(1,idx) - args.PROBABILITY_BIAS
        log_s2 = log_s2.gather(1,idx) - args.PROBABILITY_BIAS
        rho = rho.gather(1,idx)

        # you want new x1, x2 from x = mu + sigma*epsilon with epislon from normal dist

        e1 = np.random.normal(0,1,1)
        e1 = torch.Tensor(e1).view(1,-1)
        e2 = np.random.normal(0,1,1)
        e2 = torch.Tensor(e2).view(1,-1)


        x1 = m1 + log_s1.exp()*e1
        x2 = m2 + log_s2.exp()*(e2*(1-rho.pow(2)).pow(1/2) + rho*e1)

        end_stroke = np.random.binomial(1,torch.sigmoid(pred_e).data)
        end_stroke = torch.Tensor(end_stroke)
        #end_stroke = torch.bernoulli(torch.sigmoid(pred_e))

        start_new_stroke = torch.cat([end_stroke,x1,x2],-1).view(1,1,3)
        new_stroke[i] = start_new_stroke
    return new_stroke


def save_stroke(new_stroke,name):

    print()
    new_stroke = new_stroke.squeeze().data
    new_stroke = np.array(new_stroke)
    lb_uts.plot_stroke(new_stroke, name)




