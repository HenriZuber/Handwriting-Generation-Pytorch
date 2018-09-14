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

    with open(args.TEXTS_PATH) as text_file:
        texts = text_file.readlines()
    txt = []

    both = zip(strokes,texts)

    lens = []
    for (stroke,text) in both :
        if len(stroke) > args.SUBSTROKE_LENGTH +1 :
            txt.append(text)
            strk.append(stroke)
            lens.append(len(stroke)/len(text))

    number_char = len(np.unique(list(" ".join(txt))))

    two_way_dict = {}
    for idx, unique_char in enumerate(np.unique(list(" ".join(txt)))):
        two_way_dict[unique_char]=idx
        two_way_dict['id_'+str(idx)]=unique_char

    two_way_dict['mean'] = np.mean(lens)

    encoded_texts = []
    for l in txt:
        l = list(l)
        encoded_line = []
        for unique_char in l:
            encoded_char = np.zeros(number_char)
            encoded_char[two_way_dict[unique_char]] = 1
            encoded_line.append(encoded_char)
        encoded_texts.append(encoded_line)

    res = list(zip(strk,txt,encoded_texts))
    np.random.shuffle(res)
    for i, sub_res in enumerate(res):
        sub_strk, sub_txt, sub_enc = sub_res
        strk[i] = sub_strk
        txt[i] = sub_txt
        encoded_texts[i] = sub_enc

    return strk, txt, encoded_texts, two_way_dict


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

    # might have some problems with log_sum_exp (wth_loss was negative at some point)

    wtheck_loss = -torch.distributions.utils.log_sum_exp(from_z+from_sigmas+from_pi, keepdim=True).mean()
    #wtheck_loss = -torch.logsumexp(from_z+from_sigmas+from_pi, dim=-1).mean()

    loss = wtheck_loss + easy_class_loss

    return loss


def get_batch(strokes,encoded_texts,two_way_dict):

    size = len(strokes)

    X_batch = torch.zeros((args.BATCH_SIZE,args.SUBSTROKE_LENGTH,3))
    Y_batch = torch.zeros((args.BATCH_SIZE,args.SUBSTROKE_LENGTH,3))
    dim = int(args.SUBSTROKE_LENGTH//two_way_dict['mean'])
    encoded_batch = torch.zeros((args.BATCH_SIZE,dim,np.shape(encoded_texts[0])[-1]))

    list_idx = list(range(size))
    np.random.shuffle(list_idx)

    for batch_id in range(args.BATCH_SIZE):
        stroke = strokes[list_idx[batch_id]]
        encode = encoded_texts[list_idx[batch_id]]
        encode = torch.Tensor(encode)
        # no more random_starts otherwise it wouldn't really learn characters
        sub_stroke_x = stroke[: args.SUBSTROKE_LENGTH,:]
        sub_stroke_y = stroke[1: args.SUBSTROKE_LENGTH+1,:]
        sub_stroke_x = torch.Tensor(sub_stroke_x)
        sub_stroke_y = torch.Tensor(sub_stroke_y)
        X_batch[batch_id] = sub_stroke_x
        Y_batch[batch_id] = sub_stroke_y
        mid = np.shape(encoded_texts)[0]
        encoded_batch[batch_id,:mid,:] = encode[:dim,:]

    return X_batch, Y_batch, encoded_batch


def compute_stroke(con_gen_model,two_way_dict,text='how about example'):

    encoded_text = torch.zeros((1,len(text),(len(two_way_dict.keys())-1)/2))

    for idx, unique_char in enumerate(text):
        encoded_text[0,idx, two_way_dict[unique_char]] = 1

    states_value_hidden = con_gen_model.state_values_inits()
    start_new_stroke = torch.Tensor([[[1,0,0]]])

    states_value_hidden = list(states_value_hidden)
    for i,state_value_hidden in enumerate(states_value_hidden):
        (H,C) = state_value_hidden
        H = H[:,1,:]
        H = H.reshape(len(H),1,len(H[0]))
        C = C[:,1,:]
        C = C.reshape(len(C),1,len(C[0]))
        states_value_hidden[i] = (H,C)
    states_value_hidden = tuple(states_value_hidden)


    full_k = torch.zeros(1,args.WINDOWS_NUM,1)

    stroke_length = int(args.STROKE_LENGTH_COEFF*len(text)*two_way_dict['mean'])
    #new_stroke = torch.zeros(args.NEW_STROKE_LENGTH,1,3)
    new_stroke = torch.zeros(stroke_length,1,3)

    # got 'inspired' by biased sampling part of the paper

    #for i in range(args.NEW_STROKE_LENGTH):
    for i in range(stroke_length):

        (states_value_hidden, mix_dens_net_dict, pred_e) = con_gen_model(start_new_stroke, encoded_text, states_value_hidden,K_cumsum=full_k)
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

        start_new_stroke = torch.cat([end_stroke,x1,x2],-1).view(1,1,3)
        full_k = con_gen_model.K
        new_stroke[i] = start_new_stroke

    return new_stroke


def save_stroke(new_stroke,name):

    print()
    new_stroke = new_stroke.squeeze().data
    new_stroke = np.array(new_stroke)
    lb_uts.plot_stroke(new_stroke, name)




