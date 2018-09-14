import sys
sys.path.insert(0,'..')
import numpy as np

import torch

from conditional_gen import congen_utils as utils
import config as args

def generate_conditional(random_seed=args.RANDOM_SEED, text=args.TEXT_TO_HANDWRITE):

    congen_model = torch.load(args.CON_GEN_MODEL_PATH)
    _,_,_,two_way_dict = utils.get_data()

    np.random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)

    new_stroke = utils.compute_stroke(congen_model,two_way_dict, text=text)
    new_stroke = new_stroke.squeeze().data
    new_stroke = np.array(new_stroke)

    return new_stroke

def main(random_seed=args.RANDOM_SEED):

    stroke = generate_conditional(random_seed=args.RANDOM_SEED)
    utils.save_stroke(stroke, args.IMAGES_INF_PATH + 'con_gen_inf_%s.png'%random_seed)
