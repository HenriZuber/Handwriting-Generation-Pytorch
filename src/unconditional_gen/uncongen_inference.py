import sys
sys.path.insert(0,'..')
import numpy as np

import torch

from unconditional_gen import uncongen_utils as utils
import config as args

def generate_unconditional(random_seed=args.RANDOM_SEED):

    uncongen_model = torch.load(args.UNCON_GEN_MODEL_PATH)

    np.random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)

    new_stroke = utils.compute_stroke(uncongen_model)
    new_stroke = new_stroke.squeeze().data
    new_stroke = np.array(new_stroke)

    return new_stroke

def main(random_seed=args.RANDOM_SEED):

    stroke = generate_unconditional(random_seed=args.RANDOM_SEED)
    utils.save_stroke(stroke, args.IMAGES_INF_PATH + 'uncon_gen_inf_%s.png'%random_seed)
