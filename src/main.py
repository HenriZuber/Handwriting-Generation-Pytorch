from unconditional_gen import uncongen_inference, uncongen_train
from conditional_gen import congen_inference, congen_train
import config as args

if __name__ == "__main__":

    if args.CONDITIONAL:
        if args.INFERENCE:
            congen_inference.main()
        else:
            congen_train.main()
    else:
        if args.INFERENCE:
            uncongen_inference.main()
        else:
            uncongen_train.main()
