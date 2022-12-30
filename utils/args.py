import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", type = bool, default=True, help = "whether to train the model"
    )
    parser.add_argument(
        "--test", dest='train', action='store_false', help = "only test the model"
    )
    parser.add_argument(
        "--n_epoch", type = int, default=3, help = "number of epochs"
    )
    parser.add_argument(
        "--model", type = str, default = "LeNet", help = "the model to use"
    )
    parser.add_argument(
        "--change_col", action='store_true', help = "whether to change the color of the image"
    )
    parser.add_argument(
        "--bs", type = int, default = 16, help = "batch size"
    )
    parser.add_argument(
        "--lr", type = float, default=1e-4, help = "learning rate"
    )
    parser.add_argument(
        "--channel", type = int, default=3, help = "the number of channel of the images fed into the net"
    )
    parser.add_argument(
        "--out_path", type = str, default="./out", help = "the path where the model outputs"
    )
    parser.add_argument(
        "--root_path", type = str, default='.', help = "the root path"
    )
    parser.add_argument(
        "--reweight", default = False, action = "store_true", help= "whether use weight in CrossEntropy."
    )
    parser.add_argument(
        "--trainset", default="all_train", type = str, help="Which train dataset. all_train, train1, train2"
    )
    parser.add_argument(
        "--backdoor_adjustment", default = False, action = "store_true", help= "whether to adjust the backdoor."
    )

    ### for CNC only
    parser.add_argument(
        "--force_train_erm", action="store_true", help="whether to force train the erm model"
    )
    parser.add_argument(
        "--n_epoch_cnc", type = int, default=3, help = "number of epochs for cnc"
    )
    parser.add_argument(
        "--lambda_contrast", type = float, default=0.5, help = "lambda for cnc"
    )




    args = parser.parse_args()

    # print opt
    print('------------ Options -------------')
    for k, v in sorted(vars(args).items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    return args
