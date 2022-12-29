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
        "--model", type = str, default = "LeNet", help = "the model to use"
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

    args = parser.parse_args()

    args.backdoor_adjustment = args.model.lower() in ['lenet']

    return args
