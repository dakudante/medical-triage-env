from rl.train_ppo import build_argparser, train

if __name__ == "__main__":
    train(build_argparser().parse_args())
