from rl.evaluate_ppo import build_argparser, evaluate
from pathlib import Path
import json

if __name__ == "__main__":
    args = build_argparser().parse_args()
    print(json.dumps(evaluate(Path(args.checkpoint), args.episodes, args.seed), indent=2))
