from tqdm import tqdm
from neural_chat.gpt2 import GPT2
from neural_chat.craigslist import Craigslist
import argparse
import torch
import logging

logging.basicConfig(level=logging.DEBUG, filename="app.log", filemode='w')
logger=logging.getLogger('my_logger')


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", default="./train_output/best_model")
parser.add_argument("--input-file", default="/opt/ml/chai-naacl-2022/data/train_google.json")
parser.add_argument("--output-file", default="sentences.pkl")
parser.add_argument("--num-outputs", type=int, default=5)
args = parser.parse_args()

print("Opening input file...")
cg = Craigslist(args.input_file)

print("Loading GPT...")
gpt = GPT2(args.checkpoint_dir, device="cuda")
events = list(cg.events.values())
data = {}
for i, ev in tqdm(enumerate(events)):
    try:
        evs = ev.get_events()
        dialog = [ev.data["price"].utterance for ev in evs]
        agents = [ev.agent for ev in evs]
        val = gpt.generate(
            ev.scenario, dialog[:-1], agents[:-1], num_outputs=args.num_outputs
            # 왜 -1까지 인지 이해가 안돼.. 마지막만 5개 candidate generate해서 return하려는게 목적인가?
        )
        data[ev.event_id] = val
    except Exception as e:
        logger.error(f"index {i} : {e}")

print("Saving file")
with open(args.output_file, "wb") as f:
    torch.save(data, f)
