# Basic things work for legacy CLI, we need to make new thinkgs.
import argparse
from arrakis.src.core_arrakis.activation_cache import *
import torch

def main():
    print("Hello World")

RED = "\33[91m"
BLUE = "\33[94m"
GREEN = "\033[32m"
YELLOW = "\033[93m"
PURPLE = '\033[0;35m' 
CYAN = "\033[36m"
END = "\033[0m"

parser = argparse.ArgumentParser(
    prog='arrakis-cli',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=f"""
    {BLUE}               
Welcome to Arrakis CLI, a tool to interact with the Arrakis library.
--------------------------------------------  
                           __     _      
 ___ _  ____  ____ ___ _  / /__  (_)  ___
/ _ `/ / __/ / __// _ `/ /  '_/ / /  (_-<
\_,_/ /_/   /_/   \_,_/ /_/\_\ /_/  /___/
                                         
-------------------------------------------                                         
{YELLOW}
Using arrakis is easy, just go through the commands and call it to create a `HookedAutoConfig`, which in turn invoke a `HookedAutoModel`.{END}

""",)

parser.add_argument("--model_name", type=str, help="Name of the model to be fetched from HF", choices=MODELS_THAT_ARE_SUPPORTED, required=True)
# vocab_size=50256, hidden_size=4, intermediate_size=1, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2
parser.add_argument("--vocab_size", type=int, default=50256, help="Vocab size of the model")
parser.add_argument("--hidden_size", type=int, default=4, help="Hidden size of the model")
parser.add_argument("--intermediate_size", type=int, default=1, help="Intermediate size of the model")
parser.add_argument("--num_hidden_layers", type=int, default=3, help="Number of hidden layers in the model")
parser.add_argument("--num_attention_heads", type=int, default=4, help="Number of attention heads in the model")
parser.add_argument("--num_key_value_heads", type=int, default=2, help="Number of key value heads in the model")

args = parser.parse_args()
cfg = HookedAutoConfig(args.model_name, vocab_size=args.vocab_size, hidden_size=args.hidden_size, intermediate_size=args.intermediate_size, num_hidden_layers=args.num_hidden_layers, num_attention_heads=args.num_attention_heads, num_key_value_heads=args.num_key_value_heads)
model = HookedAutoModel(cfg)

input_ids = torch.randint(0, args.vocab_size, (1, 50))
output,activations = model(input_ids)

print(activations.keys())