from main_torch_latest import all_data_loader, test
import utils
import main_torch_latest
import argparse
import os

parser = argparse.ArgumentParser(description="test")
parser.add_argument("epoch", type=int, help="epoch that you want to evaluate")
parser.add_argument("save_weights", type=str, help="location for saving/loading weights")
parser.add_argument("dataset_location", type=str, help="The location for the samples")
args = parser.parse_args()
main_torch_latest.save_weights = args.save_weights
loc = args.save_weights
utils.uid = args.save_weights
utils.set_dataset_location("", "", args.dataset_location)
_, _, test_loader = all_data_loader()
test(test_loader, args.epoch, showAttn=True)
os.system("./test.sh " + str(f"pred_{loc}_logs/test_predict_seq.{args.epoch}.log"))
