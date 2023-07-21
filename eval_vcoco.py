import sys
sys.path.append('/home/leiting/v-coco')
from vsrl_eval import VCOCOeval
import utils

if __name__ == "__main__":
    vsrl_annot_file = "/home/leiting/v-coco/data/vcoco/vcoco_test.json"
    coco_file = "/home/leiting/v-coco/data/instances_vcoco_all_2014.json"
    split_file = "/home/leiting/v-coco/data/splits/vcoco_test.ids"

    # Change this line to match the path of your cached file
    det_file = "/home/leiting/ADA-CM/matlab/" + sys.argv[1] # vcoco-injector/10/cache.pkl

    print(f"Loading cached results from {det_file}.")
    vcocoeval = VCOCOeval(vsrl_annot_file, coco_file, split_file)
    vcocoeval._do_eval(det_file, ovr_thresh=0.5)