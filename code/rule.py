import os
from loguru import logger
import utils


def rule_id2rel(dataset):
    rel_path = os.path.join("data/dataset", dataset, "relations.dict")
    rule_path = os.path.join("data/dataset", dataset, "mined_rules.txt")
    outpath = os.path.join("data/dataset", dataset, "mined_rules_trans.txt")
    n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']


    id2rel = dict()
    with open(rel_path, 'r', encoding='utf-8') as f:
        for line in f:
            id, rel = line.strip().split('\t')
            id2rel[int(id)] = rel
            id2rel[int(id)+n_rel] = "inv_"+rel
    logger.info(dataset + " #rel: {}".format(len(id2rel)))
    
    result = []
    with open(rule_path, 'r', encoding='utf-8') as f:
        for line in f:
            rule = line.strip().split(' ')
            r_t = []
            for r in rule:
                r_t.append(id2rel[int(r)])
            result.append(r_t)

    with open(outpath, 'w', encoding='utf-8') as f:
        for r in result:
            f.write("\t".join(r)+"\n")



if __name__ == '__main__':
    datasets = ["FB15k-237", "WN18RR"]
    for d in datasets:
        rule_id2rel(d)
