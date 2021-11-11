#!/usr/bin/env python

# outline of mainlines:
# take alignment or alignment directory as input
# sample sites from the alignment and concatenate them
# run fasttree
# run treecluster-max like "guessing game" to reach the desire number of clusters
# designate the highest occupancy species as representative of the cluster
# return the alignment(s) induced to set of representatives.
import sys
import tempfile
from functools import reduce
from itertools import groupby
from os import listdir
from os.path import isfile, join
from subprocess import Popen, PIPE, call

import numpy as np
import treeswift as ts

BINARY_SEARCH_STOP_MULTIPLIER = 0.0001


def readfq(fp):  # this is a generator function
    last = None  # this is a buffer keeping the last unprocessed line
    while True:  # mimic closure; is it a bad idea?
        if not last:  # the first record or a record following a fastq
            for l in fp:  # search for the start of the next record
                if l[0] in '>@':  # fasta/q header line
                    last = l[:-1]  # save this line
                    break
        if not last: break
        name, seqs, last = last[1:].partition(" ")[0], [], None
        for l in fp:  # read the sequence
            if l[0] in '@+>':
                last = l[:-1]
                break
            seqs.append(l[:-1])
        if not last or last[0] != '+':  # this is a fasta record
            yield name, ''.join(seqs), None  # yield a fasta record
            if not last: break
        else:  # this is a fastq record
            seq, leng, seqs = ''.join(seqs), 0, []
            for l in fp:  # read the quality
                seqs.append(l[:-1])
                leng += len(l) - 1
                if leng >= len(seq):  # have read enough quality
                    last = None
                    yield name, seq, ''.join(seqs);  # yield a fastq record
                    break
            if last:  # reach EOF before reading enough quality
                yield name, seq, None  # yield a fasta record instead
                break


def fasta2mat(ref_fp, prot_flag, mask_flag):
    with open(ref_fp) as f:
        mask_translation = str.maketrans('abcdefghijklmnopqrstuvwxyz', '-' * 26)

        if prot_flag:
            invalid_translation = str.maketrans('BJOUXZ', '-' * 6)
        else:
            invalid_translation = str.maketrans('BDEFHIJKLMNOPQRSUVWXYZ', '-' * 22)

        def makeupper(s):
            if mask_flag:
                return s.translate(mask_translation)
            else:
                return s.upper()

        names = []
        refs = []
        for name, seq, qual in readfq(f):
            names.append(name)
            refs.append(np.frombuffer(makeupper(seq).translate(invalid_translation).encode(), dtype='S1'))
    return np.array(names), np.array(refs)


def gap_filter(names, mats, thr=0.95):
    keep = sum(mats != b'-') / len(names) >= thr
    mats = mats[:, keep]
    # remove all gap sequences
    allgapseqs = (mats == b'-').all(axis=1)
    return names[~allgapseqs], mats[~allgapseqs, :]


def subsample_align(nm, mins, size):
    name, mats = nm
    mins = np.random.choice([True, False], size=len(mins))
    scores_1 = sum(mats[mins[name]] != b'-')  # number of nongaps for the least occupant
    scores_2 = sum(mats != b'-')  # number of nongaps for all
    scores_3 = list(range(len(scores_1)))  # break ties randomly
    np.random.shuffle(scores_3)
    order = sorted(zip(scores_1, scores_2, scores_3, range(len(scores_1))), reverse=True)[:size]
    selected = [i for _, _, _, i in order]
    res = np.full((len(mins), len(selected)), b'-')
    res[name, :] = mats[:, selected]
    return res


def gen_med_score(nm):
    """
    compute occupancy scores for each taxa. Occupancy score is a real number between 0 and 1.
    the score is equal to min(1,taxa_nongap_length/median_nongap_length)
    """
    name, mats = nm
    non_gap_count = np.sum(mats != b'-', axis=1)
    med_scores = np.clip(non_gap_count / np.median(non_gap_count), 0, 1)
    med_scores_with_zeros = np.zeros(len(catalog))
    med_scores_with_zeros[name] = med_scores
    return med_scores_with_zeros


if __name__ == "__main__":
    only_files = [join(sys.argv[1], f) for f in listdir(sys.argv[1]) if isfile(join(sys.argv[1], f))]
    np.random.seed(42)
    gap_thr = 0.95
    concat_len = 500
    target_num = 50
    names_and_mats = [fasta2mat(f, False, False) for f in only_files]
    names_and_mats_ungapped = [gap_filter(n, m, gap_thr) for n, m in names_and_mats]
    # union all taxon names
    catalog = reduce(np.union1d, [n for n, m in names_and_mats_ungapped])
    # create bidirectional map before substituting taxon names with numbers
    id_to_name = dict(enumerate(catalog))
    name_to_id = {j: i for i, j in id_to_name.items()}
    # substitute names with ids
    names_and_mats_ungapped = [(np.array([name_to_id[ni] for ni in n]), m) for n, m in names_and_mats_ungapped]
    # compute total median-normalized occupancy scores per taxa
    tot_med_scores = sum([gen_med_score(nm) for nm in names_and_mats_ungapped])

    spec_counts = np.array(len(catalog) * [0])
    sites_per_gene = int(np.ceil(concat_len / len(names_and_mats_ungapped)))
    subsamples = []
    while names_and_mats_ungapped:
        # find the least occupant taxa until this iteration
        mins = spec_counts == min(spec_counts)
        # find the gene with most copies of the least occupant taxa
        scores = [sum(mins[n]) for n, _ in names_and_mats_ungapped]
        # if there is a tie, pick the gene with most taxa (ignore occupancy)
        tiebraker_scores = [len(n) for n, _ in names_and_mats_ungapped]
        max_scores = scores == max(scores)
        maxind = np.argmax(tiebraker_scores * max_scores)

        subsample = subsample_align(names_and_mats_ungapped[maxind], mins, sites_per_gene)
        subsamples.append(subsample)
        # update spec_counts
        spec_counts += ~(subsample == b'-').all(axis=1)
        # remove the processed alignment from the list of unprocessed
        del names_and_mats_ungapped[maxind]

    concat = np.concatenate(subsamples, axis=1)
    # convert byte array to string
    concat = [i.tobytes().decode("utf-8") for i in concat]
    concat_fp = tempfile.NamedTemporaryFile(delete=False, mode='w+t')
    fasttree_log = tempfile.NamedTemporaryFile(delete=False, mode='w+t').name
    fasttree_out = tempfile.NamedTemporaryFile(delete=False, mode='w+t').name

    with open(concat_fp.name, "w") as f:
        for ids, seq in enumerate(concat):
            f.write(">" + id_to_name[ids] + "\n")
            f.write(seq + "\n")

    # run fasttree and write its output to a file
    s = ["FastTree", "-nopr", "-gtr", "-nt", "-log", fasttree_log]
    # s = ["FastTree", "-nopr", "-gtr", "-nt", "-gamma", "-log", fasttree_log]

    with open(concat_fp.name, "r") as rf:
        with Popen(s, stdout=PIPE, stdin=rf, stderr=sys.stderr) as p:
            tree_string = p.stdout.read().decode('utf-8')
            with open(fasttree_out, "w") as fout:
                fout.write(tree_string.strip() + "\n")
    if not tree_string:
        sys.stderr.write("FastTree failed. Check your FastTreeMP installation.\n")
        exit(1)

    tmax = ts.read_tree_newick(tree_string).diameter()
    tmin = 0
    tcur = tmax / 2
    stop_cond = BINARY_SEARCH_STOP_MULTIPLIER * tmax

    numiter = 0

    treecluster_out = tempfile.NamedTemporaryFile(delete=False, mode='w+t').name
    nldef = tempfile.NamedTemporaryFile(delete=True, mode='w+t')
    clusters = None
    num_clusters = 0
    while tcur - tmin >= stop_cond and tmax - tcur >= stop_cond:
        numiter += 1
        #print(numiter)
        s = ["TreeCluster.py", "-i", fasttree_out, "-m", "max_clade", "-t", str(tcur), "-o", treecluster_out]
        call(s, stdout=nldef, stderr=nldef)
        with open(treecluster_out, "r") as tc_output:
            tc_output.readline()
            lines = map(lambda x: x.strip().split('\t'), tc_output.readlines())
        lines_sorted = sorted(lines, key=lambda x: x[1])
        clusters = [(key, [i[0] for i in list(group)]) for key, group in groupby(lines_sorted, lambda x: x[1])]
        num_singletons = sum([len(tags) for idx, tags in clusters if idx == "-1"])
        num_clusters = len(clusters) + max(0, num_singletons - 1)
        if num_clusters == target_num:
            break
        elif num_clusters < target_num:
            tmax = tcur
            tcur = (tmin + tcur) / 2
        else:
            tmin = tcur
            tcur = (tmax + tcur) / 2

    select = []
    for ci, tags in clusters:
        if ci == "-1":
            select += tags
        else:
            hi_med, tag = sorted(zip(list(map(lambda x: tot_med_scores[name_to_id[x]],tags)),tags),reverse=True)[0]
            select.append(tag)

    for i in select:
        print(i)

    # if not options.protein_seqs:
    #     s.append("-nt")
