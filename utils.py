from Bio.Seq import Seq
from Bio.Align import PairwiseAligner
import pandas as pd
import numpy as np

HOTSPOTS = [175, 245, 248, 249, 273, 282]

def clean_sequence(seq):
    seq = seq.replace("\n","")
    seq = ''.join([c for c in seq if c.isalpha()])
    return seq.upper()

def is_nucleotide(seq):
    valid = set("ACGTN")
    return sum(1 for c in seq if c in valid) / len(seq) > 0.9

def translate_if_needed(seq):
    if is_nucleotide(seq):
        return str(Seq(seq).translate(to_stop=True))
    return seq

def align_and_map(human_seq, query_seq):
    aligner = PairwiseAligner()
    aligner.mode = "global"
    alignment = aligner.align(human_seq, query_seq)[0]

    aligned_human = alignment.seqA
    aligned_query = alignment.seqB

    results = []
    human_pos = 0

    for i in range(len(aligned_human)):
        if aligned_human[i] != "-":
            human_pos += 1

        if human_pos in HOTSPOTS:
            results.append({
                "Codon": human_pos,
                "Human": aligned_human[i],
                "Query": aligned_query[i],
                "Conserved": aligned_human[i] == aligned_query[i]
            })

    return pd.DataFrame(results)
