from typing import List
import re
import pandas as pd
from collections import defaultdict

from lib.vcf import LocusInfo, SnipInfo, Genome


def read_phase_inp_file(directory: str, chromosome: str):

    filename = f"{directory}/phase_invicta.{chromosome}.inp"
    file = open(filename, "r")
    lines = file.read().splitlines()

    phase_inp_dict = {}

    current_patient_id = None
    current_patient_haplotype = None
    reading_genotypes = False
    for line in lines[4:]:

        match = re.match(r"^(\d+)$", line)
        if match is not None:
            current_patient_id = match.groups()[0]
            current_patient_haplotype = ""
            continue
        line = line.replace(" ", "")
        if current_patient_haplotype == "":
            current_patient_haplotype = line
            continue
        current_patient_haplotype += f"_{line}"
        phase_inp_dict[current_patient_id] = {
            f"phase_inp_{chromosome}": current_patient_haplotype,
        }

    phase_inp_df = pd.DataFrame.from_dict(phase_inp_dict, orient="index")
    for col in list(phase_inp_df.columns):
        phase_inp_df[col] = phase_inp_df[col].astype("category")
    phase_inp_df = phase_inp_df.reset_index().rename(columns={"index": "patient_id"})
    phase_inp_df["patient_id"] = phase_inp_df["patient_id"].astype(int)

    return phase_inp_df


def read_phase_inp_files(
    directory: str,
    chromosomes: List[str],
    data_900_df: pd.DataFrame,
):

    print(len(data_900_df))
    for chromosome in chromosomes:
        phase_inp_df = read_phase_inp_file(directory, chromosome)
        data_900_df = data_900_df.merge(
            phase_inp_df, left_on="__id__", right_on="patient_id", how="inner"
        )
    print(len(data_900_df))

    return data_900_df


def read_fastphase_file(directory: str, chromosome: str, error_type: str):

    filename = f"{directory}/fastphase.{chromosome}_hapguess_{error_type}.out"
    file = open(filename, "r")
    lines = file.read().splitlines()

    fastphase_dict = {}

    current_patient_id = None
    current_patient_haplotype = None
    reading_genotypes = False
    for line in lines:

        if re.match(r"^BEGIN GENOTYPES$", line):
            reading_genotypes = True
            continue

        if re.match(r"^END GENOTYPES$", line):
            reading_genotypes = False
            continue

        if reading_genotypes:
            match = re.match(r"^(\d+)$", line)
            if match is not None:
                current_patient_id = match.groups()[0]
                current_patient_haplotype = ""
                continue
            match = re.match(r"^$", line)
            if match is not None:
                continue
            line = line.replace(" ", "")
            if current_patient_haplotype == "":
                current_patient_haplotype = line
                continue
            current_patient_haplotype += f"_{line}"
            fastphase_dict[current_patient_id] = {
                f"fastphase_{chromosome}_{error_type}": current_patient_haplotype,
            }

    fastphase_df = pd.DataFrame.from_dict(fastphase_dict, orient="index")
    for col in list(fastphase_df.columns):
        fastphase_df[col] = fastphase_df[col].astype("category")
    fastphase_df = fastphase_df.reset_index().rename(columns={"index": "patient_id"})
    fastphase_df["patient_id"] = fastphase_df["patient_id"].astype(int)

    return fastphase_df


def read_fastphase_files(
    directory: str,
    chromosomes: List[str],
    data_900_df: pd.DataFrame,
):

    print(len(data_900_df))
    for chromosome in chromosomes:
        for error_type in ["indiv", "switch"]:
            fastphase_df = read_fastphase_file(directory, chromosome, error_type)
            data_900_df = data_900_df.merge(
                fastphase_df, left_on="__id__", right_on="patient_id", how="inner"
            )
    print(len(data_900_df))

    return data_900_df


def read_phase_file(chromosome: str, filename: str):

    file = open(filename, "r")
    lines = file.read().splitlines()

    phase_dict = {}

    reading_bestpairs = False
    for line in lines:

        if re.match(r"^BEGIN BESTPAIRS_SUMMARY$", line):
            reading_bestpairs = True
            continue

        if re.match(r"^END BESTPAIRS_SUMMARY$", line):
            reading_bestpairs = False
            continue

        if reading_bestpairs:
            match = re.match(r"^(\d*): \((\d*,\d*)\)$", line)
            patient_id = match.groups()[0]
            bestpair = match.groups()[1]
            bestpair = bestpair.replace(",", "_")
            bestpair_left = bestpair.split("_")[0]
            bestpair_right = bestpair.split("_")[1]

            phase_dict[patient_id] = {
                f"phase_summary_{chromosome}": bestpair,
                f"phase_sum_left_{chromosome}": bestpair_left,
                f"phase_sum_right_{chromosome}": bestpair_right,
                f"phase_bestpair_{chromosome}_{bestpair}": True,
            }

    phase_df = pd.DataFrame.from_dict(phase_dict, orient="index")
    for col in list(phase_df.columns):
        if "summary" in col or "sum_left" in col or "sum_right" in col:
            phase_df[col] = phase_df[col].astype("category")
        if "bestpair" in col:
            phase_df.fillna({col: False}, inplace=True)
            phase_df[col] = phase_df[col].astype(bool)
    phase_df = phase_df.reset_index().rename(columns={"index": "patient_id"})
    phase_df["patient_id"] = phase_df["patient_id"].astype(int)

    return phase_df


def read_phase_files(
    directory: str,
    chromosomes: List[str],
    data_900_df: pd.DataFrame,
):

    print(len(data_900_df))
    for chromosome in chromosomes:
        phase_df = read_phase_file(chromosome, f"{directory}/out_{chromosome}")
        data_900_df = data_900_df.merge(
            phase_df, left_on="__id__", right_on="patient_id", how="inner"
        )
    print(len(data_900_df))

    return data_900_df


def read_hmap_file(filename: str):

    file = open(filename, "r")
    lines = file.read().splitlines()

    patient_ids = None
    locus_infos = []
    locus_dict = {}
    patient_genome = {}

    for line in lines:

        if re.match(r"^#@", line):
            continue

        if re.match(r"^rs#", line):
            line = line.split(" ")
            patient_ids = [na_id.split("_")[1] for na_id in line[11:]]
            continue

        assert re.match(r"^rs_.*$", line)
        line = line.split(" ")
        locus_info = LocusInfo(
            chrom=line[2],
            pos=line[3],
        )
        snip_info = SnipInfo(ref=line[1].split("/")[0], alts=[line[1].split("/")[1]])
        locus_infos += [locus_info]
        locus_dict[locus_info] = snip_info
        for patient_id, two_letters_genome in zip(patient_ids, line[11:]):
            patient_genome[(patient_id, locus_info)] = Genome(
                gt0=two_letters_genome[0],
                gt1=two_letters_genome[1],
            )

    return patient_ids, locus_infos, locus_dict, patient_genome


def read_haploview_file(
    filename: str,
    display_blocks: bool = False,
):

    mapping = {}
    mapping["1"] = "A"
    mapping["2"] = "C"
    mapping["3"] = "G"
    mapping["4"] = "T"

    file = open(filename, "r")
    lines = file.read().splitlines()

    current_block = None
    blocks = {}
    haplotypes = defaultdict(list)

    for line in lines:

        block_match = re.match(r"^BLOCK (\d*)\.\s*MARKERS: (.*)$", line)
        if block_match:
            blocks[int(block_match[1])] = [
                int(marker) for marker in block_match[2].split(" ")
            ]
            current_block = int(block_match[1])
            if display_blocks:
                print(f"Haploview output - block{current_block}")
            continue
        if re.match("^Multiallelic Dprime:", line):
            continue

        haplotype_match = re.match(r"^(\d*) \((\d\.\d*)\).*$", line)
        assert haplotype_match
        haplotype = haplotype_match[1]
        percentage = haplotype_match[2]
        assert len(haplotype) == len(blocks[current_block])
        for mapping_key, mapping_value in mapping.items():
            haplotype = haplotype.replace(mapping_key, mapping_value)
        haplotypes[current_block] += [haplotype]
        if display_blocks:
            print(f"{haplotype} : {percentage}")

    return blocks, haplotypes


def read_haploview_column(
    directory: str,
    chromosome: str,
    method: str,
    display_blocks: bool = False,
):

    patient_ids, locus_infos, locus_dict, patient_genome = read_hmap_file(
        f"{directory}/hmap_invicta.{chromosome}.hmap"
    )

    blocks, haplotypes = read_haploview_file(
        f"{directory}/hmap_invicta.{chromosome}.hmap.{method}blocks",
        display_blocks,
    )

    patient_haplotype_dict = defaultdict(lambda: defaultdict(str))

    for patient_id in patient_ids:
        for block_id, block in blocks.items():
            patient_haplotype_at_block = ""
            for marker_pos in block:
                locus_info = locus_infos[marker_pos - 1]
                genome = patient_genome[(patient_id, locus_info)]
                snip_info = locus_dict[locus_info]
                if genome.gt0 == snip_info.ref and genome.gt1 == snip_info.ref:
                    haplotype_at_marker = snip_info.ref
                else:
                    haplotype_at_marker = ",".join(snip_info.alts)
                patient_haplotype_at_block += haplotype_at_marker

            if patient_haplotype_at_block not in haplotypes[block_id]:
                patient_haplotype_at_block = "NONE"
            haplotype_col_name = (
                f"{chromosome}_{method}_block{block_id}_{patient_haplotype_at_block}"
            )
            patient_haplotype_dict[patient_id][haplotype_col_name] = True

    patient_haplotype_df = pd.DataFrame.from_dict(
        patient_haplotype_dict, orient="index"
    )
    patient_haplotype_df.fillna(False, inplace=True)
    block_cols = list(patient_haplotype_df.columns)

    for col in block_cols:
        patient_haplotype_df[col] = patient_haplotype_df[col].astype(bool)

    if display_blocks:
        print("=====================================")
        for block in sorted(list(set([col.split("_")[2] for col in block_cols]))):
            print(f"patient_haplotype_df - {block}")

            for col in block_cols:
                if col.split("_")[2] != block:
                    continue
                count_true = len(patient_haplotype_df[patient_haplotype_df[col]])
                print(
                    f'{col.split("_")[3]} : {"%.3f"%(count_true / len(patient_haplotype_df))}'
                )

    patient_haplotype_df = patient_haplotype_df.reset_index().rename(
        columns={
            "index": "patient_id",
        }
    )
    patient_haplotype_df["patient_id"] = patient_haplotype_df["patient_id"].astype(int)

    return patient_haplotype_df, block_cols


def read_haploview_data(
    directory: str,
    chromosomes: List[str],
    methods: List[str],
    data_900_df: pd.DataFrame,
):

    all_block_cols = []

    for chromosome in chromosomes:
        for method in methods:
            patient_haplotype_df, block_cols = read_haploview_column(
                directory, chromosome, method, False
            )
            all_block_cols += block_cols

            if len(patient_haplotype_df) != 0:
                data_900_df = data_900_df.merge(
                    patient_haplotype_df,
                    left_on="__id__",
                    right_on="patient_id",
                    how="inner",
                )

    return data_900_df, all_block_cols
