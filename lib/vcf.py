from typing import Dict, Tuple, List
from dataclasses import dataclass
import os
import re
from collections import defaultdict, OrderedDict
import pandas as pd
from IPython.display import display


@dataclass
class LocusInfo:
    chrom: str
    pos: int

    def __hash__(self):
        return hash((self.chrom, self.pos))

    def __gt__(self, other) -> bool:
        return (self.chrom, int(self.pos)) > (other.chrom, int(other.pos))

    def __eq__(self, other):
        return (self.chrom, int(self.pos)) == (other.chrom, int(other.pos))


@dataclass
class SnipInfo:
    ref: str
    alts: List[str]

    def __hash__(self):
        return hash((self.ref, str(self.alts)))


@dataclass
class Genome:
    gt0: str
    gt1: str

    def __hash__(self):
        return hash((self.gt0, self.gt1))


@dataclass
class GeneticData:
    chromosomes: List[str]
    locus_snip_infos: Dict[LocusInfo, List[SnipInfo]]
    PATIENT_DICT: Dict[int, Dict[LocusInfo, Genome]]


@dataclass
class FileInfo:
    filename: str
    vcf_patient_id: str
    genotype: Dict[LocusInfo, Tuple[SnipInfo, Genome]]


def get_genotype(ref: str, alts: List[str], gt: int) -> str:
    gt = int(gt)
    assert gt <= len(alts)
    if gt == 0:
        return ref
    else:
        return alts[gt - 1]


def read_mappings(directory: str) -> Tuple[Dict[str, str], Dict[str, str]]:

    # Read rpo mapping
    rpo_df = pd.read_csv(f"{directory}/rpo_1_1_1_full.csv", sep=";")
    rpo_dict = {}
    for _index, row in rpo_df.iterrows():
        project_order_number = str(row["project_order_number"])
        archival_order_number = str(row["archival_order_number"])
        assert project_order_number not in rpo_dict.keys()
        rpo_dict[project_order_number] = archival_order_number

    # Read archival mapping
    archival_df = pd.read_csv(
        f"{directory}/mapping_existing_patients_aftergosia.csv"  # mapping_archival_order_number_patientId
    )
    archival_dict = {}
    for _index, row in archival_df.iterrows():
        number = str(row["number"])
        patient_id = str(row["patientId"])
        if number == "0":
            continue
        assert number not in archival_dict
        archival_dict[number] = patient_id

    return rpo_dict, archival_dict


def find_in_mappings(value, rpo_dict, archival_dict):
    if str(value) not in rpo_dict:
        return None
    rpo_id = str(rpo_dict[str(value)])
    if rpo_id not in archival_dict:
        return None
    return archival_dict[rpo_id]


def read_allele_file(
    directory: str,
    rpo_dict: Dict[str, str],
    archival_dict: Dict[str, str],
) -> pd.DataFrame:
    allele_df = pd.read_csv(f"{directory}/result_allele_call.csv", sep=";")

    allele_df["patient_id"] = (
        allele_df["ORDER_NUMBER"]
        .apply(find_in_mappings, rpo_dict=rpo_dict, archival_dict=archival_dict)
        .fillna("-1")
        .astype(int)
    )
    allele_df = allele_df[allele_df["patient_id"] != -1]

    allele_df.drop(columns="ORDER_NUMBER", inplace=True)
    allele_df.drop_duplicates(inplace=True)

    return allele_df


def prepare_genetic_data(allele_df: pd.DataFrame) -> GeneticData:

    chromosomes = sorted(list(allele_df["CHROME"].unique()))

    locus_snip_infos = {}
    for _index, row in (
        allele_df[["CHROME", "POS"]]
        .groupby(["CHROME", "POS"])
        .size()
        .reset_index()
        .iterrows()
    ):
        chrom = row["CHROME"]
        pos = row["POS"]
        locus_count = row[0]

        locus_info = LocusInfo(
            chrom=chrom,
            pos=pos,
        )
        locus_filter = (allele_df["CHROME"] == chrom) & (allele_df["POS"] == pos)
        locus_df = allele_df[locus_filter]
        locus_agg_df = (
            locus_df[["REF", "ALT"]].groupby(["REF", "ALT"]).size().reset_index()
        )

        if len(locus_agg_df) != 1:
            print(f"Skiping ambigous locus: {locus_info}: {locus_count} records")
            continue

        ref, alt = locus_agg_df[["REF", "ALT"]].values[0]
        snip_info = SnipInfo(
            ref=ref,
            alts=[alt],
        )

        locus_snip_infos[locus_info] = [snip_info]

    allele_dict = {}
    for index, row in allele_df.iterrows():
        allele_dict[
            row["patient_id"], row["CHROME"], row["POS"], row["REF"], row["ALT"]
        ] = row["ALLELE_CALL"]

    PATIENT_DICT = {}
    for patient_id in list(allele_df["patient_id"].unique()):
        patient_info = {}
        for locus_info, snip_infos in locus_snip_infos.items():
            chrom = locus_info.chrom
            pos = locus_info.pos
            assert len(snip_infos) == 1
            snip_info = snip_infos[0]
            ref = snip_info.ref
            assert len(snip_info.alts) == 1
            alt = snip_info.alts[0]

            if (patient_id, chrom, pos, ref, alt) not in allele_dict:
                patient_info[locus_info] = Genome(
                    gt0=ref,
                    gt1=ref,
                )
                continue
            allele_call = allele_dict[patient_id, chrom, pos, ref, alt]
            if allele_call == "Homozygous":
                patient_info[locus_info] = Genome(
                    gt0=alt,
                    gt1=alt,
                )
                continue
            if allele_call == "Heterozygous":
                patient_info[locus_info] = Genome(
                    gt0=ref,
                    gt1=alt,
                )
                continue
            raise ValueError(f"Invalid allele_call: {allele_call}")
        PATIENT_DICT[patient_id] = patient_info

    return GeneticData(
        chromosomes=chromosomes,
        locus_snip_infos=locus_snip_infos,
        PATIENT_DICT=PATIENT_DICT,
    )


def read_file(filename):

    file = open(filename, "r")
    lines = file.read().splitlines()

    vcf_patient_id = None
    genotype = {}

    for line in lines:

        if line[0:2] == "##":
            continue

        if line[0:6] == "#CHROM":
            line = line.split("\t")
            vcf_patient_id = line[9]
            continue

        line = line.split("\t")
        chrom = line[0]
        pos = line[1]
        ref = line[3]
        alts = line[4].split(",")
        format_value = line[8]
        assert format_value[0:2] == "GT"
        gt = line[9].split(":")[0].split("/")
        assert len(gt) == 2
        gt0 = get_genotype(ref, alts, gt[0])
        gt1 = get_genotype(ref, alts, gt[1])

        locus_info = LocusInfo(
            chrom=chrom,
            pos=pos,
        )
        snip_info = SnipInfo(
            ref=ref,
            alts=alts,
        )
        genome = Genome(
            gt0=gt0,
            gt1=gt1,
        )

        genotype[locus_info] = (snip_info, genome)

    return FileInfo(
        filename=filename,
        vcf_patient_id=vcf_patient_id,
        genotype=genotype,
    )


def read_vcf_files(directory: str, unikalne_vcf_directory: str, return_mapping=False):

    rpo_dict, archival_dict = read_mappings(directory)

    # Read files
    file_infos = []
    for _root, subFolders, _filenames in os.walk(f"{unikalne_vcf_directory}/"):
        for folder in subFolders:
            for _root, _subFolders, filenames in os.walk(
                f"{unikalne_vcf_directory}/{folder}/"
            ):
                for filename in filenames:
                    if not re.match(r"^.*\.vcf$", filename):
                        continue
                    filename = f"{unikalne_vcf_directory}/{folder}/{filename}"
                    file_info = read_file(filename)
                    file_infos += [file_info]
    # Prepare:
    # - CHROMOSOMES
    # - LOCUS_DICT[locus_info][snip_info] -> count
    # - TOTAL_RECORDS
    CHROMOSOMES = set()
    LOCUS_DICT = defaultdict(lambda: defaultdict(int))
    TOTAL_RECORDS = 0
    for file_info in file_infos:
        for locus_info, (snip_info, _genome) in file_info.genotype.items():
            LOCUS_DICT[locus_info][snip_info] += 1
            CHROMOSOMES.add(locus_info.chrom)
            TOTAL_RECORDS += 1

    print("===============================================")
    # Prepare:
    # - ambigous_locus_infos_set
    ambigous_locus_infos_set = set()
    for locus_info, snip_dict in LOCUS_DICT.items():
        if len(snip_dict) == 1:
            continue
        ambigous_locus_infos_set.add(locus_info)
        for snip_info, count in snip_dict.items():
            print(
                f"Ambiguous locus: count: "
                f'{str(count).ljust(5, " ")} '
                f'{locus_info.chrom.ljust(5, " ")} '
                f'{locus_info.pos.ljust(10, " ")} '
                f'{snip_info.ref.ljust(20, " ")} -> '
                f'{",".join(snip_info.alts).ljust(20, " ")} '
            )

    # Remove ambigous locus_infos
    removed_records_count = 0
    for locus_info in ambigous_locus_infos_set:
        for _snip_info, count in LOCUS_DICT[locus_info].items():
            removed_records_count += count
        del LOCUS_DICT[locus_info]
    print(
        f"\033[44m"
        f"Removed amgigous locuses - "
        f"records: {removed_records_count} "
        f"of: {TOTAL_RECORDS} "
        f"percentage: {round(100 * removed_records_count / TOTAL_RECORDS, 2)}%"
        f"\033[0m"
    )
    LOCUS_DICT = OrderedDict(sorted(LOCUS_DICT.items()))

    # Prepare:
    # - PATIENT_DICT[patient_id][locus_info][genome] -> count
    # - not_found_rpo_ids
    # - not_found_archiva_ids
    # - PATIENT_FILES
    PATIENT_DICT = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    PATIENT_FILES = defaultdict(set)
    MAPPING_list = []
    not_found_rpo_ids = set()
    not_found_archival_ids = set()
    for file_info in file_infos:
        patient_id = None
        vcf_patient_id = file_info.vcf_patient_id.split("_")
        print(vcf_patient_id)

        # Find patient_id in mappings
        rpo_id = None
        if len(vcf_patient_id) >= 2 and vcf_patient_id[1] in rpo_dict:
            vcf_patient_id_mapping = vcf_patient_id[1]
            rpo_id = rpo_dict[vcf_patient_id[1]]
        if len(vcf_patient_id) >= 3 and vcf_patient_id[2] in rpo_dict:
            vcf_patient_id_mapping = vcf_patient_id[2]
            rpo_id = rpo_dict[vcf_patient_id[2]]
        print(rpo_id)
        if rpo_id is None:
            not_found_rpo_ids.add("_".join(vcf_patient_id))
            continue
        if rpo_id not in archival_dict:
            not_found_archival_ids.add("_".join(vcf_patient_id))
            continue
        patient_id = archival_dict[rpo_id]

        for locus_info, (_snip_info, genome) in file_info.genotype.items():
            PATIENT_DICT[patient_id][locus_info][genome] += 1
        PATIENT_FILES[patient_id].add(file_info.filename)
        MAPPING_list.append(
            [file_info.filename, vcf_patient_id_mapping, rpo_id, patient_id]
        )

    print("===============================================")
    # Prepare:
    # - UNAMBIGOUS_PATIENT_DICT[patient_id][locus_info][genome] -> count
    UNAMBIGOUS_PATIENT_DICT = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    all_genomes_count = 0
    ambigous_genomes_count = 0
    for patient_id, locus_dict in PATIENT_DICT.items():
        for locus_info, genome_dict in locus_dict.items():
            all_genomes_count += 1
            if len(genome_dict) == 1:
                genome = list(genome_dict.keys())[0]
                print(genome)
                UNAMBIGOUS_PATIENT_DICT[patient_id][locus_info][genome] += 1
                print(UNAMBIGOUS_PATIENT_DICT[patient_id][locus_info][genome])
                continue
            for genome, count in genome_dict.items():
                print(
                    f"Removing ambiguous genome: "
                    f'{patient_id.ljust(10, " ")} '
                    f'{locus_info.chrom.ljust(5, " ")} '
                    f'{locus_info.pos.ljust(10, " ")} '
                    f'{genome.gt0.ljust(20, " ")} , '
                    f'{genome.gt1.ljust(20, " ")}'
                )
                ambigous_genomes_count += 1
    print(
        f"\033[44m"
        f"Removed ambigous_genomes - "
        f"records: {ambigous_genomes_count} "
        f"of: {all_genomes_count} "
        f"percentage: {round(100 * ambigous_genomes_count / all_genomes_count, 2)}%"
        f"\033[0m"
    )
    if return_mapping:
        return (
            GeneticData(
                chromosomes=CHROMOSOMES,
                locus_snip_infos=LOCUS_DICT,
                PATIENT_DICT=UNAMBIGOUS_PATIENT_DICT,
            ),
            MAPPING_list,
        )

    return GeneticData(
        chromosomes=CHROMOSOMES,
        locus_snip_infos=LOCUS_DICT,
        PATIENT_DICT=UNAMBIGOUS_PATIENT_DICT,
    )


def prepare_hmap_files(
    genetic_data: GeneticData,
    output_directory: str,
):

    # Prepare .hmap format
    pedigree_list = []
    for patient_id in genetic_data.PATIENT_DICT.keys():
        pedigree_entry = {
            "prefix": "#@",
            "family": f"FAM_{patient_id}",
            "individual": f"NA_{patient_id}",
            "father": "0",  # 0 = unknown father
            "mother": "0",  # 0 = unknown mother
            "sex": "2",  # 2 = female
            "affected": "0",  # 0 = unknown
        }
        pedigree_list += [pedigree_entry]

    pedigree_df = pd.DataFrame(pedigree_list)

    hmap_list = []
    for locus_info, snip_infos in genetic_data.locus_snip_infos.items():
        assert len(snip_infos) == 1
        snip_info = list(snip_infos)[0]

        if len(snip_info.ref) != 1:
            print(f"Skiping {locus_info} - {snip_info}")
            continue
        if snip_info.ref == "-":
            print(f"Skiping {locus_info} - {snip_info}")
            continue
        if len(snip_info.alts) != 1:
            print(f"Skiping {locus_info} - {snip_info}")
            continue
        if len(snip_info.alts[0]) != 1:
            print(f"Skiping {locus_info} - {snip_info}")
            continue
        if snip_info.alts == ["-"]:
            print(f"Skiping {locus_info} - {snip_info}")
            continue

        hmap_entry = {
            "rs#": f"rs_{locus_info.chrom}_{locus_info.pos}",
            "alleles": f'{snip_info.ref}/{",".join(snip_info.alts)}',
            "chrom": f"{locus_info.chrom}",
            "pos": f"{locus_info.pos}",
            "strand": "+",
            "assembly#": "NA",
            "center": "NA",
            "protLSID": "NA",
            "assayLSID": "NA",
            "panelLSID": "NA",
            "QCcode": "NA",
        }
        for patient_id in genetic_data.PATIENT_DICT.keys():
            # Default = {ref}{ref}
            genome = f"{snip_info.ref}{snip_info.ref}"
            if locus_info in genetic_data.PATIENT_DICT[patient_id]:
                assert len(genetic_data.PATIENT_DICT[patient_id][locus_info]) == 1
                genome = list(genetic_data.PATIENT_DICT[patient_id][locus_info].keys())[
                    0
                ]
                genome = f"{genome.gt0}" f"{genome.gt1}"
            hmap_entry[f"NA_{patient_id}"] = genome

        hmap_list += [hmap_entry]

    hmap_df = pd.DataFrame(hmap_list)

    for chrom in genetic_data.chromosomes:

        filename = f"{output_directory}/hmap_invicta.{chrom}.hmap"
        pedigree_df.to_csv(filename, sep=" ", index=False, header=False)
        hmap_df[hmap_df["chrom"] == chrom].to_csv(
            filename, sep=" ", index=False, mode="a"
        )


def prepare_phase_files(
    genetic_data: GeneticData,
    output_directory: str,
):

    # Prepare PHASE .inp format
    for chrom in genetic_data.chromosomes:
        filename = f"{output_directory}/phase_invicta.{chrom}.inp"
        positions = []
        for locus_info, snip_infos in genetic_data.locus_snip_infos.items():
            if locus_info.chrom != chrom:
                continue
            assert len(snip_infos) == 1
            snip_info = list(snip_infos)[0]

            if len(snip_info.ref) != 1:
                print(f"Skiping {locus_info} - {snip_info}")
                continue
            if snip_info.ref == "-":
                print(f"Skiping {locus_info} - {snip_info}")
                continue
            if len(snip_info.alts) != 1:
                print(f"Skiping {locus_info} - {snip_info}")
                continue
            if len(snip_info.alts[0]) != 1:
                print(f"Skiping {locus_info} - {snip_info}")
                continue
            if snip_info.alts == ["-"]:
                print(f"Skiping {locus_info} - {snip_info}")
                continue
            positions += [locus_info.pos]

        file = open(filename, "w")
        file.write(f"{len(genetic_data.PATIENT_DICT)}\n")
        file.write(f"{len(positions)}\n")
        file.write("P")
        for pos in positions:
            file.write(f" {pos}")
        file.write("\n")
        file.write(f"S" * len(positions))
        file.write("\n")

        for patient_id, _patient_info in genetic_data.PATIENT_DICT.items():

            file.write(f"{patient_id}\n")

            genomes = []
            for pos in positions:
                locus_info = LocusInfo(
                    chrom=chrom,
                    pos=pos,
                )

                snip_infos = genetic_data.locus_snip_infos[locus_info]
                assert len(snip_infos) == 1
                snip_info = list(snip_infos)[0]

                # Default = {ref}{ref}
                genome = Genome(
                    gt0=snip_info.ref,
                    gt1=snip_info.ref,
                )
                if locus_info in genetic_data.PATIENT_DICT[patient_id]:
                    assert len(genetic_data.PATIENT_DICT[patient_id][locus_info]) == 1
                    genome = list(
                        genetic_data.PATIENT_DICT[patient_id][locus_info].keys()
                    )[0]
                genomes += [genome]

            file.write(" ".join([genome.gt0 for genome in genomes]))
            file.write("\n")
            file.write(" ".join([genome.gt1 for genome in genomes]))
            file.write("\n")

        file.close()


def prepare_phased_files(
    genetic_data: GeneticData,
    output_directory: str,
):

    # Prepare .phased format
    phased_list = []
    for locus_info, snip_infos in genetic_data.locus_snip_infos.items():
        assert len(snip_infos) == 1
        snip_info = list(snip_infos)[0]

        if len(snip_info.ref) != 1:
            print(f"Skiping {locus_info} - {snip_info}")
            continue
        if snip_info.ref == "-":
            print(f"Skiping {locus_info} - {snip_info}")
            continue
        if len(snip_info.alts) != 1:
            print(f"Skiping {locus_info} - {snip_info}")
            continue
        if len(snip_info.alts[0]) != 1:
            print(f"Skiping {locus_info} - {snip_info}")
            continue
        if snip_info.alts == ["-"]:
            print(f"Skiping {locus_info} - {snip_info}")
            continue

        phased_entry = {
            "rsID": f"rs_{locus_info.chrom}_{locus_info.pos}",
            "position": f"{locus_info.pos}",
            "chrom": locus_info.chrom,
        }
        for patient_id in genetic_data.PATIENT_DICT.keys():
            # Default = {ref}{ref}
            genome = f"{snip_info.ref}{snip_info.ref}"
            if locus_info in genetic_data.PATIENT_DICT[patient_id]:
                assert len(genetic_data.PATIENT_DICT[patient_id][locus_info]) == 1
                genome = list(genetic_data.PATIENT_DICT[patient_id][locus_info].keys())[
                    0
                ]
                genome = f"{genome.gt0}" f"{genome.gt1}"
            phased_entry[f"NA_{patient_id}_A"] = genome[0]
            phased_entry[f"NA_{patient_id}_B"] = genome[1]

        phased_list += [phased_entry]

    phased_df = pd.DataFrame(phased_list)

    for chrom in genetic_data.chromosomes:

        filename = f"{output_directory}/phased_invicta.{chrom}.phased"
        chromosome_df = phased_df[phased_df["chrom"] == chrom].copy()
        chromosome_df = chromosome_df.drop(columns=["chrom"])
        chromosome_df.to_csv(filename, sep=" ", index=False, mode="w")


def prepare_genome_df(
    genetic_data: GeneticData,
):

    genome_dict = {}
    for patient_id in genetic_data.PATIENT_DICT.keys():
        genome_dict[patient_id] = {}
        for locus_info, snip_infos in genetic_data.locus_snip_infos.items():
            assert len(snip_infos) == 1
            snip_info = list(snip_infos)[0]

            col_name = (
                f"{locus_info.chrom}_{locus_info.pos}_"
                f'{snip_info.ref}_{",".join(snip_info.alts)}'
            )

            # Default = {ref},{ref} - no change
            ref_diffs = 0
            if locus_info in genetic_data.PATIENT_DICT[patient_id]:
                assert len(genetic_data.PATIENT_DICT[patient_id][locus_info]) == 1
                genome = list(genetic_data.PATIENT_DICT[patient_id][locus_info].keys())[
                    0
                ]
                if genome.gt0 != snip_info.ref:
                    ref_diffs += 1
                if genome.gt1 != snip_info.ref:
                    ref_diffs += 1

            genome_dict[patient_id][f"genome_012_{col_name}"] = ref_diffs
            genome_dict[patient_id][f"has_alt_{col_name}"] = ref_diffs >= 1
            genome_dict[patient_id][f"hetero_{col_name}"] = ref_diffs == 1
            genome_dict[patient_id][f"homo_{col_name}"] = ref_diffs == 2

    genome_df = pd.DataFrame.from_dict(genome_dict, orient="index")
    genome_df = genome_df.reset_index().rename(columns={"index": "patient_id"})
    genome_df["patient_id"] = genome_df["patient_id"].astype(int)
    genome_df = genome_df.sort_values(by="patient_id")

    genome_012_cols = [col for col in list(genome_df.columns) if "genome_012" in col]
    hetero_cols = [col for col in list(genome_df.columns) if "hetero" in col]
    homo_cols = [col for col in list(genome_df.columns) if "homo" in col]
    has_alt_cols = [col for col in list(genome_df.columns) if "has_alt" in col]

    return genome_df, genome_012_cols, hetero_cols, homo_cols, has_alt_cols


def read_vcf_files_list(directory: str, unikalne_vcf_directory: str):
    # Read files
    rpo_dict, archival_dict = read_mappings(directory)
    file_infos = []
    for _root, subFolders, _filenames in os.walk(f"{unikalne_vcf_directory}/"):
        for folder in subFolders:
            for _root, _subFolders, filenames in os.walk(
                f"{unikalne_vcf_directory}/{folder}/"
            ):
                for filename in filenames:
                    if not re.match(r"^.*\.vcf$", filename):
                        continue
                    filename = f"{unikalne_vcf_directory}/{folder}/{filename}"
                    file_info = read_file_list(filename)
                    file_infos.append(file_info)
    file_infos = pd.DataFrame(
        [vcf for vcfs in file_infos for vcf in vcfs],
        columns=[
            "vcf_patient_id",
            "chromosome",
            "position",
            "reference",
            "alts",
            "gt0",
            "alternative",
        ],
    )
    file_infos["archival_id"] = file_infos["vcf_patient_id"].map(rpo_dict)
    file_infos["patient_id"] = file_infos["archival_id"].map(archival_dict)
    return file_infos


def read_file_list(filename):

    file = open(filename, "r")
    lines = file.read().splitlines()

    vcf_patient_id = None
    results = []

    for line in lines:

        if line[0:2] == "##":
            continue

        if line[0:6] == "#CHROM":
            line = line.split("\t")
            vcf_patient_id = line[9]
            vcf_patient_ids = vcf_patient_id.split("_")
            vcf_patient_id = max(vcf_patient_ids, key=len)
            continue

        line = line.split("\t")
        chrom = line[0]
        pos = line[1]
        ref = line[3]
        alts = line[4].split(",")
        format_value = line[8]
        assert format_value[0:2] == "GT"
        gt = line[9].split(":")[0].split("/")
        assert len(gt) == 2
        gt0 = get_genotype(ref, alts, gt[0])
        gt1 = get_genotype(ref, alts, gt[1])
        results.append([vcf_patient_id, chrom, pos, ref, alts, gt0, gt1])

    return results
