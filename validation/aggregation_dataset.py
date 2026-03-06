"""
Labelled dataset for aggregation screen validation.

Three groups:
  - Positive  (label=1): well-known amyloid / prion-forming sequences
  - Negative  (label=0): stable, soluble, non-aggregating proteins
  - Near-miss (label=0): high beta-sheet content but non-amyloid

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LabelledSequence:
    name: str
    sequence: str
    label: int  # 1 = aggregating, 0 = non-aggregating
    group: str  # 'positive' | 'negative' | 'near_miss'


# ---------------------------------------------------------------------------
# Positive sequences (known amyloid / prion formers)
# ---------------------------------------------------------------------------

POSITIVE: list[LabelledSequence] = [
    LabelledSequence(
        name='PrP_106-126',
        sequence='KTNMKHMAGAAAAGAVVGGLGGYMLGSAMSRP',
        label=1,
        group='positive',
    ),
    LabelledSequence(
        name='amyloid_beta_42',
        sequence='DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA',
        label=1,
        group='positive',
    ),
    LabelledSequence(
        name='alpha_synuclein_NAC',
        sequence='EQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKK',
        label=1,
        group='positive',
    ),
    LabelledSequence(
        name='tau_PHF6',
        sequence='VQIVYKPVDLSKVTSKCGSLGNIHHKPGGG',
        label=1,
        group='positive',
    ),
    LabelledSequence(
        name='huntingtin_polyQ',
        sequence='MATLEKLMKAFESLKSFQQQQQQQQQQQQQQQQQQQQQQQ',
        label=1,
        group='positive',
    ),
    LabelledSequence(
        name='IAPP_amylin',
        sequence='KCNTATCATQRLANFLVHSSNNFGAILSSTNVGSNTY',
        label=1,
        group='positive',
    ),
    LabelledSequence(
        name='beta2_microglobulin_K3',
        sequence='IQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKN',
        label=1,
        group='positive',
    ),
    LabelledSequence(
        name='TTR_L55P_fragment',
        sequence='GPTGTGESKCPLMVKVLDAVRGSPAINVAVHVFRKAADDTWEPFASGK',
        label=1,
        group='positive',
    ),
    LabelledSequence(
        name='FUS_QGSY_region',
        sequence='MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQ',
        label=1,
        group='positive',
    ),
    LabelledSequence(
        name='TDP43_GRR',
        sequence='GRFGGGNSSSGPSGNNQNQGNMQREPNQAFGSGNNSYSGSNSGAAIGWGSASNAGSGSGFNGGFGSSMDS',
        label=1,
        group='positive',
    ),
]

# ---------------------------------------------------------------------------
# Negative sequences (stable, soluble, well-characterised)
# ---------------------------------------------------------------------------

NEGATIVE: list[LabelledSequence] = [
    LabelledSequence(
        name='hen_egg_lysozyme',
        sequence='KVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGDRSTDYGIFQINSRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRDPQGIRAWVAWRNRCQNRDVRQYVQGCGV',
        label=0,
        group='negative',
    ),
    LabelledSequence(
        name='sperm_whale_myoglobin',
        sequence='MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGGILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISDAIIHVLHSRHPGNFGADAQGAMNKALELFRKDIAAKYKELGYQG',
        label=0,
        group='negative',
    ),
    LabelledSequence(
        name='ubiquitin',
        sequence='MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG',
        label=0,
        group='negative',
    ),
    LabelledSequence(
        name='GB1_domain',
        sequence='MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE',
        label=0,
        group='negative',
    ),
    LabelledSequence(
        name='barnase',
        sequence='AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR',
        label=0,
        group='negative',
    ),
    LabelledSequence(
        name='BPTI',
        sequence='RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA',
        label=0,
        group='negative',
    ),
    LabelledSequence(
        name='IL2',
        sequence='APTSSSTKKTQLQLEHLLLDLQMILNGINNYKNPKLTRMLTFKFYMPKKATELKHLQCLEEELKPLEEVLNLAQSKNFHLRPRDLISNINVIVLELKGSETTFMCEYADETATIVEFLNRWITFCQSIISTLT',
        label=0,
        group='negative',
    ),
    LabelledSequence(
        name='SpA_Z_domain',
        sequence='VDNKFNKEQQNAFYEILHLPNLNEEQRNGFIQSLKDDPSQSTNVLGEAQKLNDSQAPK',
        label=0,
        group='negative',
    ),
    LabelledSequence(
        name='E_coli_DHFR',
        sequence='MISLIAALAVDRVIGMENAMPWNLPADLAWFKRNTLNKPVIMGRHTWESIGRPLPGRKNIILSSQPSEECATMVTNGKYLGELDLGEEFNLHQAEKVQEHFGLQDNLNQIQRNLRDVVAAFVNNQLNRQKQILSRQNMSTFNDIIEDTLESQLSFLVSNFDYSKAKLKEKVTLLLNRFKELPGLKYRIAGCASAGGHTIEEFMQKLNNNLSVSRSAAAQTFSNQPEEGHVSQLVIRGPLEDLTPELDNFTLDPFPQRFTTLPWDNQLVSYLFPQEGTKAVKNPVNIWKGEQLKNQRQEVNEALDQHSFEQLKLSILENRSQFGFLRPIIKVLQQNLNELTSNQEQQDRKLNQPEDPETLANVSNQNVHQYSNRTPSHFLPVNRVPYLQTSGVNIDKKEMLNQTLQGSYQTAAELDLKFNTLNTLEDLFHREYNLKDIHFEYTTELGQLVGKIKSVNADRDFQMDNEEFENFSLTLEQKKVLTKIQNNTLQHIEQSFMSGLHSPNLDLRSEDLKPQMFIQNKQSRTVIQQNQNLENAYIEALKSNFNTEQKNLHQIIQILKDNHILQFHQ',
        label=0,
        group='negative',
    ),
    LabelledSequence(
        name='GFP_chromophore_domain',
        sequence='MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK',
        label=0,
        group='negative',
    ),
]

# ---------------------------------------------------------------------------
# Near-miss sequences (high beta-sheet / disorder, but non-amyloid)
# ---------------------------------------------------------------------------

NEAR_MISS: list[LabelledSequence] = [
    LabelledSequence(
        name='concanavalin_A',
        sequence='ADTIVAVELDTYPDAQSATVISWRTQIRNTNKSESPKFGQAALVTEAKSQLVLNKSTQNAAIFNVDLSSWDAIRGSSATGGRLQLNQGDALPSSNTGIIFAIPSSIKYSDSSFYLNKVQGKEGKLLASVHKVAPQNPDNLAQGSGKLHFKVLNRDGQSGSDISPIVIASRRSSETLSMAGGTVSTAEQGAWKRRGQFPPDKKTTQMLIFDAASTVNKTLAIPFHQRLLYVPQAAHTVSAINNFGMEGGLMDMAADLTSQIVEGVNRTTINVVVEEGLHKVSAIAIQFKMNATSQFKNGEFRSAIEGRGKITVEEMEPLLMPTVTYISAFFTYGAEADRSQLQLNHQGLFHFNAEFKIGNNAEGLNFNHEGSDSLF',
        label=0,
        group='near_miss',
    ),
    LabelledSequence(
        name='immunoglobulin_VH',
        sequence='EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR',
        label=0,
        group='near_miss',
    ),
    LabelledSequence(
        name='WW_domain',
        sequence='GSKLPPGWEKRMSRSSGRVYYFNHITNASQWERPSG',
        label=0,
        group='near_miss',
    ),
    LabelledSequence(
        name='SH3_domain_src',
        sequence='MAEYVRALFDFNGNDEEDLPFKKGDILKVLNEECGEWWKARLENGKETLLKEYREDGAPIQGLEWDQSTKTGAQFPGTGVTTYF',
        label=0,
        group='near_miss',
    ),
    LabelledSequence(
        name='streptavidin_monomer',
        sequence='MRKIVVAAIAVSLTTVSITASASADPSKDSKAQVSAAEAGITGTWYNQLGSTFIVTAGADGALTGTYELAPFLRSQTPNEVPVKAGAWTYNDYGLIAVDNSTVGIGTFYKVGATGIAQVTNTPGQTDYIQTPAGAVGATLSPQAPSGKDNGGQTALNVEVPDLNPFQNLKAAAQAAGFNVTSMTLQNL',
        label=0,
        group='near_miss',
    ),
]


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class AggregationTestDataset:
    """Container for the full labelled dataset."""

    def __init__(self) -> None:
        self.sequences: list[LabelledSequence] = POSITIVE + NEGATIVE + NEAR_MISS

    @property
    def positives(self) -> list[LabelledSequence]:
        return [s for s in self.sequences if s.label == 1]

    @property
    def negatives(self) -> list[LabelledSequence]:
        return [s for s in self.sequences if s.label == 0]

    def __len__(self) -> int:
        return len(self.sequences)

    def __iter__(self):
        return iter(self.sequences)
