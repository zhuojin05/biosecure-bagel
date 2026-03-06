"""
One-time script to generate pre-computed ESM2 mean embeddings for the
reference aggregating sequences and save them as a .npy file.

Usage
-----
    # With Modal (no local GPU needed):
    uv run python scripts/generate_reference_embeddings.py --use_modal True

    # With local GPU:
    uv run python scripts/generate_reference_embeddings.py --use_modal False

The output file is committed to the repository so that EmbeddingSignal works
without re-running this script.
"""

from __future__ import annotations

import pathlib as pl

import numpy as np

import bagel as bg
from bagel.screening.data.reference_sequences import REFERENCE_SEQUENCES

_OUTPUT_PATH = pl.Path(__file__).parent.parent / 'src' / 'bagel' / 'screening' / 'data' / 'reference_embeddings.npy'


def generate(use_modal: bool = False) -> None:
    print(f'Generating reference embeddings for {len(REFERENCE_SEQUENCES)} sequences ...')

    oracle = bg.oracles.ESM2(use_modal=use_modal)

    embeddings_list: list[np.ndarray] = []
    for name, seq in REFERENCE_SEQUENCES:
        print(f'  Embedding {name!r} ({len(seq)} residues) ...')
        chain = bg.Chain(
            residues=[bg.Residue(name=aa, chain_ID='A', index=i, mutable=True) for i, aa in enumerate(seq)]
        )
        result = oracle.predict([chain])
        # mean-pool residue embeddings → shape (D,)
        mean_emb: np.ndarray = np.mean(result.embeddings, axis=0)
        embeddings_list.append(mean_emb)

    ref_array = np.stack(embeddings_list, axis=0)  # shape: (n_refs, D)
    print(f'Reference array shape: {ref_array.shape}')

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(_OUTPUT_PATH, ref_array)
    print(f'Saved to {_OUTPUT_PATH}')


if __name__ == '__main__':
    import fire

    fire.Fire(generate)
