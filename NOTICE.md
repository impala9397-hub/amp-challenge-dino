# Third-party notices

The MIT license in `LICENSE` covers this repository's own source code and the
trained generator checkpoint. The files below come from elsewhere and keep their
own terms.

## Redistributed data

| File | Origin | Terms | Role here |
|---|---|---|---|
| `data/antibacterial.fasta` | [szczurek-lab/amp-challenge-2027](https://github.com/szczurek-lab/amp-challenge-2027), commit `258b661cff7099703120d17354e6c9bd3e458e0f` | BSD-3-Clause (`licenses/amp-challenge-2027-BSD-3-Clause.txt`) | Training corpus for the generator, and the reference set for the novelty gate |
| `data/marlys-v3-sequences.txt.gz` | [MarLys v3](https://data.mendeley.com/datasets/w4hb5grjwb/3), DOI `10.17632/w4hb5grjwb.3` | CC0 1.0 (`licenses/marlys-v3-CC0.txt`) | Exclusion list only. Never used for training |

## Training data disclosure

The generator was trained **only** on the training partition of the organizers'
corpus: 37,935 sequences for training and 1,513 for validation, out of the 39,448
in `data/antibacterial.fasta`. The corpus should not be described as 39,448
training examples.

No private, unpublished or otherwise non-public dataset was used at any point.
AMPSphere and other expansion corpora were evaluated during development and were
**not** used to train the released checkpoint. There is consequently no
non-public training data to release alongside this repository.

MarLys v3 enters only as an exclusion list, to keep the library free of exact
matches against the reference set and its supersets. Using a sequence list to
*exclude* candidates is not training on it.

## Numeric tables

The Eisenberg hydrophobicity values and the Bjellqvist pKa values in
`src/dino_amp/properties.py` are published constants, cited in that file. They
are transcribed to match the values used by `modlamp`, which is what the
challenge's scoring stack calls; `tests/test_properties.py` verifies exact
agreement. Where `modlamp`'s table differs from the original publication's
printed precision, this repository follows `modlamp` and says so in the code.

## Dependencies deliberately avoided

Two widely used packages were replaced rather than depended on, so that
installing this repository does not pull in copyleft terms:

- `python-Levenshtein` is GPL-2.0-or-later. The Levenshtein ratio is provided by
  `rapidfuzz` (MIT), with a pure-Python implementation in `similarity.py` as a
  fallback. `tests/test_similarity.py` pins them to each other.
- `modlamp` is BSD-3-Clause but depends on `mysql-connector-python` (GPL-2.0).
  Its three descriptors are reimplemented in `properties.py`. `modlamp` remains
  a development-only dependency, used solely to test that agreement.

## Scope of these notices

These notices record the sources and terms we verified. They are not a legal
clearance, and they do not assert that every record in every upstream database
has been cleared by its original depositor.
