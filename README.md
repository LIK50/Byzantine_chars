# Character Recognition in Byzantine seals


This repository is related to the article

 > *Théophile Rageau Laurence Likforman-Sulem, Attilio Fiandrotti, Victoria Eyharabide, Béatrice Caseau, Jean-Claude Cheynet*, **Character Recognition in Byzantine Seals with Deep Neural Networks, Digital Applications in Archaeology and Cultural Heritage**, Volume 37, June 2025.[![DOI:10.1016/j.daach.2025.e00403](https://zenodo.org/badge/DOI/10.1016/j.daach.2025.e00403.svg)](https://doi.org/10.1016/j.daach.2025.e00403)


## Classification

This repository demonstrates the inference part of the classification step described in the article.

The `images` directory contains 2224 cropped characters (28 classes), including the background classe `bg`.

The classes are:

 - `C moon-shaped sigma`
 - `CT ligature`
 - `Croisette`
 - `R = Beta`
 - `S = καί`
 - `V = Y`
 - `bg`
 - `ligature OU`
 - `sigma`
 - `Α`
 - `Β`
 - `Γ`
 - `Δ`
 - `Ε`
 - `Ζ`
 - `Η`
 - `Θ`
 - `Ι`
 - `Κ`
 - `Λ`
 - `Μ`
 - `Ν`
 - `Ξ`
 - `Ο`
 - `Π`
 - `Ρ = rho`
 - `Τ`
 - `Φ`
 - `Χ`
 - `ω`
 - `И = N`

### Usage

You can see a demonstration of the classification by using the `read_seals.py`:

```bash
# install requirements
pip install -r requirements.txt

# run the classifier
python3 read_seals.py
```

You can change on which image the classifier is run by editing the `read_seals.py` script.


## Full inference pipeline

To demonstrate the whole pipeline of inference consisting of the 3 steps (detection, classification, transcription) that allows going from an image of a seal to the extracted character sequence, another repository has been set-up here:

### https://gitub.com/idsinge/ocrseals
