# sequenceCNNannot
genomic sequence -> CNN -> annotation

## Citation

If you use these scripts, please cite:

Dey, K.K. et al. 2020. Evaluating the informativeness of deep learning annotations for human complex diseases. bioRxiv (accepted in principle, Nat Commun), p.784439.

## Scripts 

**assemble_encoded_sequences.py**:  For two annotations, generate representative sequences around SNPs labeled as 1 by the two annotations.Separate the training sequence data into odd and evcen chromosomes.

**bed_to_encoded_sequence.py** : For a bed file, generate representative sequence set for regions in the bed file and regions complementary to the bed file.

**fit_cnn.py** : Fit a 2-class (BiClassCNN) convolutional neural net on the sequence data.

**generate_commonvars.py** : Generate encoded sequence for each commonly varying SNP. Mainly used to generate predictions at these SNP locations.

**predict_annot.py** : Predict at the common variants generated from the previous (**generate_commonvars.py**) using the trained CNN model (**fit_cnn.py**)

These annotations can then be used for S-LDSC.

## How to use these annotations?

1) Download the LDSC from git (https://github.com/bulik/ldsc/wiki/Partitioned-Heritability)
2) Download the baselineLD_v2.1 annotations from Broad webpage (https://data.broadinstitute.org/alkesgroup/LDSCORE/)
3) Download deep learning predictive annotations (see above) from https://data.broadinstitute.org/alkesgroup/LDSCORE/Dey_DeepLearning
4) Use your GWAS summary statistics formatted in LDSC details is available (https://github.com/bulik/ldsc/wiki/Summary-Statistics-File-Format)
5) Download the baseline frq file and weights for 1000G available (https://data.broadinstitute.org/alkesgroup/LDSCORE/)
6) Run S-LDSC with these annotations conditional on baselineLD_v2.1 (see https://github.com/bulik/ldsc/)

```
ANNOT FILE header (*.annot):

CHR -- chromosome
BP -- physical position (base pairs)
SNP -- SNP identifier (rs number)
CM -- genetic position (centimorgans)
all additional columns -- Annotations
```

NOTE: Although one would expect the genetic position to be non-negative for all 1000G SNPs, we have checked that
in fact the genetic position is negative for a handful of 1000G SNPs that have a physical position that is smaller
than the smallest physical position in the genetic map. The genetic positions were obtained by running PLINK on
the Oxford genetic map (http://www.shapeit.fr/files/genetic_map_b37.tar.gz).

MORE DETAIL CAN BE OBTAINED FROM https://github.com/bulik/ldsc/wiki/LD-File-Formats


```
LD SCORE FILE header (*.l2.ldscore):

CHR -- chromosome
BP -- physical position (base pairs)
SNP -- SNP identifier (rs number)
all additional columns -- LD Scores

```

## Contact 

In case of any questions, please open an issue or send an email to me at `kdey@hsph.harvard.edu`.












