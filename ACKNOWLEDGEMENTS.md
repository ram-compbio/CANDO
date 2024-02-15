# Acknowledgements

The success of the CANDO platform goes beyond the developers; here we acknowledge and show our appreciation for the funding opportunities and research done by external groups that have enabled the development and application of CANDO.

## Funding

We have received numerous grants and awards to pursue the development and application of the CANDO platform. Below is a list of these funding sources:

- University at Buffalo Accelerator Fund (2021).
- University at Buffalo CTSA pilot (2020-2021).
- US NSF SBIR subcontract from Onai, Inc. (2020).
- US NIH NCATS ASPIRE Design Challenge Award (2020-2022).
- US VHA Big Data Scientist Training Enhancement Program (2016-2018).
- US NIH Buffalo Research Innovation in Genomic and Healthcare Technology (BRIGHT) Education Award T15LM012495 (2016-2021).
- US NIH Clinical and Translational Sciences Award UL1TR001412 (2015-2020).
- US NIH Director's Pioneer Award 7DP1OD006779 (2010-2017).
- US NSF GEMSEC (2005-2011).
- US NSF CAREER Award IIS-0448502 (2005-2010).
- US NIH F30DE017522 (2006-2010).
- The University of Washington's Technology Gap Innovation Fund (2006-2007).
- Washington Research Foundation (2006-2007).
- Puget Sound Partners in Global Health (2004-2005).
- US NIH R33 (2003-2006)
- Searle Scholar Award to Ram Samudrala (2002-2005).
- The University of Washington's Advanced Technology Initiative in Infectious Diseases (2001-2014).

## External software and data sources

We leverage data and software from other groups and institutions towards the success of this platform. We would like to acknowledge those entities and cite their work when appropriate.

### I-TASSER Suite

[I-TASSER](https://zhanggroup.org/I-TASSER/) was used to generate high confidence structural models of proteins found within some of the CANDO libraries. [COACH](https://zhanggroup.org/COACH/) was used to predict binding sites and ligands for proteins in the protein libraries.

References:
- X Zhou, W Zheng, Y Li, R Pearce, C Zhang, EW Bell, G Zhang, Y Zhang. I-TASSER-MTD: a deep-learning-based platform for multi-domain protein structure and function prediction. Nature Protocols, 17: 2326-2353 (2022)
- W Zheng, C Zhang, Y Li, R Pearce, EW Bell, Y Zhang. Folding non-homology proteins by coupling deep-learning contact maps with I-TASSER assembly simulations. Cell Reports Methods, 1: 100014 (2021)
- J Yang, Y Zhang. I-TASSER server: new development for protein structure and function predictions. Nucleic Acids Research, 43: W174-W181 (2015)
- J Yang, A Roy, Y Zhang. Protein-ligand binding site recognition using complementary binding-specific substructure comparison and sequence profile alignment, Bioinformatics, 29:2588-2595 (2013)
- J Yang, A Roy, Y Zhang. BioLiP: a semi-manually curated database for biologically relevant ligand-protein interactions, Nucleic Acids Research, 41: D1096-D1103 (2013)

*Disclaimer*: 
Users should obtain an [academic license](https://zhanggroup.org/I-TASSER/download/) for the I-TASSER suite to use and/or generate protein and binding site related data for the CANDO platform.

### RDKit

[RDKit](https://www.rdkit.org) was used to optimize compound structures and generate chemical fingerprints for the BANDOCK protocol.

References:
- None. There has been no official publication of the package.

### AlphaFold

AlphaFold2 ([software](https://github.com/google-deepmind/alphafold) and [publicly available database](https://alphafold.ebi.ac.uk/) was used to build comprehensive libraries of proteins, e.g. homo sapien, etc.

References:
- Jumper, J et al. Highly accurate protein structure prediction with AlphaFold. Nature (2021).
- Varadi, M et al. AlphaFold Protein Structure Database: massively expanding the structural coverage of protein-sequence space with high-accuracy models. Nucleic Acids Research (2021).

### DrugBank

The extensive database from [DrugBank](https://go.drugbank.com/) was used to extract chemical compounds for our compound libraries, identify metadata for the compounds, and generate mappings between drugs and indications.

References:
- Wishart DS, Feunang YD, Guo AC, Lo EJ, Marcu A, Grant JR, Sajed T, Johnson D, Li C, Sayeeda Z, Assempour N, Iynkkaran I, Liu Y, Maciejewski A, Gale N, Wilson A, Chin L, Cummings R, Le D, Pon A, Knox C, Wilson M. DrugBank 5.0: a major update to the DrugBank database for 2018. Nucleic Acids Res. 2017 Nov 8. doi: 10.1093/nar/gkx1037.
- Law V, Knox C, Djoumbou Y, Jewison T, Guo AC, Liu Y, Maciejewski A, Arndt D, Wilson M, Neveu V, Tang A, Gabriel G, Ly C, Adamjee S, Dame ZT, Han B, Zhou Y, Wishart DS. DrugBank 4.0: shedding new light on drug metabolism. Nucleic Acids Res. 2014 Jan 1;42(1):D1091-7. PubMed: 24203711
- Knox C, Law V, Jewison T, Liu P, Ly S, Frolkis A, Pon A, Banco K, Mak C, Neveu V, Djoumbou Y, Eisner R, Guo AC, Wishart DS. DrugBank 3.0: a comprehensive resource for 'omics' research on drugs. Nucleic Acids Res. 2011 Jan;39(Database issue):D1035-41. PubMed: 21059682
- Wishart DS, Knox C, Guo AC, Cheng D, Shrivastava S, Tzur D, Gautam B, Hassanali M. DrugBank: a knowledgebase for drugs, drug actions and drug targets. Nucleic Acids Res. 2008 Jan;36(Database issue):D901-6. PubMed: 18048412
- Wishart DS, Knox C, Guo AC, Shrivastava S, Hassanali M, Stothard P, Chang Z, Woolsey J. DrugBank: a comprehensive resource for in silico drug discovery and exploration. Nucleic Acids Res. 2006 Jan 1;34(Database issue):D668-72.

### Comparative Toxicogenomic Database (CTD)

The [CTD](https://ctdbase.org/) was used to extract drug-indication mappings for all approved drugs in our compound libraries.

References:
- Davis AP, Wiegers TC, Johnson RJ, Sciaky D, Wiegers J, Mattingly CJ Comparative Toxicogenomics Database (CTD): update 2023. Nucleic Acids Res. 2022 Sep 28.

