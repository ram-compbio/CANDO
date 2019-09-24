import sys, os
import cando as cnd
print(os.path.dirname(cnd.__file__))
cnd.get_test()
os.chdir("test")
print('\n')

# Set all variables
matrix_file='test-matrix.tsv'
inds_map='test-inds.tsv'
cmpd_map='test-cmpds.tsv'
cmpd_dir='test-cmpds_pdb/'
cmpd_scores='test-cmpd_scores.tsv'
prot_scores='test-prot_scores.tsv'
ncpus = 3

print("Test #1 - generate a toy matrix")
print('-------')
cnd.generate_matrix(matrix_file=matrix_file, 
        cmpd_scores=cmpd_scores, prot_scores=prot_scores, 
        ncpus=ncpus)
print('\n')

print("Test #2 - create CANDO object and run classic benchmark test")
print('-------')
cando = cnd.CANDO(cmpd_map, inds_map, matrix=matrix_file, 
        compute_distance=True, ncpus=ncpus)
cando.benchmark_classic('test', SUM='summary-test')
print('\n')

print("Test #3 - create CANDO object using cosine distance metric then run continuous and associated benchmark test with 'sort' ranking")
print('-------')
cando_cos = cnd.CANDO(cmpd_map, inds_map, matrix=matrix_file, 
        compute_distance=True, dist_metric='cosine',
        ncpus=ncpus)
cando_cos.benchmark_continuous('test_continuous', 
        SUM='summary-test_continuous', ranking='sort')
cando_cos.benchmark_associated('test_associated', 
        SUM='summary-test_associated', ranking='sort')
print('\n')

print("Test #4 - canpredict_compounds()")
print("\tpredict top10 compounds for Breast Cancer")
print('-------')
cando.canpredict_compounds("MESH:D001943", n=10, topX=10)
cando.canpredict_compounds("MESH:D001943", n=10, topX=25, keep_approved=True)
print('\n')

print("Test #5 - generate fingerprint, scores, and interaction signature for three old compounds and one new compound")
print('-------')
cnd.generate_scores(fp="ob_fp4",
        cmpd_pdb="8100.pdb",out_path=".")
cnd.generate_scores(fp="rd_ecfp4",
        cmpd_pdb="8100.pdb",out_path=".")
cnd.generate_signature(cmpd_scores="rd_ecfp4/8100_scores.tsv",
        prot_scores=prot_scores, ncpus=ncpus)
print('\n')

print("Test #6 - Most similar compounds to new compound and CANDO compound signatures")
print('-------')
cando.similar_compounds(new_sig="8100_signature.tsv", 
        new_name='scy-635', n=10)
cando.similar_compounds(cando_cmpd=cando.compounds[10], 
        n=10)
print('\n')

print("Test #7 - canpredict_indications() for new compound and CANDO compound signatures")
print('-------')
cando.canpredict_indications(new_sig="8100_signature.tsv", 
        new_name='scy-635', n=10)
cando.canpredict_indications(cando_cmpd=cando.compounds[10], 
        n=10)
print('\n')

print("Test #8 - use customized protein set with 20 UniProt IDs, use benchmark with SVM ML code")
print('-------')
cando_uni = cnd.CANDO(cmpd_map, inds_map, matrix=matrix_file, protein_set="test-uniprot_set")
cando_uni.ml(benchmark=True, method='svm', seed=50, out='test_svm')
print('\n')

print("Test #9 - use random forest ML code to make predictions for Inflammation for two compounds")
print('-------')
cando_rf = cnd.CANDO(cmpd_map, inds_map, matrix=matrix_file)
inflm = cando_rf.get_indication('MESH:D007249')
lys = cando_rf.get_compound(18)
men = cando_rf.get_compound(62)
cando_rf.ml(effect=inflm, predict=[lys, men], method='rf')
print('\n')
