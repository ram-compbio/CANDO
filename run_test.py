import sys, os
import cando as cnd
pre = os.path.dirname(cnd.__file__)
print(pre)
print('\n')
cnd.get_test()
cnd.get_data(v='test.0', org='test')
os.chdir("cando/data/v2.2+/test")
print('\n')

# Set all variables
matrix_file = 'test-matrix.tsv'
inds_map = 'test-inds.tsv'
cmpd_map = 'test-cmpds.tsv'
cmpd_dir = 'test-cmpds_mol/'
new_cmpds = 'test-new_cmpds.tsv'

#cmpd_scores = 'test-cmpd_scores.tsv'
#prot_scores = 'test-prot_scores.tsv'
pwp = 'test-pathway-prot.tsv'
pwm = 'test-pathway-mesh.tsv'
ncpus = 3

print("Test #1 - generate a toy matrix")
print('-------')
cnd.generate_matrix(v="test.0", org="test", out_file=matrix_file, ncpus=ncpus)
cnd.generate_matrix(v="test.0", org="test", out_file="test-matrix_names.tsv", lig_name=True, ncpus=ncpus)
print('\n')

print("Test #2 - create CANDO object and run canbenchmark test")
print('-------')
cando = cnd.CANDO(cmpd_map, inds_map, matrix=matrix_file, compute_distance=True,
                  save_dists='test_rmsds.tsv', ncpus=ncpus)
cando.canbenchmark('test')
print('\n')

print("Test #3 - create CANDO object using cosine distance metric then run continuous, bottom, cluster, "
      "and associated benchmark test with 'sort' ranking")
print('-------')
cando_cos = cnd.CANDO(cmpd_map, inds_map, matrix=matrix_file, compute_distance=True, dist_metric='cosine', ncpus=ncpus)
cando_cos.canbenchmark('test_continuous', continuous=True, ranking='ordinal')
cando_cos.canbenchmark_associated('test_associated', ranking='modified')
cando_cos.canbenchmark_bottom('test_bottom', ranking='standard')
cando_cos.canbenchmark_bottom('test_bottom', ranking='modified')
cando_cos.canbenchmark_bottom('test_bottom', ranking='ordinal')
cando_cos.canbenchmark_cluster(n_clusters=5)
print('\n')

print("Test #4 - canpredict_compounds()")
print("\tpredict top10 compounds for Breast Cancer")
print('-------')
cando.canpredict_compounds("MESH:D001943", n=10, topX=10)
cando.canpredict_compounds("MESH:D001943", n=10, topX=25, keep_associated=True)
print('\n')

print("Test #5 - generate interaction signature for a new compound")
print('-------')
cnd.generate_signature(cmpd_file="{}/8100.mol".format(cmpd_dir), org="test", out_file="test-8100_signature.tsv", out_path=".")
#cnd.generate_scores(fp="ob_fp4", cmpd_pdb="8100.pdb", out_path=".")
#cnd.generate_scores(fp="rd_ecfp4", cmpd_pdb="8100.pdb", out_path=".")
#cnd.generate_signature(cmpd_scores="rd_ecfp4/8100_scores.tsv", prot_scores=prot_scores)
print('\n')

print("Test #6 - Most similar compounds to new compound and CANDO compound signatures")
print('-------')
cando.add_cmpd("test-8100_signature.tsv", "mk-0249")
tc = cando.get_compound(64)
cando.similar_compounds(tc, n=10)
cando.similar_compounds(cando.compounds[10], n=10)
print('\n')

print("Test #7 - canpredict_indications() for new compound and CANDO compound signatures")
print('-------')
cando.canpredict_indications(tc, n=10)
cando.canpredict_indications(cando.compounds[10], n=10)
print('\n')

print("Test #8 - use customized protein set with 20 UniProt IDs, use benchmark with SVM ML code")
print('-------')
cando_uni = cnd.CANDO(cmpd_map, inds_map, matrix=matrix_file, compute_distance=True, protein_set="test-uniprot_set.tsv")
cando_uni.ml(benchmark=True, method='rf', seed=50, out='test_svm')
print('\n')

print("Test #9 - use random forest ML code to make predictions for Inflammation for two compounds")
print('-------')
cando_rf = cnd.CANDO(cmpd_map, inds_map, compute_distance=True, matrix=matrix_file)
inflm = cando_rf.get_indication('MESH:D007249')
lys = cando_rf.get_compound(18)
men = cando_rf.get_compound(62)
cando_rf.ml(effect=inflm, predict=[lys, men], method='rf')
print('\n')

print("Test #10 - read .fpt matrices, convert_to_tsv, then fuse with 'mult'")
print('-------')
cnd.Matrix("toy64x.fpt", convert_to_tsv=True)
cnd.Matrix("vina64x.fpt", convert_to_tsv=True)
toy64 = cnd.CANDO(cmpd_map, inds_map, matrix='toy64x.tsv', compute_distance=True, save_dists='toy64x_rmsds.tsv')
vina64 = cnd.CANDO(cmpd_map, inds_map, matrix='vina64x.tsv', compute_distance=True)
vina64.normalize()
print(vina64)
v2rmsd = cnd.CANDO(cmpd_map, inds_map, read_dists='test_rmsds.tsv')
fus_mult = toy64.fusion([vina64], method='mult')
fus_sum = v2rmsd.fusion([vina64], method='sum')
tr = cnd.Matrix('toy64x_rmsds.tsv', dist=True)
tr.convert('toy64x_sim.tsv')
print('\n')

print("Test #11 - Check other download functions")
print('-------')
#cnd.get_tutorial()
cnd.get_data(v="v2.2", org="all")
print('\n')

print("Test #12 - Pathways data plus benchmark")
print('-------')
cpw = cnd.CANDO(cmpd_map, inds_map, matrix=matrix_file, pathways=pwp, pathway_quantifier='max', compute_distance=True)
cpw.canbenchmark('test-pw')
cpwi = cnd.CANDO(cmpd_map, inds_map, matrix=matrix_file, pathways=pwp, pathway_quantifier='proteins',
                 indication_pathways=pwm)
cpwi.canbenchmark('test-pw_inds')
print('\n')

print("Test #13 - canpredict_denovo() with breast cancer + all proteins and subset of proteins")
print('-------')
cando.canpredict_denovo(method='count', threshold=0.2, ind_id="MESH:D001943", topX=10, save='test-cpdn.tsv')
cando.canpredict_denovo(method='sum', threshold=0.2, topX=10, proteins=['2y9kA', '1k3zD', '1oegA', '4uotA', '2cs0A'])
print('\n')

print("Test #14 - Search for objects")
print('-------')
cando.search_indication('breast neoplasms')
cando.search_compound('bivlarudn')  # intended typo
cando.get_protein('2y9kA')
print('\n')

print("Test #15 - Add and save compound then generate new matrix")
print('-------')
cnd.add_cmpds(new_cmpds, cmpd_dir=cmpd_dir, v="test.0")
cnd.generate_matrix(v="test.1", org="test", out_file=matrix_file, ncpus=ncpus)
print('\n')

print("Test #16 - Add and save compound to new library then generate new matrix")
print('-------')
cnd.add_cmpds(new_cmpds, cmpd_dir=cmpd_dir)
cnd.generate_matrix(v="v0.0", org="test", lib_path='.', out_file=matrix_file, ncpus=ncpus)
#os.system("rm -r {}/data/v2.2+/cmpds/fps-v0.0".format(pre))
#os.system("rm -r {}/data/v2.2+/mappings/*-v0.0.tsv".format(pre))
print('\n')

