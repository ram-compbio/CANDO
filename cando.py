import os, sys, requests, random, time, operator, math, io, copy
from math import sqrt
import progressbar
import gzip, shutil
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import openbabel, pybel
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.Fingerprints import FingerprintMols
from sklearn.metrics import mean_squared_error, pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform, cdist

## An object to represent a protein.
#
class Protein(object):
    def __init__(self, id_, sig):
        ## @var id_ 
        #   String representing the PDB 
        #   or UniProt ID for the given protein
        self.id_ = id_
        ## @var sig 
        #   List of scores representing each drug
        #   interaction with the given protein
        self.sig = sig
        ## @var pathways 
        #   List of pathways in which the 
        #   given protein is involved.
        self.pathways = []

## An object to represent a compound. 
class Compound(object):
    def __init__(self, name, id_, index):
        ## @var name 
        # (e.g., 'caffeine')
        self.name = name
        ## @var id_
        # cando id from mapping file (e.g., 1, 10, 100, ...)
        self.id_ = id_
        ## @var index
        # the order in which they appear from the mapping file (e.g, 1, 2, 3, ...)
        self.index = index
        ## @var sig
        # sig is basically a column of the matrix
        self.sig = []
        ## @var aux_sig
        # potentially temporary signature for things like pathways, where "c.sig" needs to be preserved
        self.aux_sig = []
        ## @var indications
        # this is every indication it is associated with from the
        # mapping file
        self.indications = []
        ## @var similar
        # (list of tuples):
        # this is the ranked list of compounds with the most similar interaction signatures
        self.similar = []
        ## @var similar_computed
        # 
        self.similar_computed = False
        ## @var cluster_id
        # 
        self.cluster_id = []
        ## @var adrs
        # 
        self.adrs = []

    def add_indication(self, ind):
        self.indications.append(ind)

## An object to represent an indication
class Indication(object):
    def __init__(self, ind_id, name):
        self.id_ = ind_id
        self.name = name
        # every associated compound from the mapping file
        self.compounds = []
        self.pathways = []
        self.proteins = []

## An object to represent a pathway
# 
class Pathway(object):
    def __init__(self, id_):
        self.proteins = []
        self.id_ = id_
        ## @var indications
        # in case we ever want to incorporate pathway-disease associations
        self.indications = []

## An object to represent an adverse reaction
class ADR(object):
    def __init__(self, id_, name):
        self.id_ = id_
        self.name = name
        self.compounds = []

## CANDO object    
# This is the highest level object
#
class CANDO(object):
    ## To instantiate you need the cando matrix (matrix),
    # the compound mapping (c_map), and indication mapping files (i_map)
    #
    def __init__(self, c_map, i_map, matrix='', compute_distance=False, save_rmsds='', read_rmsds='',
                 pathways='', pathway_quantifier='max', indication_pathways='', indication_genes='',
                 similarity=False, dist_metric='rmsd', protein_set='', rm_zeros=False, rm_compounds='', ncpus=1,
                 adr_map=''):
        ## @var c_map 
        # compound mapping file
        self.c_map = c_map
        ## @var i_map 
        # indication mapping file
        self.i_map = i_map
        ## @var matrix 
        # cando matrix file
        self.matrix = matrix
        ## @var protein_set
        #
        self.protein_set = protein_set
        self.proteins = []
        self.protein_id_to_index = {}
        self.compounds = []
        self.indications = []
        self.indication_ids = []
        ## @var pathways
        #
        self.pathways = []
        self.accuracies = {}
        ## @var compute_distance
        #
        self.compute_distance = compute_distance
        self.clusters = {}
        ## @var rm_zeros
        #
        self.rm_zeros = rm_zeros
        ## @var rm_compounds
        #
        self.rm_compounds = rm_compounds
        self.rm_cmpds = []
        ## @var save_rmsds
        #
        self.save_rmsds = save_rmsds
        ## @var read_rmsds
        #
        self.read_rmsds = read_rmsds
        ## @var similarity
        #
        self.similarity = similarity
        ## @var dist_metric
        #
        self.dist_metric = dist_metric
        ## @var ncpus
        #
        self.ncpus = int(ncpus)
        ## @var pathway_quantifier
        #
        self.pathway_quantifier = pathway_quantifier
        ## @var indication_pathways
        #
        self.indication_pathways = indication_pathways
        ## @var indication_genes
        #
        self.indication_genes = indication_genes
        ## @var adr_map
        #
        self.adr_map = adr_map
        self.adrs = []

        self.short_matrix_path = self.matrix.split('/')[-1]
        self.short_read_rmsds = read_rmsds.split('/')[-1]
        self.short_protein_set = protein_set.split('/')[-1]
        self.cmpd_set = rm_compounds.split('/')[-1]
        self.data_name = ''

        if self.matrix:
            if self.protein_set:
                self.data_name = self.short_protein_set + '.' + self.short_matrix_path
            elif rm_compounds:
                self.data_name = self.cmpd_set + '.' + self.short_matrix_path
        if self.short_read_rmsds:
            self.data_name = self.short_read_rmsds

        # create all of the compound objects from the compound map
        with open(c_map, 'r') as c_f:
            for l in c_f.readlines():
                ls = l.split('\t')
                if len(ls) == 3:
                    name = ls[2][:-1]
                    id_ = int(ls[1])
                    index = int(ls[0])
                # Used for the v2 mappings
                # These only have 2 columns [id/index,name]
                elif len(ls) == 2:
                    name = ls[1][:-1]
                    id_ = int(ls[0])
                    index = int(ls[0])
                else:
                    print("Check the number of columns for the compound mapping.")
                cm = Compound(name, id_, index)
                self.compounds.append(cm)

        # create the indication objects and add indications to the
        # already created compound objects from previous loop
        # NOTE: if a compound is in the indication mapping file that
        # isn't in the compound mapping file, an error will occur. I
        # had to remove those compounds from the indication mapping in
        # order for it to work
        with open(i_map, 'r') as i_f:
            for l in i_f.readlines():
                ls = l.strip().split('\t')
                c_name = ls[0]
                c_id = int(ls[1])
                i_name = ls[2]
                ind_id = ls[3]
                cm = self.get_compound(c_id)
                if cm:
                    if ind_id in self.indication_ids:
                        ind = self.get_indication(ind_id)
                        ind.compounds.append(cm)
                    else:
                        ind = Indication(ind_id, i_name)
                        ind.compounds.append(cm)
                        self.indications.append(ind)
                        self.indication_ids.append(ind.id_)
                    cm.add_indication(ind)

        # add proteins, add signatures and such to compounds
        if self.protein_set:
            uniprots = []
            with open(self.protein_set, 'r') as psf:
                lines = psf.readlines()
                for line in lines:
                    uni = line.strip()
                    uniprots.append(uni)

        if matrix:
            if matrix[-4:] == '.fpt':
                print('The matrix file {} is in the old fpt format -- please '
                      'convert to tsv with the following line of code:'.format(matrix))
                print('-> Matrix({}, convert_to_tsv=True) <-'.format(matrix))
                quit()
            print('Reading signatures from matrix...')

            with open(matrix, 'r') as m_f:
                m_lines = m_f.readlines()
                if self.protein_set:
                    print('Editing signatures according to proteins in {}...'.format(self.protein_set))
                    targets = self.uniprot_set_index(self.protein_set)
                    new_i = 0
                    for l_i in range(len(m_lines)):
                        vec = m_lines[l_i].strip().split('\t')
                        name = vec[0]
                        if name in targets:
                            scores = list(map(float, vec[1:]))
                            p = Protein(name, scores)
                            self.proteins.append(p)
                            self.protein_id_to_index[name] = new_i
                            for i in range(len(scores)):
                                s = scores[i]
                                self.compounds[i].sig.append(s)
                            new_i += 1
                        else:
                            continue
                else:
                    for l_i in range(len(m_lines)):
                        vec = m_lines[l_i].strip().split('\t')
                        name = vec[0]
                        scores = list(map(float, vec[1:]))
                        p = Protein(name, scores)
                        self.proteins.append(p)
                        self.protein_id_to_index[name] = l_i
                        for i in range(len(scores)):
                            s = scores[i]
                            self.compounds[i].sig.append(s)
            print('Done reading signatures.\n')
        if pathways:
            print('Reading pathways...')
            if self.indication_pathways:
                print('Reading indication-pathway associations...')
                path_ind = {}
                with open(indication_pathways, 'r') as ipf:
                    for l in ipf:
                        ls = l.strip().split('\t')
                        pw = ls[0]
                        ind_ids = ls[1:]
                        path_ind[pw] = ind_ids

            with open(pathways, 'r') as pf:
                for l in pf:
                    ls = l.strip().split('\t')
                    pw = ls[0]
                    ps = ls[1:]
                    if not ps:
                        continue
                    PW = Pathway(pw)
                    self.pathways.append(PW)
                    for p in ps:
                        try:
                            pi = self.protein_id_to_index[p]
                            pro = self.proteins[pi]
                            pro.pathways.append(PW)
                            PW.proteins.append(pro)
                        except KeyError:
                            #print('Could not find protein chain {} for pathway {}'.format(p, pw))
                            pass

                    if self.indication_pathways:
                        try:
                            ind_ids = path_ind[pw]
                            for ind_id in ind_ids:
                                try:
                                    ind = self.get_indication(ind_id)
                                except LookupError:
                                    # print('Could not find indication {}'.format(ind_id))
                                    # this disease is not present in the platform
                                    continue
                                PW.indications.append(ind)
                                ind.pathways.append(PW)
                        except KeyError:
                            # print('Could not find pathway {}'.format(PW.id_))
                            # This pathway is not associated with diseases according to mappings
                            continue
            if not indication_pathways:
                self.quantify_pathways()
            print('Done reading pathways.')

        if self.indication_genes:
            print('Reading indication-gene associations...')
            ind_genes = {}
            with open(indication_genes, 'r') as igf:
                for l in igf:
                    ls = l.strip().split('\t')
                    ind_id = ls[0]
                    genes = ls[1].split(";")
                    for p in genes:
                        try:
                            pi = self.protein_id_to_index[p]
                            pro = self.proteins[pi]
                            ind = self.get_indication(ind_id)
                            ind.proteins.append(pro)
                        except KeyError:
                            #print('Could not find protein chain {} for indication {}'.format(p, ind_id))
                            pass



        if read_rmsds:
            print('Reading RMSDs...')
            with open(read_rmsds, 'r') as rrs:
                lines = rrs.readlines()
                for i in range(len(lines)):
                    c1 = self.compounds[i]
                    scores = lines[i].strip().split('\t')
                    for j in range(len(scores)):
                        if i == j:
                            continue
                            #c1.similar.append((self.compounds[j], -1.0))
                        else:
                            s = float(scores[j])
                            if similarity:
                                s = 1 - s
                            c1.similar.append((self.compounds[j], s))
            for c in self.compounds:
                sorted_scores = sorted(c.similar, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
                c.similar = sorted_scores
                c.similar_computed = True
            print('Done reading RMSDs.')


        if rm_compounds:
            print('Removing undesired compounds in {}...'.format(rm_compounds))
            with open(rm_compounds, 'r') as rcf:
                lines = rcf.readlines()
                for i in range(len(lines)):
                    line = lines[i]
                    cmpd_id = int(line.strip().split('\t')[0])
                    self.rm_cmpds.append(cmpd_id)
            good_cmpds = []
            for c in self.compounds:
                if c.id_ in self.rm_cmpds:
                    pass
                else:
                    good_cmpds.append(c)
            self.compounds = good_cmpds
            for c in self.compounds:
                good_sims = []
                for s in c.similar:
                    if s[0].id_ in self.rm_cmpds:
                        pass
                    else:
                        good_sims.append(s)
                c.similar = good_sims
            #self.compounds = good_cmpds
            print('Done removing undesired compounds.')


        # if compute distance is true, generate similar compounds for each
        if compute_distance:
            if self.pathways and not self.indication_pathways:
                print('Computing distances using global pathway signatures...')
                for c in self.compounds:
                    self.generate_similar_sigs(c, aux=True)
            else:
                print('Computing {} distances...'.format(self.dist_metric))
                # put all compound signatures into 2D-array
                signatures = []
                for i in range(0, len(self.compounds)):
                    signatures.append(self.compounds[i].sig)
                snp = np.array(signatures)  # convert to numpy form

                # call pairwise_distances, speed up with custom RMSD function and parallelism
                if self.dist_metric == "rmsd":
                    distance_matrix = pairwise_distances(snp, metric=lambda u, v: np.sqrt(((u - v) ** 2).mean()), n_jobs=self.ncpus)
                    distance_matrix = squareform(distance_matrix)
                elif self.dist_metric in ['cosine','correlation','euclidean','cityblock']:
                    distance_matrix = pairwise_distances(snp, metric=self.dist_metric, n_jobs=self.ncpus)
                    # Removed checks in case the diagonal is very small (close to zero) but not zero.
                    distance_matrix = squareform(distance_matrix,checks=False)
                else:
                    print("Incorrect distance metric - {}".format(self.dist_metric))
                    exit()

                # step through the condensed matrix - add RMSDs to Compound.similar lists
                N = len(self.compounds)
                n = 0
                for i in range(N):
                    for j in range(i, N):
                        c1 = self.compounds[i]
                        c2 = self.compounds[j]
                        if i == j:
                            #c1.similar.append((c1, -1.0))
                            continue
                        r = distance_matrix[n]
                        c1.similar.append((c2, r))
                        c2.similar.append((c1, r))
                        n += 1
                print('Done computing {} distances.'.format(self.dist_metric))


            if self.save_rmsds:
                def rmsds_to_str(cmpd, ci):
                    o = ''
                    for si in range(len(cmpd.similar)):
                        if ci == si:
                            if self.similarity:
                                o += '1.0\t'
                            else:
                                o += '0.0\t'
                        s = cmpd.similar[si]
                        o += '{}\t'.format(s[1])
                    o = o[:-1] + '\n'
                    return o

                with open(self.save_rmsds, 'w') as srf:
                    for ci in range(len(self.compounds)):
                        c = self.compounds[ci]
                        srf.write(rmsds_to_str(c, ci))
                print('RMSDs saved.')

            # sort the RMSDs after saving (if desired)
            for c in self.compounds:
                #sorted_scores = sorted(c.similar, key=lambda x: x[1])
                sorted_scores = sorted(c.similar, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
                c.similar = sorted_scores
                c.similar_computed = True

        if self.rm_zeros:
            print('Removing compounds with all-zero signatures...')
            def check_sig(sig):
                for s in sig:
                    if s != 0.0:
                        return True
                return False
            non_zero_compounds = []
            for c in self.compounds:
                if check_sig(c.sig):
                    non_zero_compounds.append(c)
            self.compounds = non_zero_compounds
            print('Done removing compounds with all-zero signatures.')

        if self.rm_zeros or self.rm_compounds:
            print('Filtering indication mapping...')
            for ind in self.indications:
                good_cmpds = []
                for cmpd in ind.compounds:
                    if cmpd.id_ in self.rm_cmpds:
                        pass
                    else:
                        good_cmpds.append(cmpd)
                ind.compounds = good_cmpds
            print('Done filtering indication mapping.')

        if adr_map:
            print('Reading ADR mapping file...')
            with open(adr_map, 'r') as amf:
                for l in amf:
                    ls = l.strip().split('\t')
                    c_index = int(ls[1])
                    cmpd = self.compounds[c_index]
                    adr_id = ls[4]
                    adr_name = ls[3]
                    try:
                        adr = self.get_adr(adr_id)
                        adr.compounds.append(cmpd)
                        cmpd.adrs.append(adr)
                    except LookupError:
                        adr = ADR(adr_id, adr_name)
                        adr.compounds.append(cmpd)
                        cmpd.adrs.append(adr)
                        self.adrs.append(adr)
            print('Read {} ADRs.'.format(len(self.adrs)))


    def get_compound(self, id_):
        for c in self.compounds:
            if c.id_ == id_:
                return c
        print("{0} not in {1}".format(id_, self.c_map))
        return None

    def get_indication(self, ind_id):
        for i in self.indications:
            if i.id_ == ind_id:
                return i
        raise LookupError

    def get_pathway(self, id_):
        for p in self.pathways:
            if p.id_ == id_:
                return p
        raise LookupError

    def get_adr(self, id_):
        for a in self.adrs:
            if a.id_ == id_:
                return a
        raise LookupError

    def uniprot_set_index(self, prots):
        if not os.path.exists('v2_0/mappings/pdb_2_uniprot.csv'):
            print('Downloading UniProt to PDB mapping file...')
            url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2_0/mappings/pdb_2_uniprot.csv'
            dl_file(url,'v2_0/mappings/pdb_2_uniprot.csv')
        pdct = {}
        with open('v2_0/mappings/pdb_2_uniprot.csv', 'r') as u2p:
            for l in u2p.readlines()[1:]:
                spl = l.strip().split(',')
                pdb = spl[0] + spl[1]
                uni = spl[2]
                try:
                    if pdb not in pdct[uni]:
                        pdct[uni].append(pdb)
                except KeyError:
                    pdct[uni] = [pdb]
        targets = []
        targets += prots
        with open(prots, 'r') as unisf:
            for lp in unisf:
                try:
                    targets += pdct[lp.strip().upper()]
                except KeyError:
                    pass
        return targets

    # for a given compound, generate the similar compounds based on rmsd of sigs
    # this is canpredict for all intents and purposes
    def generate_similar_sigs(self, cmpd, sort=False, proteins=[], aux=False):
        # find index of query compound, collect signatures for both
        q = 0
        c_sig = []
        if proteins is None:
            c_sig = cmpd.sig
        elif proteins:
            for pro in proteins:
                index = self.protein_id_to_index[pro.id_]
                c_sig.append(cmpd.sig[index])
        else:
            if aux:
                c_sig = cmpd.aux_sig
            else:
                c_sig = cmpd.sig
        ca = np.array([c_sig])

        other_sigs = []
        for ci in range(len(self.compounds)):
            c = self.compounds[ci]
            if cmpd.id_ == c.id_:
                q = ci
            other = []
            if proteins is None:
                other_sigs.append(c.sig)
            elif proteins:
                for pro in proteins:
                    index = self.protein_id_to_index[pro.id_]
                    other.append(c.sig[index])
                other_sigs.append(other)
            else:
                if aux:
                    other_sigs.append(c.aux_sig)
                else:
                    other_sigs.append(c.sig)
        oa = np.array(other_sigs)

        # call cdist, speed up with custom RMSD function
        if self.dist_metric == "rmsd":
            distances = pairwise_distances(ca, oa, lambda u, v: np.sqrt(((u - v) ** 2).mean()), n_jobs=self.ncpus)
            #distances = cdist(ca, oa, lambda u, v: np.sqrt(((u - v) ** 2).mean()))
        elif self.dist_metric in ['cosine','correlation','euclidean','cityblock']:
            distances = pairwise_distances(ca, oa, self.dist_metric, n_jobs=self.ncpus)
            #distances = cdist(ca, oa, self.dist_metric)
        else:
            print("Incorrect distance metric - {}".format(self.dist_metric))

        cmpd.similar = []
        # step through the cdist list - add RMSDs to Compound.similar list
        n = len(self.compounds)
        for i in range(n):
            c2 = self.compounds[i]
            if i == q:
                #c1.similar.append((c1, -1.0))
                continue
            d = distances[0][i]
            cmpd.similar.append((c2, d))
            #c2.similar.append((c1, r))
            n += 1

        if sort:
            sorted_scores = sorted(cmpd.similar, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
            cmpd.similar = sorted_scores
            cmpd.similar_computed = True
            return sorted_scores
        else:
            cmpd.similar_computed = True
            return cmpd.similar


    # for a given list of compounds, generate the similar compounds based on rmsd of sigs
    # this is pathways/genes for all intents and purposes
    def generate_some_similar_sigs(self, cmpds, sort=False, proteins=[], aux=False):
        #ca = []
        #q = []
        #index = []
        
        q = [cmpd.id_ for cmpd in cmpds]
        
        if proteins is None:
            ca = [cmpd.sig for cmpd in cmpds]
            oa = [cmpd.sig for cmpd in self.compounds]
        elif proteins:
            index = [self.protein_id_to_index[pro.id_] for pro in proteins]
            ca = [[cmpd.sig[i] for i in index] for cmpd in cmpds]
            oa = [[cmpd.sig[i] for i in index] for cmpd in self.compounds]
        else:
            if aux:
                ca = [cmpd.aux_sig for cmpd in cmpds]
                oa = [cmpd.aux_sig for cmpd in self.compounds]
            else:
                ca = [cmpd.sig for cmpd in cmpds]
                oa = [cmpd.sig for cmpd in self.compounds]
        ca = np.asarray(ca)
        oa = np.asarray(oa)
        
        '''
        for cmpd in cmpds:
            # find index of query compound, collect signatures for both
            c_sig = []
            if proteins is None:
                c_sig = cmpd.sig
            elif proteins:
                c_sig = [cmpd.sig[i] for i in index]
            else:
                if aux:
                    c_sig = cmpd.aux_sig
                else:
                    c_sig = cmpd.sig
            ca.append(c_sig)
            q.append(cmpd.id_)
        ca = np.asarray(ca)

        other_sigs = []
        for ci in range(len(self.compounds)):
            c = self.compounds[ci]
            other = []
            if proteins is None:
                other_sigs.append(c.sig)
            elif proteins:
                other_sigs.append([c.sig[i] for i in index])
            else:
                if aux:
                    other_sigs.append(c.aux_sig)
                else:
                    other_sigs.append(c.sig)
        oa = np.array(other_sigs)
        ''' 
        # call cdist, speed up with custom RMSD function
        if self.dist_metric == "rmsd":
            distances = pairwise_distances(ca, oa, lambda u, v: np.sqrt(((u - v) ** 2).mean()), n_jobs=self.ncpus)
            #distances = cdist(ca, oa, lambda u, v: np.sqrt(((u - v) ** 2).mean()))
        elif self.dist_metric in ['cosine','correlation','euclidean','cityblock']:
            distances = pairwise_distances(ca, oa, self.dist_metric, n_jobs=self.ncpus)
            #distances = cdist(ca, oa, self.dist_metric)
        else:
            print("Incorrect distance metric - {}".format(self.dist_metric))

        # step through the cdist list - add RMSDs to Compound.similar list
        n = len(self.compounds)
        for j in range(len(cmpds)):
            cmpds[j].similar = []
            for i in range(n):
                c2 = self.compounds[i]
                id2 = c2.id_
                if id2 == q[j]:
                    continue
                d = distances[j][i]
                cmpds[j].similar.append((c2, d))

            if sort:
                sorted_scores = sorted(cmpds[j].similar, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
                cmpds[j].similar = sorted_scores
                cmpds[j].similar_computed = True
            else:
                cmpds[j].similar_computed = True


    # use this method (instead of the 'compute_distance' boolean) in the event that you are
    # manipulating signatures or something but still need to compute the rmsds
    def generate_all_similar_sigs(self):
        print('Computing RMSDs (this will take a while)...')
        for i in range(0, len(self.compounds)):
            if i % 10 == 0:
                print(i)
            for j in range(i, len(self.compounds)):
                c1 = self.compounds[i]
                c2 = self.compounds[j]
                if i == j:
                    #c1.similar.append((c2, -1.0))
                    continue
                else:
                    r = sqrt(mean_squared_error(c1.sig, c2.sig))
                    c1.similar.append((c2, r))
                    c2.similar.append((c1, r))
                    self.all_rmsds.append(r)
            for c in self.compounds:
                sorted_scores = sorted(c.similar, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
                c.similar = sorted_scores
                c.similar_computed = True
        print('Done computing RMSDs.')

    def quantify_pathways(self, indication=None):
        pq = self.pathway_quantifier
        if pq == 'max':
            func = max
        elif pq == 'sum':
            func = sum
        elif pq == 'avg':
            func = np.average
        elif pq == 'proteins':
            if not self.indication_pathways:
                print('Pathway quantifier "proteins" should only be used in combination with a '
                      'pathway-disease mapping (indication_pathways), quitting.')
                quit()
            func = None
        else:
            print('Please enter a proper pathway quantify method, quitting.')
            func = None
            quit()

        # this is a recursive function for checking if the pathways have proteins
        def check_proteins(paths):
            L = []  # list of pathways with >1 protein
            n = 0
            for path in paths:
                if len(path.proteins) > 0:
                    L.append(path)
                    n += 1
            if n > 0:
                return L
            else:
                print('The associated pathways for this indication ({}) do not have enough proteins, '
                      'using all pathways'.format(indication.id_))
                return check_proteins(self.pathways)

        if indication:
            if len(indication.pathways) == 0:
                print('Warning: {} does not have any associated pathways - using all pathways'.format(indication.name))
                pws = self.pathways
            else:
                pws = check_proteins(indication.pathways)
        else:
            pws = check_proteins(self.pathways)

        for ci in range(len(self.compounds)):
            pw_sig_all = []
            c = self.compounds[ci]
            for pw in pws:
                if len(pw.proteins) == 0:
                    print('No associated proteins for pathway {}, skipping'.format(pw.id_))
                    continue
                pw_sig = []
                for p in pw.proteins:
                    ch = p.id_
                    ch_i = self.protein_id_to_index[ch]
                    pw_sig.append(c.sig[ch_i])

                if pq == 'proteins':
                    pw_sig_all += pw_sig
                else:
                    pw_sig_all.append(pw_sig)
            if pq != 'proteins':
                c.aux_sig = list(map(func, pw_sig_all))
            else:
                c.aux_sig = pw_sig_all


    def results_analysed(self, f, metrics, adrs=False):
        fo = open(f, 'w')
        effects = list(self.accuracies.keys())
#       effects_sorted = sorted(effects, key=lambda x:len(x[0].compounds))[::-1]
        if not adrs:
            effects_sorted = sorted(effects, key=lambda x: (len(x[0].compounds), x[0].id_))[::-1]
        else:
            effects_sorted = sorted(effects, key=lambda x: (len(x[0].compounds), x[0].id_))[::-1]
        l = len(effects)
        final_accs = {}
        for m in metrics:
            final_accs[m] = 0.0
        for effect, c in effects_sorted:
            fo.write("{0}\t{1}\t".format(effect.id_, c))
            accs = self.accuracies[(effect,c)]
            for m in metrics:
                n = accs[m]
                y = str(n / c * 100)[0:4]
                fo.write("{}\t".format(y))

                final_accs[m] += n / c / l
            fo.write("|\t{}\n".format(effect.name))
        fo.close()
        return final_accs

    def benchmark_classic(self, file_name='', SUM='', v1=False, indications=[],
                          bottom=False, ranking='geetika', adrs=False):
        if not os.path.exists('./results_analysed_named'):
            print("Directory 'results_analysed_named' does not exist, creating directory")
            os.system('mkdir results_analysed_named')
        if not os.path.exists('./raw_results'):
            print("Directory 'raw_results' does not exist, creating directory")
            os.system('mkdir raw_results')

        ra_named = 'results_analysed_named/results_analysed_named_' + file_name
        ra = 'raw_results/raw_results_' + file_name
        ra_out = open(ra, 'w')
        if adrs:
            ra_out.write("Compound, ADR, Top10, Top25, Top50, Top100, "
                     "TopAll, Top1%, Top5%, Top10%, Top50%, Top100%, Rank\n")
        else:
            ra_out.write("Compound, Disease, Top10, Top25, Top50, Top100, "
                         "TopAll, Top1%, Top5%, Top10%, Top50%, Top100%, Rank\n")

        x = (len(self.compounds)) / 100.0  # changed this...no reason to use similar instead of compounds
        # had to change from 100.0 to 100.0001 because the int function
        # would chop off an additional value of 1 for some reason...
        metrics = [(1,10), (2,25), (3,50), (4,100), (5,int(x*100.0001)),
                   (6,int(x*1.0001)), (7,int(x*5.0001)), (8,int(x*10.0001)),
                   (9,int(x*50.0001)), (10,int(x*100.0001))]

        if bottom:
            def rank_compound(sims, r):
                rank = 0
                for sim in sims:
                    if sim[1] >= r:
                        rank += 1.0
                    else:
                        return rank
                return len(sims)
        else:
            # Geetika's code
            def rank_compound(sims, r):
                rank = 0
                for sim in sims:
                    if sim[1] <= r:
                        rank += 1.0
                    else:
                        return rank
                return len(sims)
            # Geetika's reverse
            def rank_compound_reverse(sims, r):
                rank = 0
                for sim in sims:
                    if sim[1] < r:
                        rank += 1.0
                    else:
                        return rank
                return len(sims)

        effect_dct = {}
        ss = []
        c_per_effect = 0


        if indications:
            effects = list(map(self.get_indication, indications))
        elif adrs:
            effects = self.adrs
        else:
            effects = self.indications

        # Open for write the PDBs per ind file 
        if self.indication_genes:
            o = open("{}-ind2genes.tsv".format(file_name),'w')

        for effect in effects:
            count = len(effect.compounds)
            if count < 2:
                continue
            if not adrs:
                if self.indication_pathways:
                    if len(effect.pathways) == 0:
                        print('No associated pathways for {}, skipping'.format(effect.id_))
                        continue
                    elif len(effect.pathways) < 10:
                        print('Less than 10 associated pathways for {}, skipping'.format(effect.id_))
                        continue
            c_per_effect += count
            effect_dct[(effect, count)] = {}
            for m in metrics:
                effect_dct[(effect, count)][m] = 0.0
            # retrieve the appropriate proteins/pathway indices here, should be
            # incorporated as part of the ind object during file reading
            vs = []
            if self.pathways:
                if self.indication_pathways:
                    if self.pathway_quantifier == 'proteins':
                        for pw in effect.pathways:
                            for p in pw.proteins:
                                if p not in vs:
                                    vs.append(p)
                    else:
                        self.quantify_pathways(indication=effect)

            # Indication2genes
            # retrieve the appropriate protein indices here, should be
            # incorporated as part of the ind object during file reading
            if self.indication_genes:
                dg = []
                for p in effect.proteins:
                    if p not in dg:
                        dg.append(p)
                # write the PDBs per ind
                if len(dg) < 2:
                    o.write("{}\t{}\t{}\n".format(effect.id_,len(self.proteins),[prot.id_ for prot in self.proteins]))
                else:
                    o.write("{}\t{}\t{}\n".format(effect.id_,len(dg),[prot.id_ for prot in dg]))

            c = effect.compounds
            if self.pathways:
                if self.indication_pathways:
                    if self.pathway_quantifier == 'proteins':
                        if not vs:
                            print('Warning: protein list empty for {}, using all proteins'.format(effect.id_))
                            self.generate_some_similar_sigs(c, sort=True, proteins=None, aux=True)
                        else:
                            self.generate_some_similar_sigs(c, sort=True, proteins=vs, aux=True)
                    else:
                        self.generate_some_similar_sigs(c, sort=True, aux=True)
            elif self.indication_genes:
                if len(dg) < 2:
                    self.generate_some_similar_sigs(c, sort=True, proteins=None)
                else:
                    self.generate_some_similar_sigs(c, sort=True, proteins=dg)
            # call c.generate_similar_sigs()
            # use the proteins/pathways specified above


            for c in effect.compounds:
                for cs in c.similar:
                    if adrs:
                        if effect in cs[0].adrs:
                            cs_rmsd = cs[1]
                        else:
                            continue
                    else:
                        if effect in cs[0].indications:
                            cs_rmsd = cs[1]
                        else:
                            continue
                        # Test different ranking methods
                    if ranking=='geetika':
                        # Geetika's code
                        rank = rank_compound(c.similar, cs_rmsd)
                    elif ranking=='reverse':
                        # Geetika's reverse
                        rank = rank_compound_reverse(c.similar, cs_rmsd)
                    elif ranking=='sort':
                        # df sort_values
                        rank = c.similar.index(cs)
                    if adrs:
                        s = [str(c.index), effect.name]
                    else:
                        s = [str(c.index), effect.id_]
                    for x in metrics:
                        if rank <= x[1]:
                            effect_dct[(effect, count)][x] += 1.0
                            s.append('1')
                        else:
                            s.append('0')
                    s.append(str(int(rank)))
                    ss.append(s)
                    break
        
        if self.indication_genes:
            # Close PDB list per ind file
            o.close()

        self.accuracies = effect_dct
        if adrs:
            final_accs = self.results_analysed(ra_named, metrics, adrs=True)
        else:
            final_accs = self.results_analysed(ra_named, metrics, adrs=False)
        ss = sorted(ss, key=lambda xx: int(xx[0]))
        top_pairwise = [0.0] * 10
        for s in ss:
            if s[2] == '1':
                top_pairwise[0] += 1.0
            if s[3] == '1':
                top_pairwise[1] += 1.0
            if s[4] == '1':
                top_pairwise[2] += 1.0
            if s[5] == '1':
                top_pairwise[3] += 1.0
            if s[6] == '1':
                top_pairwise[4] += 1.0
            if s[7] == '1':
                top_pairwise[5] += 1.0
            if s[8] == '1':
                top_pairwise[6] += 1.0
            if s[9] == '1':
                top_pairwise[7] += 1.0
            if s[10] == '1':
                top_pairwise[8] += 1.0
            if s[11] == '1':
                top_pairwise[9] += 1.0
            sj = ','.join(s)
            sj += '\n'
            ra_out.write(sj)
        ra_out.close()

        cov = [0] * 10
        cov_count = [0.0] * 10
        for effect,c in list(self.accuracies.keys()):
            accs = self.accuracies[effect,c]
            for m_i in range(len(metrics)):
                v = accs[metrics[m_i]]
                if v > 0.0:
                    cov[m_i] += 1
                    cov_count[m_i] += v / c

        def float_to_str0(x):
            o = str(x * 100.0 / len(ss))[0:6]
            o_s = o.split('.')
            if len(o_s[-1]) == 1:
                return o + '00'
            if len(o_s[-1]) == 2:
                return o + '0'
            else:
                return o

        def float_to_str1(x):
            if len(ss) == 0:
                o = '0.00'
            else:
                o = str(x * 100.0 / len(ss))[0:6]
            o_s = o.split('.')
            if len(o_s[-1]) == 1:
                return o + '00'
            if len(o_s[-1]) == 2:
                return o + '0'
            if len(o_s[-1]) == 4:
                return o[:-1]
            else:
                return o

        def float_to_str2(x):
            o = str(x * 100.0)[0:6]
            o_s = o.split('.')
            if len(o_s[-1]) == 1:
                return o + '00'
            if len(o_s[-1]) == 2:
                return o + '0'
            if len(o_s[-1]) == 4:
                return o[:-1]
            else:
                return o

        for c_i in range(len(cov_count)):
            if cov[c_i] == 0:
                cov_count[c_i] = 0.0
            else:
                cov_count[c_i] /= cov[c_i]

        cov_count = list(map(float_to_str2, cov_count))

        if SUM:
            if v1:
                summary = open(SUM, 'w')
                summary.write('{0}\t{1}\t'.format(len(self.accuracies), c_per_effect/len(self.accuracies)))
                for m in metrics:
                    l = final_accs[m]
                    summary.write('{0}\t'.format(float_to_str2(l)))
                summary.write('-\t{}\n'.format(self.data_name))
                summary.write('.\t{0}\t{1}\t-\t{2}\n'.format(len(ss), '\t'.join(list(map(float_to_str1,top_pairwise))),
                                                             self.data_name))
                summary.write('{0} / {1} - {2} / {3} - {4} / {5} - {6} / {7} - {8} / {9} - {10}'
                              .format(cov[0], cov_count[0], cov[1], cov_count[1], cov[2], cov_count[2],
                                      cov[3], cov_count[3], cov[5], cov_count[5], self.data_name))
                summary.close()
            else:
                headers = ['top10','top25','top50','top100','top{}'.format(len(self.compounds)),
                           'top1%','top5%','top10%','top50%','top100%']
                # Create empty df with cutoff headers
                df = pd.DataFrame(columns=headers)
                # Create average indication accuracy list
                ia = []
                for m in metrics:
                    ia.append(float_to_str2(final_accs[m]))
                # Create average pairwise accuracy list
                pa = list(map(float_to_str1, top_pairwise))
                # Indication coverage is already done
                # Append 3 lists to df and write to file
                df = df.append(pd.Series(ia, index=df.columns), ignore_index=True)
                df = df.append(pd.Series(pa, index=df.columns), ignore_index=True)
                df = df.append(pd.Series(cov, index=df.columns), ignore_index=True)
                df.rename(index={0:'aia',1:'apa',2:'ic'}, inplace=True)
                df.to_csv(SUM, sep="\t")
       
        # pretty print the average indication accuracies
        cut = 0
        print("\taia")
        for m in metrics:
            print("{}\t{:.2f}".format(headers[cut], final_accs[m] * 100.0))
            cut+=1
        print('\n')
        #return final_accs

    # benchmark only the compounds in the indication mapping, aka get rid of "noisy" compounds
    # this function returns the filtered CANDO object in the event that you want to explore further
    def benchmark_associated(self, file_name='', SUM='', indications=[], continuous=False, ranking='geetika'):
        print("Making CANDO copy with only benchmarking-associated compounds")
        cp = copy.copy(self)
        good_cs = []
        good_ids = []
        for ind in cp.indications:
            if len(ind.compounds) >= 2:
                for c in ind.compounds:
                    if c.id_ not in good_ids:
                        good_cs.append(c)
                        good_ids.append(c.id_)
        cp.compounds = good_cs
        for c in cp.compounds:
            good_sims = []
            for s in c.similar:
                if s[0].id_ not in good_ids:
                    pass
                else:
                    good_sims.append(s)
            c.similar = good_sims
            sorted_scores = sorted(c.similar, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
            c.similar = sorted_scores
            c.similar_computed = True
        if continuous:
            cp.benchmark_continuous(file_name=file_name, SUM=SUM, indications=indications)
        else:
            cp.benchmark_classic(file_name=file_name, SUM=SUM, indications=indications, ranking=ranking)
        #return cp

    def benchmark_bottom(self, file_name='', SUM='', indications=[]):
        print("Making CANDO copy with reversed compound ordering")
        cp = CANDO(self.c_map, self.i_map, read_rmsds=self.read_rmsds)
        for ic in range(len(cp.compounds)):
            cp.compounds[ic].similar[0] = (cp.compounds[ic].similar[0][0], 100000.0)
            sorted_scores = sorted(cp.compounds[ic].similar, key=lambda x: x[1])[::-1]
            sorted_scores = sorted(cp.compounds[ic].similar, key=lambda x: x[1] if not math.isnan(x[1]) else -100000, reverse=True)
            cp.compounds[ic].similar = sorted_scores
            cp.compounds[ic].similar_computed = True
        cp.benchmark_classic(file_name, SUM, bottom=True)
        return cp

    def benchmark_indication(self, n, ind_id):
        ind = self.get_indication(ind_id)
        if len(ind.compounds) < 2:
            print("Sorry, there are not at least two compounds for {}".format(ind.name))
            return
        ind_acc = 0.0
        ind_count = 0
        for c in ind.compounds:
            if not self.compute_distance:
                self.generate_similar_sigs(c)
            for cs in c.similar[1:n+1]:
                if ind in cs[0].indications:
                    print(ind.name, cs[0].name, c.name)
                    ind_acc += 1.0
                    break
            ind_count += 1
        return ind_acc / ind_count

    def benchmark_continuous(self, file_name='', SUM='', v1=False, indications=[],
                              bottom=False, ranking='geetika'):
        if not os.path.exists('./results_analysed_named'):
            print("Directory 'results_analysed_named' does not exist, creating directory")
            os.system('mkdir results_analysed_named')
        if not os.path.exists('./raw_results'):
            print("Directory 'raw_results' does not exist, creating directory")
            os.system('mkdir raw_results')

        ra_named = 'results_analysed_named/results_analysed_named_cont_' + file_name
        ra = 'raw_results/raw_results_cont_' + file_name
        ra_out = open(ra, 'w')

        x = (len(self.compounds)) / 100.0  # changed this...no reason to use similar instead of compounds
        # had to change from 100.0 to 100.0001 because the int function
        # would chop off an additional value of 1 for some reason...

        def cont_metrics():
            all_v = []
            for c in self.compounds:
                for s in c.similar:
                    if s[1] != 0.0:
                        all_v.append(s[1])
            avl = len(all_v)
            all_v_sort = sorted(all_v)
            # for tuple 9, have to add the '-1' for index out of range reasons
            metrics = [(1, all_v_sort[int(avl/1000.0)]), (2, all_v_sort[int(avl/200.0)]), (3, all_v_sort[int(avl/100.0)]),
                       (4, all_v_sort[int(avl/20.0)]), (5, all_v_sort[int(avl/10.0)]), (6, all_v_sort[int(avl/5.0)]),
                       (7, all_v_sort[int(avl/3.0)]), (8, all_v_sort[int(avl/2.0)]), (9, all_v_sort[int(avl/1.0)-1])]
            return metrics

        metrics = cont_metrics()
        ra_out.write("Compound, Disease, 0.1%({}), 0.5%({}), 1%({}), 5%({}), "
                     "10%({}), 20%({}), 33%({}), 50%({}), "
                     "100%({}), Value\n".format(metrics[0][1], metrics[1][1], metrics[2][1], metrics[3][1],
                                                                metrics[4][1], metrics[5][1], metrics[6][1], metrics[7][1],
                                                                metrics[8][1]))

        if bottom:
            def rank_compound(sims, r):
                rank = 0
                for sim in sims:
                    if sim[1] >= r:
                        rank += 1.0
                    else:
                        return rank
                return len(sims)
        else:
            # Geetika's code
            def rank_compound(sims, r):
                rank = 0
                for sim in sims:
                    if sim[1] <= r:
                        rank += 1.0
                    else:
                        return rank
                return len(sims)

            # Geetika's reverse
            def rank_compound_reverse(sims, r):
                rank = 0
                for sim in sims:
                    if sim[1] < r:
                        rank += 1.0
                    else:
                        return rank
                return len(sims)

        ind_dct = {}
        ss = []
        c_per_ind = 0

        if indications:
            inds = list(map(self.get_indication, indications))
        else:
            inds = self.indications

        for ind in inds:
            count = len(ind.compounds)
            if count < 2:
                continue
            if self.indication_pathways:
                if len(ind.pathways) == 0:
                    print('No associated pathways for {}, skipping'.format(ind.id_))
                    continue
                elif len(ind.pathways) < 10:
                    print('Less than 10 associated pathways for {}, skipping'.format(ind.id_))
                    continue
            c_per_ind += count
            ind_dct[(ind, count)] = {}
            for m in metrics:
                ind_dct[(ind, count)][m] = 0.0
            # retrieve the appropriate proteins/pathway indices here, should be
            # incorporated as part of the ind object during file reading
            vs = []
            if self.pathways:
                if self.indication_pathways:
                    if self.pathway_quantifier == 'proteins':
                        for pw in ind.pathways:
                            for p in pw.proteins:
                                if p not in vs:
                                    vs.append(p)
                    else:
                        self.quantify_pathways(indication=ind)

            for c in ind.compounds:
                if self.pathways:
                    if self.indication_pathways:
                        if self.pathway_quantifier == 'proteins':
                            if not vs:
                                print('Warning: protein list empty for {}, using all proteins'.format(c.name,
                                                                                                      ind.id_))
                                self.generate_similar_sigs(c, sort=True, proteins=None, aux=True)
                            else:
                                self.generate_similar_sigs(c, sort=True, proteins=vs, aux=True)
                        else:
                            self.generate_similar_sigs(c, sort=True, aux=True)
                # call c.generate_similar_sigs()
                # use the proteins/pathways specified above
                for cs in c.similar:
                    if ind in cs[0].indications:
                        cs_rmsd = cs[1]
                        s = [str(c.index), ind.id_[5:]]
                        for x in metrics:
                            if cs_rmsd <= x[1]:
                                ind_dct[(ind, count)][x] += 1.0
                                s.append('1')
                            else:
                                s.append('0')
                        s.append(str(cs_rmsd))
                        ss.append(s)
                        break
        self.accuracies = ind_dct
        final_accs = self.results_analysed(ra_named, metrics)
        ss = sorted(ss, key=lambda xx: int(xx[0]))
        top_pairwise = [0.0] * 9
        for s in ss:
            if s[2] == '1':
                top_pairwise[0] += 1.0
            if s[3] == '1':
                top_pairwise[1] += 1.0
            if s[4] == '1':
                top_pairwise[2] += 1.0
            if s[5] == '1':
                top_pairwise[3] += 1.0
            if s[6] == '1':
                top_pairwise[4] += 1.0
            if s[7] == '1':
                top_pairwise[5] += 1.0
            if s[8] == '1':
                top_pairwise[6] += 1.0
            if s[9] == '1':
                top_pairwise[7] += 1.0
            if s[10] == '1':
                top_pairwise[8] += 1.0
            sj = ','.join(s)
            sj += '\n'
            ra_out.write(sj)
        ra_out.close()

        cov = [0] * 9
        cov_count = [0.0] * 9
        for ind, c in list(self.accuracies.keys()):
            accs = self.accuracies[ind, c]
            for m_i in range(len(metrics)):
                v = accs[metrics[m_i]]
                if v > 0.0:
                    cov[m_i] += 1
                    cov_count[m_i] += v / c

        def float_to_str0(x):
            o = str(x * 100.0 / len(ss))[0:6]
            o_s = o.split('.')
            if len(o_s[-1]) == 1:
                return o + '00'
            if len(o_s[-1]) == 2:
                return o + '0'
            else:
                return o

        def float_to_str1(x):
            if len(ss) == 0:
                o = '0.00'
            else:
                o = str(x * 100.0 / len(ss))[0:6]
            o_s = o.split('.')
            if len(o_s[-1]) == 1:
                return o + '00'
            if len(o_s[-1]) == 2:
                return o + '0'
            if len(o_s[-1]) == 4:
                return o[:-1]
            else:
                return o

        def float_to_str2(x):
            o = str(x * 100.0)[0:6]
            o_s = o.split('.')
            if len(o_s[-1]) == 1:
                return o + '00'
            if len(o_s[-1]) == 2:
                return o + '0'
            if len(o_s[-1]) == 4:
                return o[:-1]
            else:
                return o

        for c_i in range(len(cov_count)):
            if cov[c_i] == 0:
                cov_count[c_i] = 0.0
            else:
                cov_count[c_i] /= cov[c_i]

        cov_count = list(map(float_to_str2, cov_count))

        if SUM:
            if v1:
                summary = open(SUM, 'w')
                summary.write('{0}\t{1}\t'.format(len(self.accuracies), c_per_ind / len(self.accuracies)))
                for m in metrics:
                    l = final_accs[m]
                    summary.write('{0}\t'.format(float_to_str2(l)))
                summary.write('-\t{}\n'.format(self.data_name))
                summary.write(
                    '.\t{0}\t{1}\t-\t{2}\n'.format(len(ss), '\t'.join(list(map(float_to_str1, top_pairwise))),
                                                   self.data_name))
                summary.write('{0} / {1} - {2} / {3} - {4} / {5} - {6} / {7} - {8} / {9} - {10}'
                              .format(cov[0], cov_count[0], cov[1], cov_count[1], cov[2], cov_count[2],
                                      cov[3], cov_count[3], cov[5], cov_count[5], self.data_name))
                summary.close()
            else:
                headers = ['0.1%', '0.5%', '1%', '5%', '10%',
                           '20%', '33%', '50%', '100%']
                # Create empty df with cutoff headers
                df = pd.DataFrame(columns=headers)
                # Create average indication accuracy list
                ia = []
                for m in metrics:
                    ia.append(float_to_str2(final_accs[m]))
                # Create average pairwise accuracy list
                pa = list(map(float_to_str1, top_pairwise))
                # Indication coverage is already done
                # Append 3 lists to df and write to file
                df = df.append(pd.Series(ia, index=df.columns), ignore_index=True)
                df = df.append(pd.Series(pa, index=df.columns), ignore_index=True)
                df = df.append(pd.Series(cov, index=df.columns), ignore_index=True)
                df.rename(index={0:'aia',1:'apa',2:'ic'}, inplace=True)
                df.to_csv(SUM, sep="\t")

        # pretty print the average indication accuracies
        cut = 0
        print("\taia")
        for m in metrics:
            print("{}\t{:.2f}".format(headers[cut], final_accs[m] * 100.0))
            cut+=1
        print('\n')
        #return final_accs


    def benchmark_cluster(self, n_clusters):
        def cluster_kmeans(cmpds):
            def f(x):
                return x.sig
            def g(x):
                return x.indications
            def h(x):
                return x.id_

            sigs = np.array(list(map(f, cmpds)))
            pca = PCA(n_components=10).fit(sigs)
            sigs = pca.transform(sigs)
            inds = np.array(list(map(g, cmpds)))
            ids = np.array(list(map(h, cmpds)))
            sigs_train, sigs_test, inds_train, inds_test, ids_train, ids_test = train_test_split(sigs, inds, ids, test_size=0.20, random_state=1)
            clusters = KMeans(n_clusters, random_state=1).fit(sigs_train)
            return clusters,sigs_test,inds_train,inds_test,ids_train,ids_test

        def pred_kmeans(d_test):

            for ind in self.indications:
                if len(ind.compounds) < 2:
                    continue
                for c in ind.compounds:
                    if c.id_ != c2.id_ and c.cluster_id == c2.cluster_id:
                            total_acc +=1.0
                            break
                    total_count += 1


        # Calculate the K means clusters for all compound signatures
        cs,sigs_test,inds_train,inds_test,ids_train,ids_test = cluster_kmeans(self.compounds)
        labels = cs.labels_

        # Determine how many compounds are in each cluster
        # Plot the results and output the mean, median, and range
        all_clusters = range(n_clusters)
        c_clusters = [0] * n_clusters
        for l in labels:
            c_clusters[l] += 1
        '''
        plt.scatter(all_clusters,c_clusters)
        plt.text(1, 1, "Average cluster size = {}".format(np.mean(c_clusters)), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        plt.text(1, 1, "Median cluster size = {}".format(np.median(c_clusters)), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        plt.text(1, 1, "Range of cluster sizes = {}".format(np.min(c_clusters), np.max(c_clusters)), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        plt.savefig("cluster_size.png")
        '''
        # Map the labels for each compound to the cluster_id for each compound object
        for ci in range(len(labels)):
            self.compounds[ids_train[ci]].cluster_id = labels[ci]

        total_acc = 0.0
        total_count = 0

        # Calculate the benchmark accuracy by
        # mimicking classic benchmark -- leave one out
        # and recapture at least one for each indication-drug pair
        for i in range(len(sigs_test)):
            lab = cs.predict(sigs_test[i].reshape(1,-1))
            for ind in inds_test[i]:
                for c in range(len(inds_train)):
                    done = False
                    for ind_train in inds_train[c]:
                        if ind.name == ind_train.name and lab[0] == labels[c]:
                            total_acc+=1.0
                            done = True
                            break
                    if done == True:
                        break
                total_count += 1

        print("Number of cluster = {}".format(n_clusters))
        print("Mean cluster size = {}".format(np.mean(c_clusters)))
        print("Median cluster size = {}".format(np.median(c_clusters)))
        print("Range of cluster sizes = [{},{}]".format(np.min(c_clusters),np.max(c_clusters)))
        print("% Accuracy = {}".format(total_acc / total_count * 100.0))


    # create a random forest classifier for a specified indication or all inds (to benchmark)
    # predict (used w/ 'effect=' - indication or ADR) is a list of compounds to classify with the trained RF model
    # out=X saves benchmark SUMMARY->SUMMARY_rf_X; raw results->raw_results/raw_results_rf_X (same for RAN)
    # currently supports random forest ('rf'), 1-class SVM ('1csvm'), and logistic regression ('log')
    # models are trained with leave-one-out cross validation
    def ml(self, method='rf', effect=None, benchmark=False, adrs=False, predict=[], seed=42, out=''):

        if out:
            if not os.path.exists('./raw_results/'):
                os.system('mkdir raw_results')
            if not os.path.exists('./results_analysed_named/'):
                os.system('mkdir results_analysed_named')

        # gather approved compound signatures for training
        def split_cs(efct, cmpd=None):
            mtrx = []
            for cm in efct.compounds:
                if cmpd:
                    if cm.id_ == cmpd.id_:
                        pass
                    else:
                        mtrx.append(cm.sig)
                else:
                    mtrx.append(cm.sig)
            return mtrx, [1] * len(mtrx)

        # choose random 'neutral' compounds for training, seed for reproducibility
        def random_neutrals(efct, s=None, benchmark=False):
            neutrals = []
            used = []
            shuffled = list(range(len(self.compounds)))
            if s:
                random.seed(s)
                random.shuffle(shuffled)
                #print(shuffled)
            else:
                s = random.randint(0, len(self.compounds)-1)
                random.seed(s)
                random.shuffle(shuffled)
            i = 0
            if benchmark:
                n_neu = len(efct.compounds) - 1
            else:
                n_neu = len(efct.compounds)
            while len(neutrals) < n_neu:
                n = shuffled[i]
                if n in used:
                    i += 1
                    continue
                neu = self.compounds[n]
                if neu not in efct.compounds:
                    neutrals.append(neu.sig)
                    used.append(n)
                    i += 1
                else:
                    i += 1
            return neutrals, [0] * len(neutrals)

        def model(meth, samples, labels, params=None, seed=None):
            if meth == 'rf':
                m = RandomForestClassifier(n_estimators=100, random_state=seed)
                m.fit(samples, labels)
                return m
            elif meth == 'svm':
                m = svm.SVC(kernel='rbf', gamma='scale', degree=3, random_state=seed)
                m.fit(samples, labels)
                return m
            elif meth == '1csvm':
                keep = []
                for i in range(len(samples)):
                    if labels[i] == 1:
                        keep.append(samples[i])
                m = svm.OneClassSVM(kernel='poly', gamma='scale', degree=2)
                m.fit(keep)
                return m
            elif meth == 'log':
                m = LogisticRegression(penalty='l2', solver='newton-cg', random_state=seed)
                m.fit(samples, labels)
                return m
            else:
                print("Please enter valid machine learning method ('rf', '1csvm', 'log')")
                quit()

        if benchmark:
            if adrs:
                effects = sorted(self.adrs, key=lambda x: (len(x.compounds), x.id_))[::-1]
            else:
                effects = sorted(self.indications, key=lambda x: (len(x.compounds), x.id_))[::-1]
            if out:
                frr = open('./raw_results/raw_results_ml_{}'.format(out), 'w')
                frr.write('Compound, Effect, Class\n')
                fran = open('./results_analysed_named/results_analysed_named_ml_{}'.format(out), 'w')
                fsum = open('SUMMARY_ml_{}'.format(out), 'w')
        else:
            if len(effect.compounds) < 1:
                print('No compounds associated with this indication, quitting.')
                quit()
            effects = [effect]

        rf_scores = []
        for e in effects:
            if len(e.compounds) < 2:
                continue
            neu = random_neutrals(e, s=seed, benchmark=True)
            tp_fn = [0, 0]
            for c in e.compounds:
                pos = split_cs(e, cmpd=c)
                train_samples = np.array(pos[0] + neu[0])
                train_labels = np.array(pos[1] + neu[1])
                mdl = model(method, train_samples, train_labels, seed=seed)
                pred = mdl.predict(np.array([c.sig]))
                if pred[0] == 1:
                    tp_fn[0] += 1
                else:
                    tp_fn[1] += 1
                if benchmark and out:
                    frr.write('{}, {}, {}\n'.format(c.id_, e.id_, pred[0]))

            # predict whether query drugs are associated with this indication
            if predict:
                print('Indication: {}'.format(e.name))
                print('Leave-one-out cross validation: TP={}, FN={}, Acc={:0.2f}'.format(
                    tp_fn[0], tp_fn[1], 100 * (tp_fn[0] / float(len(e.compounds)))))
                neu = random_neutrals(e, s=seed, benchmark=False)
                pos = split_cs(e)
                train_samples = np.array(pos[0] + neu[0])
                train_labels = np.array(pos[1] + neu[1])
                mdl = model(method, train_samples, train_labels, seed=seed)
                print('\tCompound\tClass')
                for c in predict:
                    pred = mdl.predict(np.array([c.sig]))
                    print('\t{}\t{}'.format(c.name, pred[0]))

            # append loocv results to combined list
            rf_scores.append((e, tp_fn))


        sm = [0, 0, 0, 0]
        if benchmark:
            for rf_score in rf_scores:
                efct = rf_score[0]
                tfp = rf_score[1]
                acc = tfp[0] / float(len(efct.compounds))
                sm[0] += len(efct.compounds)
                sm[1] += acc
                sm[2] += (acc * len(efct.compounds))
                if acc > 0.5:
                    sm[3] += 1
                if out:
                    fran.write('{}\t{}\t{}\t{}\t{:0.2f}\t|\t{}\n'.format(efct.id_, len(efct.compounds),
                                                                      tfp[0], tfp[1], 100 * acc, efct.name))
            if out:
                fsum.write('aia\t{:0.2f}\n'.format(100 * (sm[1]/len(rf_scores))))
                fsum.write('apa\t{:0.2f}\n'.format(100 * (sm[2] / sm[0])))
                fsum.write('ic\t{}\n'.format(sm[3]))

            print('aia\t{:0.2f}'.format(100 * (sm[1]/len(rf_scores))))
            print('apa\t{:0.2f}'.format(100 * (sm[2] / sm[0])))
            print('ic\t{}'.format(sm[3]))

        return


    # this function is an extension of canpredict - basically, give it a
    # ind_id id and for each of the associated compounds, it will generate
    # the similar compounds (based on rmsd) and add them to a dictionary
    # with a value of how many times it shows up (enrichment). If a
    # compound not approved for the indication of interest keeps showing
    # up, that means it is similar in signature to the drugs that are
    # ALREADY approved for the indication, so it may be a target for repurposing.
    # Control how many similar compounds to consider with the argument 'n'.
    # Use ind_id=None to find greatest score sum across all proteins (sum_scores must be True)
    def canpredict_compounds(self, ind_id, n=10, topX=10, sum_scores=False, keep_approved=False):
        if ind_id:
            i = self.indication_ids.index(ind_id)
            ind = self.indications[i]
            print("{0} compounds found for {1} --> {2}".format(len(ind.compounds), ind.id_, ind.name))
        else:
            print("Finding compounds with greatest summed scores in {}...".format(self.matrix))

        if not sum_scores:
            if self.pathways:
                if self.indication_pathways:
                    self.quantify_pathways(ind)
                else:
                    self.quantify_pathways()
            for c in ind.compounds:
                if c.similar_computed:
                    continue
                if self.pathways:
                    self.generate_similar_sigs(c, aux=True, sort=True)
                elif self.indication_genes:
                    self.generate_similar_sigs(c, sort=True, proteins=ind.proteins)
                else:
                    self.generate_similar_sigs(c, sort=True)
        if not sum_scores:
            print("Generating compound predictions using top {} most similar compounds...\n".format(n))
            c_dct = {}
            for c in ind.compounds:
                for c2_i in range(n):
                    c2 = c.similar[c2_i]
                    if c2[1] == 0.0:
                        continue
                    already_approved = ind in c2[0].indications
                    k = c2[0].id_
                    if k not in c_dct:
                        c_dct[k] = [1, already_approved]
                    else:
                        c_dct[k][0] += 1
        else:
            c_dct = {}
            if self.indication_genes and ind_id:
                indices = []
                for p in ind.proteins:
                    indices.append(self.protein_id_to_index[p.id_])
            else:
                indices = range(len(self.proteins))
            for c in self.compounds:
                ss = 0.0
                for pi in indices:
                    ss += c.sig[pi]
                if ind_id:
                    already_approved = ind in c.indications
                else:
                    already_approved = False  # Not relevant since there is no indication
                c_dct[c.id_] = [ss, already_approved]

        sorted_x = sorted(c_dct.items(), key=lambda x:x[1][0])[::-1]
        print("Printing top {} compounds...\n".format(topX))
        if not keep_approved:
            i = 0
            print('rank\tscore\tid\tname')
            for p in enumerate(sorted_x):
                if i >= topX and topX != -1:
                    break
                if p[1][1][1]:
                    continue
                else:
                    print("{}\t{}\t{}\t{}".format(i + 1, p[1][1][0],
                                                  self.get_compound(p[1][0]).id_,self.get_compound(p[1][0]).name))
                    i+=1
        else:
            i = 0
            print('rank\tscore\tapproved\tid\tname')
            for p in enumerate(sorted_x):
                if i >= topX and topX != -1:
                    break
                print("{}\t{}\t{}\t\t{}\t{}".format(i + 1, p[1][1][0], p[1][1][1],
                                                    self.get_compound(p[1][0]).id_,self.get_compound(p[1][0]).name))
                i+=1
        print('\n')
        #return sorted_x


    def canpredict_indications(self, new_sig='', new_name='', cando_cmpd='', n='10', topX=10):
        if new_sig != '':
            cmpd = self.add_cmpd(new_sig, new_name)
        elif cando_cmpd != '':
            cmpd = cando_cmpd
            print("Using CANDO compound {}".format(cmpd.name))
            print("Compound has id {} and index {}".format(cmpd.id_,cmpd.index))
        print("Comparing signature to all CANDO compound signatures...")
        self.generate_similar_sigs(cmpd, sort=True)
        print("Generating indication predictions using top{} most similar compounds...".format(n))
        i_dct = {}
        for c in cmpd.similar[1:n+1]:
            for ind in c[0].indications:
                if ind not in i_dct:
                    i_dct[ind] = 1
                else:
                    i_dct[ind] += 1
        sorted_x = sorted(i_dct.items(), key=operator.itemgetter(1),reverse=True)
        print("Printing top {} indications...\n".format(topX))
        print("rank\tscore\tind_id    \tindication")
        #print("rank\tscore\tindication\tind_id")
        for i in range(topX):
            print("{}\t{}\t{}\t{}".format(i+1,sorted_x[i][1],sorted_x[i][0].id_,sorted_x[i][0].name))
            #print("{}\t{}\t{}\t{}".format(i+1,sorted_x[i][1],sorted_x[i][0].name,sorted_x[i][0].id_))
        print('')
        #return sorted_x
        

    def similar_compounds(self, new_sig='', new_name='', cando_cmpd='', n='10'):
        if new_sig != '':
            cmpd = self.add_cmpd(new_sig, new_name)
        elif cando_cmpd != '':
            cmpd = cando_cmpd
            print("Using CANDO compound {}".format(cmpd.name))
            print("Compound has id {} and index {}".format(cmpd.id_,cmpd.index))
        print("Comparing signature to all CANDO compound signatures...")
        self.generate_similar_sigs(cmpd, sort=True)
        print("Generating {} most similar compound predictions...\n".format(n))
        print("rank\tdist\tid\tname")
        #for c in cmpd.similar[1:n+1]:
        for i in range(1,n+1):
            print("{}\t{:.3f}\t{}\t{}".format(i,cmpd.similar[i][1],cmpd.similar[i][0].id_,cmpd.similar[i][0].name))
        print('\n')
        #return sorted_x

    
    def add_cmpd(self, new_sig, new_name):
        new_sig = pd.read_csv(new_sig , sep='\t', index_col=0, header=None)
        if new_name == '':
            print("Compound needs a name. Set 'new_name'")
            return
        i = len(self.compounds)
        cmpd = Compound(new_name, i, i)
        cmpd.sig = new_sig[[1]].T.values[0].tolist()
        print("New compound is " + cmpd.name)
        print("New compound has id {} and index {}".format(cmpd.id_,cmpd.index))
        return cmpd


    # return a list of all signatures, rm is a list of compound ids you do not want in the list
    def sigs(self, rm):
        return [x.sig for x in self.proteins if x.id_ not in rm]

    def save_rmsds_to_file(self, f):
        def rmsds_to_str(cmpd):
            o = ''
            for s in cmpd.similar:
                o += '{}\t'.format(s[1])
            o = o + '\n'
            return o

        with open(f, 'w') as srf:
            for c in self.compounds:
                srf.write(rmsds_to_str(c))

    # this function re-ranks the compounds according to the desired comparison specified by
    # 'method' -> currently supports 'min', 'avg', 'mult', and 'sum'
    def fusion(self, cando_objs, out_file='', method='sum'):
        print("Fusing CANDO objects using " + method)
        cnd = CANDO(self.c_map, self.i_map)
        if self.rm_cmpds:
            cnd.compounds = self.compounds
            cnd.indications = self.indications
            for c in cnd.compounds:
                c.similar = []
                c.sig = []
        dn = [self.data_name]
        for obj in cando_objs:
            dn.append(obj.data_name)
        cnd.data_name = "-".join(dn) + '-' + method
        cid_to_ranks = {}
        for c in self.compounds:
            cid_to_ranks[c.id_] = {}
            sims = c.similar
            for i in range(len(sims)):
                cid_to_ranks[c.id_][sims[i][0].id_] = [i]
        for cando_obj in cando_objs:
            for c2 in cando_obj.compounds:
                sims2 = c2.similar
                for j in range(len(sims2)):
                    cid_to_ranks[c2.id_][sims2[j][0].id_].append(j)
        for c3 in cnd.compounds:
            ranks_dct = cid_to_ranks[c3.id_]
            for c4 in cnd.compounds:
                ranks = ranks_dct[c4.id_]
                if method == 'min':
                    c3.similar.append((c4, float(min(ranks))))
                if method == 'sum':
                    c3.similar.append((c4, float(sum(ranks))))
                if method == 'avg':
                    c3.similar.append((c4, (float(sum(ranks))) / len(ranks)))
                if method == 'mult':
                    m = 1.0
                    for r in ranks:
                        m *= r
                    c3.similar.append((c4, m))
        if out_file:
            with open(out_file, 'w') as fo:
                for co in cnd.compounds:
                    s = list(map(str, [x[1] for x in co.similar]))
                    fo.write('\t'.join(s) + '\n')
        for cf in cnd.compounds:
            sorted_scores = sorted(c.similar, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
            cf.similar = sorted_scores
            cf.similar_computed = True
        return cnd

    # Normalize the distance scores to between [0,1]. Simply divides all scores by the largest distance
    # between any two compounds.
    def normalize(self):
        if len(self.compounds[0].similar) == 0:
            print('Similar scores not computed yet -- quitting')
            return

        mx = 0
        for c in self.compounds:
            for s in c.similar:
                if s[1] > mx:
                    mx = s[1]

        print('Max value is {}'.format(mx))

        def norm(x):
            v = x[1] / mx
            return x[0], v

        for c in self.compounds:
            c.similar = list(map(norm, c.similar))
        return

    def __str__(self):
        N = len(self.compounds)
        b = self.compounds[0].similar_computed
        I = len(self.indications)
        P = len(self.proteins)
        if P:
            return 'CANDO: {0} compounds, {1} proteins, {2} indications\n' \
                   '\tMatrix - {3}\nIndication mapping - {4}\n' \
                   '\tDistances computed - {5}'.format(N, P, I, self.matrix, self.i_map, b)
        elif self.read_rmsds:
            return 'CANDO: {0} compounds, {1} indications\n' \
                   '\tCompound comparison file - {2}\n' \
                   '\tIndication mapping - {3}'.format(N, I, self.read_rmsds, self.i_map)
        else:
            return 'CANDO: {0} compounds, {1} indications\n' \
                   '\tIndication mapping - {2}'.format(N, I, self.i_map)


# intended for easier handling of matrices
# convert between fpt and tsv as well as distance to similarity (and vice versa)
class Matrix(object):
    def __init__(self, matrix_file, rmsd=False, convert_to_tsv=False):
        self.matrix_file = matrix_file
        self.proteins = []
        self.values = []

        def pro_name(l):
            name = l[0]
            curr = l[1]
            index = 1
            while curr != ' ':
                name += curr
                index += 1
                curr = l[index]
            return name

        if not rmsd:
            with open(matrix_file, 'r') as f:
                lines = f.readlines()
                if convert_to_tsv:
                    if matrix_file[-4:] == '.fpt':
                        out_file = '.'.join(matrix_file.split('.')[:-1]) + '.tsv'
                    else:
                        out_file = matrix_file + '.tsv'
                    of = open(out_file, 'w')
                    for l_i in range(len(lines)):
                        name = pro_name(lines[l_i])
                        scores = []
                        i = 24
                        while i < len(lines[l_i]):
                            score = lines[l_i][i:i + 5]
                            i += 8
                            scores.append(score)
                        self.proteins.append(name)
                        self.values.append(list(map(float, scores)))
                        of.write("{0}\t{1}\n".format(name, '\t'.join(scores)))
                    of.close()
                else:
                    for l_i in range(len(lines)):
                        vec = lines[l_i].strip().split('\t')
                        if len(vec) < 2:
                            print('The matrix file {} is in the old fpt format -- please ' \
                                  'convert to tsv with the following line of code:'.format(self.matrix_file))
                            print('-> Matrix("{}", convert_to_tsv=True) <-'.format(self.matrix_file))
                            quit()
                        name = vec[0]
                        scores = vec[1:]
                        self.proteins.append(name)
                        self.values.append(list(map(float, scores)))
        else:
            with open(matrix_file, 'r') as rrs:
                lines = rrs.readlines()
                for i in range(len(lines)):
                    scores = list(map(float, lines[i].strip().split('\t')))
                    self.values.append(scores)

    def fusion(self, matrix, metric='mult', out_file=''):
        if len(self.values[0]) != len(self.values):
            print('{} not symmetric, quitting.'.format(self.matrix_file))
            quit()
        if len(matrix.values[0]) != len(matrix.values):
            print('{} not symmetric, quitting.'.format(matrix.matrix_file))
            quit()
        if self.values[0][0] != matrix.values[0][0]:
            print('The first values of these matrices do not match; ' \
                  'please ensure they are using both distance or similarity -- quitting.')
            quit()
        out = []
        for i in range(len(self.values)):
            vs = []
            for j in range(len(self.values)):
                v1 = self.values[i][j]
                v2 = matrix.values[i][j]
                if metric == 'mult':
                    v = v1 * v2
                elif metric == 'sum':
                    v = v1 + v2
                else:
                    v = v1 * v2
                vs.append(v)
            out.append(vs)

        if out_file:
            of = open(out_file, 'w')
            for vs in out:
                of.write("{}\n".format('\t'.join(list(map(str, vs)))))
            of.close()

    def convert(self, out_file):
        if self.values[0][0] == 0.0:
            metric = 'd'
        elif self.values[0][0] == -1.0:
            metric = 'd'
            self.fix()
        elif self.values[0][0] == 1.0:
            metric = 's'
        elif self.values[0][0] == 2.0:
            metric = 's'
            self.fix()
        else:
            metric = None
            print('The first value is not 0.0 (-1.0) or 1.0 (2.0); ' \
                  'please ensure the matrix is generated properly')
            quit()

        def to_dist(s):
            return 1 - s

        def to_sim(d):
            return 1 / (1 + d)

        of = open(out_file, 'w')
        if metric == 'd':
            for vs in self.values:
                vs = list(map(to_sim, vs))
                of.write("{}\n".format('\t'.join(list(map(str, vs)))))
        else:
            if metric == 's':
                for vs in self.values:
                    vs = list(map(to_dist, vs))
                    of.write("{}\n".format('\t'.join(list(map(str, vs)))))
        of.close()

    def fix(self):
        if self.values[0][0] == -1.0:
            for i in range(len(self.values)):
                self.values[i][i] = 0.0
        if self.values[0][0] == 2.0:
            for i in range(len(self.values)):
                self.values[i][i] = 1.0


def generate_matrix(cmpd_scores='', prot_scores='', matrix_file='cando_interaction_matrix.tsv', ncpus=1):
    start = time.time()
    ob_cutoff = 0.0
    bs_cutoff = 0.0

    print("Compiling compound scores...")
    c_scores = pd.read_csv(cmpd_scores, sep='\t', index_col=0)

    print("Compiling binding site scores...")
    p_scores = pd.read_csv(prot_scores, sep='\t', index_col=0, header=None)

    print("Calculating interaction scores...")
    pool = mp.Pool(ncpus)
    scores_temp = pool.starmap_async(get_scores, [(int(c), p_scores, c_scores.loc[:,c]) for c in c_scores.columns]).get()
    pool.close()
    first = True
    scores = pd.DataFrame()
    print("Generating matrix...")
    for i in scores_temp:
        if first:
            scores = pd.DataFrame(i)
            first = False
        else:
            scores = scores.join(pd.DataFrame(i))
    scores.rename(index=dict(zip(range(len(p_scores.index)),p_scores.index)),inplace=True)
    scores.to_csv(matrix_file, sep='\t', header=None, float_format='%.3f')

    end = time.time()
    print("Matrix written to {}.".format(matrix_file))
    print("Matrix generation took {:.0f} seconds to finish.".format(end-start))


def generate_signature(cmpd_scores='', prot_scores='', matrix_file='', ncpus=1):
    start = time.time()
    ob_cutoff = 0.0
    bs_cutoff = 0.0
    if matrix_file == '':
        matrix_file = "{}_signature.tsv".format(cmpd_scores.split('/')[-1].split('.')[0].split('_')[0])

    print("Compiling compound scores...")
    c_scores = pd.read_csv(cmpd_scores, sep='\t', index_col=0)

    print("Compiling binding site scores...")
    p_scores = pd.read_csv(prot_scores, sep='\t', index_col=0, header=None)

    print("Generating interaction signature...")
    print(c_scores.columns[0])
    c = c_scores.columns[0]
    scores_temp = get_scores(c, p_scores, c_scores.loc[:,c])
    scores = pd.DataFrame(scores_temp)
    scores.rename(index=dict(zip(range(len(p_scores.index)),p_scores.index)),inplace=True)
    scores.to_csv(matrix_file, sep='\t', header=None, float_format='%.3f')

    end = time.time()
    print("Signature written to {}.".format(matrix_file))
    print("Signature generation took {:.0f} seconds to finish.".format(end-start))


def get_scores(c, p_scores, c_score):
    l = []
    for pdb in p_scores.index:
        temp_value = 0.000
        if pd.isnull(p_scores.loc[pdb][1]):
            l.append(temp_value)
            continue
        for bs in p_scores.loc[pdb][1].split(','):
            try:
                if temp_value < c_score[bs]:
                    temp_value = c_score[bs]
            except KeyError:
                continue
        l.append(temp_value)
    return {c:l}


def get_ob(drug,cmpd_sig,ob_cutoff):
    if not os.path.exists("{}/{}.tsv".format(cmpd_dir,drug)):
        print("No signature for ")
        return
    scores = {}
    scores[drug] = {}
    # Read in OB output
    ob = pd.read_csv("{}/{}.tsv".format(cmpd_dir,drug), header=None, sep='\t')
    for i in ob.index:
        if float(ob[1][i]) >= ob_cutoff:
            # Set the [drug index][pdb+ligand] dict to the OBscore
            scores[drug][ob[0][i]] = float(ob[1][i])
    return {drug:scores[drug]}

def get_bs(pdb,prot_dir,bs_cutoff):
    bs_path = "{}/{}.lst".format(prot_dir,pdb)
    # Check if PDB has any BS
    if not os.path.exists(bs_path) or open(bs_path).readline().startswith("No Binding sites"):
        print("No file {}".format(bs_path))
        return
    # Create BSscores dictionary
    scores = {}
    scores[pdb] = {}
    # Read in COFACTOR output
    '''
    # Bsites_model1.dat
    bs = pd.read_csv(bs_path, header=None, sep='\s+')
    for i in bs.index:
        if float(bs[9][i]) >= bs_cutoff:
            # Set the [pdb][pdb_ligand] dict to BSscore
            scores[pdb][bs[2][i]+"_"+bs[8][i]] = float(bs[9][i])
    return {pdb:scores[pdb]}
    '''
    # selected_templates.lst
    bs = pd.read_csv(bs_path, header=None, sep='\s+')
    for i in bs.index:
        if float(bs[7][i]) >= bs_cutoff:
            # Set the [pdb][pdb_ligand] dict to BSscore
            scores[pdb][bs[0][i]+"_"+bs[6][i]] = float(bs[7][i])
    return {pdb:scores[pdb]}


def score_fp(fp,cmpd_file,cmpd_id,bs):
    l = []
    # Use RDkit
    if fp[0] == 'rd':
        try:
            cmpd = Chem.MolFromPDBFile(cmpd_file)
            # ECFP4 - extended connectivity fingerprint
            if fp[1] == 'ecfp4':
                cmpd_fp = AllChem.GetMorganFingerprintAsBitVect(cmpd,2,nBits=1024)
            # Daylight
            elif fp[1] == 'daylight':
                cmpd_fp = rdmolops.RDKFingerprint(cmpd)
            else:
                return
            bit_fp = DataStructs.BitVectToText(cmpd_fp)
        except:
            print ("Reading Exception: ", cmpd_id)
            for pdb in bs.index:
                l.append(0.000)
                continue
        print("Calculating tanimoto scores for compound {} against all binding site ligands...".format(cmpd_id))
        for pdb in bs.index:
            if bs.loc[(pdb)][1] == '':
                l.append(0.000)
                continue
            try:
                # Tanimoto similarity
                score = tanimoto_sparse(bit_fp,str(bs.loc[(pdb)][1]))
                #cmpd_scores.at[pdb, cmpd_id] = score
                l.append(score)
            except:
                #print("Fingerprint Exception: ", pdb)
                l.append(0.000)
                continue
    # Use OpenBabel
    elif fp[0] == 'ob':
        cmpd = next(pybel.readfile("pdb", cmpd_file))
        # FP2 - Daylight
        if fp[1] == 'fp2':
            cmpd_fp = cmpd.calcfp('fp2')
        # FP4 - SMARTS
        elif fp[1] == 'fp4':
            cmpd_fp = cmpd.calcfp('fp4')
        print("Calculating tanimoto scores for {} against all binding site ligands...".format(cmpd_id))
        for pdb in bs.index:
            if bs.loc[(pdb)][1] == '':
                #print("Empty Fingerprint: ", pdb)
                l.append(0.000)
                continue
            bs_fp = bs.loc[(pdb)][1].split(',')
            bs_fp = [int(bs_fp[x]) for x in range(len(bs_fp))]
            score = tanimoto_dense(bs_fp,cmpd_fp.bits)
            #cmpd_scores.at[pdb, i] = 0.00
            l.append(0.000)
    return {cmpd_id:l}


def generate_scores(fp="rd_ecfp4",cmpd_pdb='',cmpd_map='',cmpd_dir='.',out_path='.',ncpus=1):
    """Generate the fingerprint for new compound(s) and calculate 
    the tanimoto similarities against all binding site ligands.
    
    Parameters:
        fp (str): The fingerprinting software and method used, e.g. 'rd_ecfp4', 'ob_fp2'
        cmpd_map (str): tsv file containing the compound id and compound name. 
                The id for each compound should match the filename in cmpd_dir
                e.g. In the mapping you will have the line:
                        7561   sarcatinib
                the filename needs to be 7561.pdb
        cmpd_dir (int): Directory that contains compound pdb file(s).
    """
    fp_name = fp
    fp = fp.split("_")
    # Check for correct fingerprinting method
    if fp[0] not in ['rd','ob']:
        print("{} is not a correct fingerprinting method.".format(fp_name))
    else: 
        if fp[0] == 'ob' and fp[1] not in ['fp4','fp2']:
            print("{} is not a correct fingerprinting method.".format(fp_name))
        elif fp[0] == 'rd' and fp[1] not in ['daylight','ecfp4']:
            print("{} is not a correct fingerprinting method.".format(fp_name))

    # Pull and read in fingerprints for ligands
    #if not os.path.exists("v2_0/ligands_fps/{}.tsv".format(fp_name)):
    get_fp_lig(fp_name)
    pre = os.path.dirname(__file__)
    bs = pd.read_csv("{}/v2_0/ligands_fps/{}.tsv".format(pre,fp_name),sep='\t',header=None,index_col=0)
    bs = bs.replace(np.nan,'',regex=True)
    sites = bs.index
    print("Generating {} fingerprints and scores...".format('_'.join(fp)))
    if cmpd_pdb != '':
        cmpd_name = cmpd_pdb.split('/')[-1].split('.')[0]
        try:
            cmpd_id = int(cmpd_name)
        except:
            cmpd_id = 0
        out_name = "{}_scores.tsv".format(cmpd_id)
        scores = [score_fp(fp,cmpd_pdb,cmpd_id,bs)]
    elif cmpd_map != '':    
        out_name = "{}_scores.tsv".format(cmpd_map.split('/')[-1].split('.')[0])
        # Create df of compounds
        cmpds = pd.read_csv(cmpd_map,sep='\t',header=None)
        pool = mp.Pool(ncpus)
        scores = pool.starmap_async(score_fp, [(fp,"{}/{}.pdb".format(cmpd_dir,cmpds.loc[c][0]),cmpds.loc[c][0],bs) for c in cmpds.index]).get()
        pool.close()

    cmpd_scores = pd.DataFrame(index=sites)
    cmpd_scores = cmpd_scores.T
    for i in scores:
        for key,value in i.items():
            temp = pd.DataFrame({key:value},index=sites)
            cmpd_scores = cmpd_scores.append(temp.T)
    cmpd_scores = cmpd_scores.T

    if not os.path.exists('{}/{}'.format(out_path,fp_name)):
        os.makedirs('{}/{}'.format(out_path,fp_name))

    #for cmpd_id in new_cmpds.index:
    cmpd_scores.to_csv('{}/{}/{}'.format(out_path,fp_name,out_name),index=True,header=True,sep='\t',float_format='%.3f')
    print("Tanimoto scores written to {}/{}/{}\n".format(out_path,fp_name,out_name))


def tanimoto_sparse(str1,str2):
    n_c = 0.0
    n_a = 0.0
    n_b = 0.0
    for i in range(len(str1)):
        if str1[i] == '1' and str2[i] == '1':
            n_c += 1
        if str1[i] == '1':
            n_a += 1
        if str2[i] == '1':
            n_b += 1
    if n_c + n_a + n_b == 0:
        return 0.000
    return float(n_c/(n_a+n_b-n_c))    


def tanimoto_dense(list1, list2):
    c = [common_item for common_item in list1 if common_item in list2]
    return float(len(c))/(len(list1) + len(list2) - len(c))


def get_fp_lig(fp):
    print('Downloading ligand fingerprints for {}...'.format(fp))
    #if not os.path.exists('v2_0'):
    #    os.mkdir('v2_0')
    #if not os.path.exists('v2_0/ligands_fps'):
    #    os.mkdir('v2_0/ligands_fps')
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2_0/ligands_fps/{}.tsv'.format(fp)
    pre = os.path.dirname(__file__)
    out_file = '{}/v2_0/ligands_fps/{}.tsv'.format(pre,fp)
    dl_file(url,out_file)
    print("Ligand fingerprints downloaded.")


def get_v2_0():
    print('Downloading data for v2_0...')
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2_0/mappings/drugbank-approved.tsv'
    dl_file(url,'v2_0/mappings/drugbank-approved.tsv')
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2_0/mappings/drugbank-all.tsv'
    dl_file(url,'v2_0/mappings/drugbank-all.tsv')
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2_0/mappings/ctd_2_drugbank.tsv'
    dl_file(url,'v2_0/mappings/ctd_2_drugbank.tsv')
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2_0/cmpds/scores/drugbank-approved-rd_ecfp4.tsv.gz'
    dl_file(url,'v2_0/cmpds/scores/drugbank-approved-rd_ecfp4.tsv.gz')
    os.chdir("v2_0/cmpds/scores")
    os.system("gunzip drugbank-approved-rd_ecfp4.tsv.gz")
    os.chdir("../../..")


def get_tutorial():
    print('Downloading data for tutorial...')
    if not os.path.exists('examples'):
        os.mkdir('examples')
    # Example matrix (rd_ecfp4 w/ 64 prots x 2,162 drugs)
    if not os.path.exists('./examples/example-matrix.tsv'):
        url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2_0/examples/example-matrix.tsv'
        dl_file(url,'./examples/example-matrix.tsv')
    # Protein scores
    if not os.path.exists('./examples/example-prots_scores.tsv'):
        url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2_0/examples/example-prots_scores.tsv'
        dl_file(url,'./examples/example-prots_scores.tsv')

    if not os.path.exists('v2_0'):
        os.mkdir('v2_0')
    if not os.path.exists('v2_0/mappings'):
        os.mkdir('v2_0/mappings')
    # Compound mapping
    if not os.path.exists('v2_0/mappings/drugbank-approved.tsv'):
        url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2_0/mappings/drugbank-approved.tsv'
        dl_file(url,'v2_0/mappings/drugbank-approved.tsv')
    # Compound-indication mapping (CTD)
    if not os.path.exists('v2_0/mappings/ctd_2_drugbank.tsv'):
        url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2_0/mappings/ctd_2_drugbank.tsv'
        dl_file(url,'v2_0/mappings/ctd_2_drugbank.tsv')
    # Compounds scores
    if not os.path.exists('v2_0/cmpds/'):
        os.mkdir('v2_0/cmpds')
    if not os.path.exists('v2_0/cmpds/scores'):
        os.mkdir('v2_0/cmpds/scores')
    if not os.path.exists('v2_0/cmpds/scores/drugbank-approved-rd_ecfp4.tsv'):
        url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2_0/cmpds/scores/drugbank-approved-rd_ecfp4.tsv.gz'
        dl_file(url,'v2_0/cmpds/scores/drugbank-approved-rd_ecfp4.tsv.gz')
        os.chdir("v2_0/cmpds/scores")
        os.system("gunzip drugbank-approved-rd_ecfp4.tsv.gz")
        os.chdir("../../..")
    # New compound
    if not os.path.exists('./examples/8100.pdb'):
        url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2_0/examples/8100.pdb'
        dl_file(url,'./examples/8100.pdb')
    print('All data for tutorial downloaded.')
    # Pathways
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2_0/examples/example-uniprot_set'
    dl_file(url,'examples/example-uniprot_set')


def get_test():
    print('Downloading data for generate_matrix...')
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2_0/test/test-cmpd_scores.tsv'
    dl_file(url,'test/test-cmpd_scores.tsv')
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2_0/test/test-prot_scores.tsv'
    dl_file(url,'test/test-prot_scores.tsv')
    print('Done.\n')
    print('Downloading data for benchmark_classic...')
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2_0/test/test-cmpds.tsv'
    dl_file(url,'test/test-cmpds.tsv')
    with open('test/test-cmpds.tsv','r') as f:
        l = []
        for i in f:
            i = i.split('\t')[0]
            i = "{}.pdb".format(i)
            l.append(i)
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2_0/test/test-cmpds_pdb'
    out = 'test/test-cmpds_pdb'
    dl_dir(url,out,l)
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2_0/test/test-inds.tsv'
    dl_file(url,'test/test-inds.tsv')
    print('Done.\n')
    print('Downloading data for generate_fp...')
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2_0/test/8100.pdb'
    dl_file(url,'test/8100.pdb')
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2_0/test/test-cmpds_short.tsv'
    dl_file(url,'test/test-cmpds_short.tsv')
    print('Done.\n')
    print('Downloading data for pathways...')
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2_0/test/test-uniprot_set'
    dl_file(url,'test/test-uniprot_set')
    print('Done.\n')


def dl_dir(url,out,l):
    if not os.path.exists(out):
        os.makedirs(out)
    format_custom_text = progressbar.FormatCustomText(
        'Downloading dir: %(f)s',
        dict(
            f='',
        ),
    )
    widgets=[
        format_custom_text,
        ' [', progressbar.DataSize(format='%(scaled)i files',), '] ',
        progressbar.Bar(left='[', right=']'),
        ' [', progressbar.ETA(), '] ',
    ]
    num_bars = len(l)
    bar = progressbar.ProgressBar(max_value=num_bars, widgets=widgets).start()
    i = 0
    for n in l:
        format_custom_text.update_mapping(f=out)
        url2 = "{}/{}".format(url,n)
        r = requests.get(url2)
        out_file = "{}/{}".format(out,n)
        with open(out_file, 'wb') as f:
            f.write(r.content)
        bar.update(i)
        i+=1
    bar.finish()


def dl_file(url,out_file):
    if os.path.exists(out_file):
        return
    elif not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
    r = requests.get(url,stream=True)
    format_custom_text = progressbar.FormatCustomText(
        'Downloading file: %(f)s',
        dict(
            f='',
        ),
    )
    widgets=[
        format_custom_text,
        ' [', progressbar.DataSize(prefixes=('K', 'M', 'G')), '] ',
        progressbar.Bar(left='[', right=']'),
        ' [', progressbar.ETA(), '] ',
    ]
    with open(out_file, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        if total_length >= 1024:
            chunk_size = 1024
            num_bars = total_length / chunk_size
        else:
            chunk_size = 1
            num_bars = total_length / chunk_size
        bar = progressbar.ProgressBar(max_value=num_bars, widgets=widgets).start()
        i = 0
        for chunk in r.iter_content(chunk_size=chunk_size):
            format_custom_text.update_mapping(f=out_file)
            f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
            bar.update(i)
            i+=1
        bar.finish()

