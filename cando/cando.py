import os, sys, pickle
import requests
import random
import time
import operator
import math
import ast
import numpy as np
import pandas as pd
import multiprocessing as mp
import difflib
import matplotlib.pyplot as plt
from decimal import Decimal
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked, roc_curve, roc_auc_score, average_precision_score, ndcg_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import squareform
from scipy import stats
import shutil # change1
from scipy.stats import hypergeom

import sqlite3, json
from sqlalchemy import create_engine
import polars as pl

class Protein(object):
    """!
    An object to represent a protein
    """
    def __init__(self, id_, sig):
        ## @var id_ 
        #   PDB or UniProt ID for the given protein
        self.id_ = id_
        ## @var alt_id
        #   Used when a second identifier mapping is available (such as SIFTs project)
        self.alt_id = ''
        ## @var sig 
        #   List of scores representing each drug interaction with the given protein
        self.sig = sig
        ## @var pathways 
        #   List of Pathway objects in which the given protein is involved
        self.pathways = []
        ## @var indications
        #   List of Indication objects to which the protein is associated
        self.indications = []
        ## @var name
        #   str: the common name of the protein (not currently used)
        self.name = ''
        ## @var gene
        #   str: the gene name from which the protein is produced
        self.gene = ''


class Compound(object):
    """!
    An object to represent a compound/drug
    """
    def __init__(self, name, id_, index, status='N/A'):
        ## @var name 
        # str: Name of the Compound (e.g., 'caffeine')
        self.name = name
        ## @var id_
        # int: CANDO id from mapping file (e.g., 1, 10, 100, ...)
        self.id_ = id_
        ## @var index
        # int: The order in which the Compound appears in the mapping file (e.g, 1, 2, 3, ...)
        self.index = index
        ## @var status
        # str: The clinical trial status of the compound from DrugBank ('approved' or 'other')
        self.status = status
        ## @var sig
        # list: Signature is essentially a column of the Matrix
        self.sig = []
        ## @var aux_sig
        # list: Potentially temporary signature for things like pathways, where "c.sig" needs to be preserved
        self.aux_sig = []
        ## @var indications
        # list: This is every indication the Compound is associated with from the
        # mapping file
        self.indications = []
        ## @var similar
        # list: This is the ranked list of compounds with the most similar interaction signatures
        self.similar = []
        ## @var similar_computed
        # bool: Have the distances of all Compounds to the given Compound been computed?
        self.similar_computed = False
        ## @var similar_sorted
        # bool: Have the most similar Compounds to the given Compound been sorted?
        self.similar_sorted = False
        ## @var cluster_id
        # int: The cluster id this Compound was assigned from clustering method
        self.cluster_id = []
        ## @var adrs
        # list: List of ADRs associated with this Compound
        self.adrs = []
        ## @var alt_ids
        # dict: dict of other ids inputted with compound mapping
        self.alt_ids = {}
        ## @var metabolites
        # list: List of all metabolites from the compound
        self.metabolites = []
        ## @var is_metabolite
        # bool: bool if the drug is a metabolite itself
        self.is_metabolite = False
        ## @var parent
        # Compound: Compound object to which this compound is a metabolite
        self.parent = None
        ## @var compounds
        # List Compound: Compound objects to which this compound is associated
        self.compounds = []

    def add_indication(self, ind):
        """!
        Add an Indication to the list of Indications associated to this Compound

        @param ind object: Indication object to add
        """
        self.indications.append(ind)


class Compound_pair(object):
    """!
    An object to represent a compound/drug-pair
    """
    def __init__(self, name, id_, index):
        ## @var name
        # str: Name of the Compound (e.g., 'caffeine')
        self.name = name
        ## @var id_
        # int: CANDO id from mapping file (e.g., 1, 10, 100, ...)
        self.id_ = id_
        ## @var index
        # int: The order in which the Compound appears in the mapping file (e.g, 1, 2, 3, ...)
        self.index = index
        ## @var sig
        # list: Signature is essentially a column of the Matrix
        self.sig = []
        ## @var aux_sig
        # list: Potentially temporary signature for things like pathways, where "c.sig" needs to be preserved
        self.aux_sig = []
        ## @var similar
        # list: This is the ranked list of compound_pairs with the most similar interaction signatures
        self.similar = []
        ## @var similar_computed
        # bool: Have the distances of all Compounds to the given Compound been computed?
        self.similar_computed = False
        ## @var similar_sorted
        # bool: Have the most similar Compounds to the given Compound been sorted?
        self.similar_sorted = False
        ## @var adrs
        # list: List of ADRs associated with this Compound
        self.adrs = []

    def add_adr(self, adr):
        """!
        Add an ADR to the list of Indications associated to this Compound

        @param ind object: Indication object to add
        """
        self.adrs.append(adr)


class Indication(object):
    """!
    An object to represent an indication (disease)
    """
    def __init__(self, ind_id, name):
        ## @var id_
        # str: MeSH or OMIM ID for the indication from the mapping file
        self.id_ = ind_id
        ## @var name
        # str: Name for the indication from the mapping file
        self.name = name
        ## @var compounds
        # list: Every associated compound object from the mapping file
        self.compounds = []
        ## @var pathways
        # list: Every pathway associated to the indication from the mapping file
        self.pathways = []
        ## @var proteins
        # list: Every protein associated to the indication form the mapping file
        self.proteins = []
        ## @var pathogen
        # bool: Whether or not this indication is caused by a pathogen
        self.pathogen = None


class Pathway(object):
    """!
    An object to represent a pathway
    """
    def __init__(self, id_):
        ## @var proteins
        # list: Protein objects associated with the given Pathway
        self.proteins = []
        ## @var id_
        # str: Identification for the given Pathway
        self.id_ = id_
        ## @var indications
        # list: Indication objects associated with the given Pathway
        self.indications = []


class ADR(object):
    """!
    An object to represent an adverse reaction
    """
    def __init__(self, id_, name):
        ## @var id_
        # str: Identification for the given ADR
        self.id_ = id_
        ## @var name
        # str: Name of the given ADR
        self.name = name
        ## @var compounds
        # list: Compound objects associated with the given ADR
        self.compounds = []
        ## @var compound_pairs
        # List: Compound_pair object pairs (tuples) associated with the given ADR
        self.compound_pairs = []


class CANDO(object):
    """!
    An object to represent all aspects of CANDO (compounds, indications, matrix, etc.)

    To instantiate you need the compound mapping (c_map), an
    indication mapping file (i_map), and typically and a compound-protein matrix (matrix=) or
    or precomputed compound-compound distance matrix (read_rmsds=), but those are optional.

    """
    def __init__(self, c_map, i_map, matrix='', db_name='', compound_set='all', compute_distance=False, save_dists='',
                 read_dists='', pathways='', pathway_quantifier='max', indication_pathways='', indication_proteins='',
                 similarity=False, dist_metric='rmsd', protein_set='', rm_zeros=False, rm_compounds='',
                 ddi_compounds='', ddi_adrs='', adr_map='', sig_fusion='sum', 
                 protein_distance=False, protein_map='', ncpus=1, pbar=True):
        ## @var c_map
        # str: File path to the compound mapping file (relative or absolute)
        self.c_map = c_map
        ## @var i_map 
        # str: File path to the indication mapping file (relative or absolute)
        self.i_map = i_map
        ## @var matrix 
        # str: File path to the cando matrix file (relative or absolute)
        self.matrix = matrix
        ## @var db_name
        # str: Name of the sqlite database file that will be created
        self.db_name = db_name
        ## @var compound_set
        # str or List str: what compounds to use, such as all, approved, experimental, etc
        self.compound_set = compound_set
        ## @var protein_set
        # str: File path to protein subset file (relative or absolute) 
        self.protein_set = protein_set
        ## @var pathways
        # str: File path to pathway file
        self.pathways = []
        self.accuracies = {}
        ## @var compute_distance
        # bool: Calculate the distance for each Compound against all other Compounds using chosen distance metric
        self.compute_distance = compute_distance
        ## @var protein_distance
        # bool: Calculate the distance for each Protein against all other Proteins using chosen distance metric
        self.protein_distance = protein_distance
        self.clusters = {}
        ## @var rm_zeros
        # bool: Remove Compounds with all-zero signatures from CANDO object
        self.rm_zeros = rm_zeros
        ## @var rm_compounds
        # list: Compounds to remove from the CANDO object 
        self.rm_compounds = rm_compounds
        self.rm_cmpds = []
        ## @var save_dists
        # bool: Write the calculated distances to file after computation (set compute_distances=True)
        self.save_dists = save_dists
        ## @var read_dists
        # str: File path to pre-computed distance matrix
        self.read_dists = read_dists
        ## @var similarity
        # bool: Use similarity instead of distance
        self.similarity = similarity
        ## @var dist_metric
        # str: Distance metric to be used for computing Compound-Compound distances
        self.dist_metric = dist_metric
        ## @var ncpus
        # int: Number of CPUs used for parallelization
        self.ncpus = int(ncpus)
        ## @var pathway_quantifier
        # str: Method used to quantify a all Pathways
        self.pathway_quantifier = pathway_quantifier
        ## @var indication_pathways
        # str: File path to Indication-Pathway association file
        self.indication_pathways = indication_pathways
        ## @var indication_proteins
        # str: File path to Indication-Protein association file
        self.indication_proteins = indication_proteins
        ## @var adr_map
        # str: File path to ADR mapping file
        self.adr_map = adr_map
        ## @var protein_map
        # str: File path to Protein metadata mapping file
        self.protein_map = protein_map
        ## @var ddi_compounds
        # str: File path to Drug--drug mapping file
        self.ddi_compounds = ddi_compounds
        ## @var ddi_compounds
        # str: File path to Drug--Drug--ADE mapping file
        self.ddi_adrs = ddi_adrs
        
        ## @var pbar
        # bool: Use TQDM progress bar for certain processes
        self.pbar = pbar

        ## @var proteins
        # List: Protein objects in the platform
        self.proteins = []
        self.protein_id_to_index = {}
        ## @var compounds
        # List: Compound objects in the platform
        self.compounds = []
        self.compound_ids = []
        ## @var compound_pairs
        # List: Compound_pair objects in the platform
        self.compound_pairs = []
        self.compound_pair_ids = []
        ## @var indications
        # List: Indication objects in the platform
        self.indications = []
        self.indication_ids = []
        ## @var adrs
        # List: ADR objects in the platform
        self.adrs = []
        self.adr_ids = []
        ## @var sig_fusion
        # str: determines the method by which the signatures are combined for DDI analysis
        self.sig_fusion = sig_fusion
        self.short_matrix_path = self.matrix.replace('\\', '/').split('/')[-1]
        self.short_read_dists = read_dists.replace('\\', '/').split('/')[-1]
        self.short_protein_set = protein_set.replace('\\', '/').split('/')[-1]
        self.cmpd_set = rm_compounds.replace('\\', '/').split('/')[-1]
        self.data_name = ''
        self.short_c_map = c_map.replace('\\', '/').split('/')[-1]
        self.short_i_map = i_map.replace('\\', '/').split('/')[-1]


        if self.matrix:
            if self.protein_set:
                self.data_name = self.short_protein_set + '.' + self.short_matrix_path
            elif rm_compounds:
                self.data_name = self.cmpd_set + '.' + self.short_matrix_path
        if self.short_read_dists:
            self.data_name = self.short_read_dists

        if not self.db_name:
            self.db_name = f"cando-{self.short_matrix_path.replace('.tsv','')}-{self.short_c_map.replace('.tsv','')}-{self.short_i_map.replace('.tsv','')}.db"

        ignored_set = []
        # create all of the compound objects from the compound map
        with open(c_map, 'r', encoding="utf8") as c_f:
            lines = c_f.readlines()
            header = lines[0]
            h2i = {}
            for i, h in enumerate(header.strip().split('\t')):
                h2i[h] = i
            for l in lines[1:]:
                ls = l.strip().split('\t')
                name = ls[h2i['GENERIC_NAME']]
                id_ = int(ls[h2i['CANDO_ID']])
                db_id = ls[h2i['DRUGBANK_ID']]
                index = id_
                cm = Compound(name, id_, index)

                assert isinstance(self.compound_set, str) or isinstance(self.compound_set, list), \
                       'compound_set flag has wrong input type, please input a string compound category ("all", '+\
                          '"approved", etc) or a list of categories (["approved", "experimental"])'
                include_cmpd = False
                if self.compound_set == 'all':
                    include_cmpd = True
                    tags = None
                elif isinstance(self.compound_set, str):
                    tags = [self.compound_set]
                elif isinstance(self.compound_set, list):
                    tags = self.compound_set
                #else:
                #    tags = None
                #    print('compound_set flag has wrong input type, please input a string compound category ("all", '
                #          '"approved", etc) or a list of categories (["approved", "experimental"])')
                #    quit()

                if 'DRUG_GROUPS' in h2i:
                    stati = ls[h2i['DRUG_GROUPS']].split(';')
                    if tags is not None:
                        if len(list(set(tags) & set(stati))) > 0:
                            include_cmpd = True
                        else:
                            ignored_set.append(id_)
                            continue
                    if 'approved' in stati:
                        cm.status = 'approved'
                    elif 'metabolite' in stati:
                        cm.status = 'other'
                        cm.is_metabolite = True
                    else:
                        cm.status = 'other'
                else:
                    if self.compound_set != 'all':
                        print('This mapping does not have drug groups/approval status - '
                              'please re-run with compound_set="all".')
                        sys.exit()
                    cm.status = 'N/A'

                if include_cmpd:
                    self.compounds.append(cm)
                    self.compound_ids.append(id_)
        del lines

        assert not self.compound_set or len(self.compounds) > 0,\
               'No compounds passed filtering, please check input parameters.'

        #if self.compound_set and len(self.compounds) == 0:
        #   print('No compounds passed filtering, please check input parameters.')
        #   quit()

        # create the indication objects and add indications to the
        # already created compound objects from previous loop
        # NOTE: if a compound is in the indication mapping file that
        # isn't in the compound mapping file, an error will occur. I
        # had to remove those compounds from the indication mapping in
        # order for it to work
        with open(i_map, 'r', encoding="utf8") as i_f:
            lines = i_f.readlines()
            header = lines[0]
            h2i = {}
            for i, h in enumerate(header.strip().split('\t')):
                h2i[h] = i
            for l in lines[1:]:
                ls = l.strip().split('\t')
                c_id = int(ls[h2i['CANDO_ID']])
                if c_id in ignored_set:
                    continue
                i_name = ls[h2i['INDICATION_NAME']]
                ind_id = ls[h2i['MESH_ID']]
                cm = self.get_compound(c_id, quiet=True)
                if cm:
                    if ind_id in self.indication_ids:
                        ind = self.get_indication(ind_id)
                        # Modified to remove object within an object
                        # Reduce memory usage
                        #ind.compounds.append(cm)
                        ind.compounds.append(c_id)
                    else:
                        ind = Indication(ind_id, i_name)
                        # Modified to remove object within an object
                        # Reduce memory usage
                        #ind.compounds.append(cm)
                        ind.compounds.append(c_id)
                        self.indications.append(ind)
                        self.indication_ids.append(ind.id_)
                    # Modified to remove object within an object
                    # Reduce memory usage
                    #cm.add_indication(ind)
                    cm.add_indication(ind_id)
        del lines

        # add proteins, add signatures and such to compounds
        if self.protein_set:
            uniprots = []
            with open(self.protein_set, 'r', encoding="utf8") as psf:
                lines = psf.readlines()
                for line in lines:
                    uni = line.strip()
                    uniprots.append(uni)

        if matrix:
            assert matrix[-4:] != '.fpt', 'The matrix file {} is in the old fpt format -- please '\
                      'convert to tsv with the following line of code:\n>> Matrix({}, convert_to_tsv=True)'.format(matrix,matrix)
            #if matrix[-4:] == '.fpt':
            #    print('The matrix file {} is in the old fpt format -- please '
            #          'convert to tsv with the following line of code:'.format(matrix))
            #    print('>> Matrix({}, convert_to_tsv=True)'.format(matrix))
            #   quit()
            print('Reading signatures from matrix...')

            '''
            # SQLite approach to saving signatures
            # Need to update other functions to account for the sigs not being in the Compound objects
            check = False
            try:
                conn = f'sqlite://{self.db_name}'
                #db = create_engine(f'sqlite:///{self.db_name}')
                # db = create_engine('sqlite:///cando.db')
                df_sigs = pl.read_database(query="SELECT * FROM sigs", connection=conn)
                sigs = json.loads(
                    df_sigs.filter(pl.col('id') == self.compounds[0]).select(['sig']).item().replace("'", '"'))
                if len(df_sigs) == len(self.compounds) and len(df_cmpds) == len(self.compounds):
                    print("  Data already in tables.")
                    check = True
            except:
                print("  Data does not exist OR data does not match.")
                os.system(f"rm {self.db_name}")

            if not check:
                db = create_engine(f'sqlite:///{self.db_name}')
                df_sigs = pd.read_csv(self.matrix, sep='\t', header=None, index_col=0)
                if len(df_sigs.columns) != len(self.proteins):
                    print(f'The number of proteins in {self.i_map} does not match the number of rows in {self.matrix}.')
                    quit()
                if len(df_sigs.index) != len(self.compounds):
                    print(f'The number of compounds in {self.c_map} does not match the number of columns in {self.matrix}.')
                    quit()
                df_sigs = df_sigs.T.reset_index(drop=True)
                d_sigs = {}
                for c in df_sigs.index:
                    d_temp = {str(p): df_sigs.loc[c,p] for p in df_sigs.columns}
                    d_sigs[str(c)] = str(d_temp)
                    if c % 1000 == 0 or c == len(self.compounds) - 1:
                        df_temp = pd.DataFrame.from_dict(d_sigs, orient='index')
                        df_temp.index.names = ['id']
                        df_temp.rename(columns={0: 'sig'}, inplace=True)
                        df_temp.to_sql('sigs', db, if_exists='append')
                        del df_temp
                        d_sig = {}
                del df_sigs, d_sigs, d_temp
            '''
            with open(matrix, 'r', encoding="utf8") as m_f:
                m_lines = m_f.readlines()
                if self.protein_set:
                    print('Editing signatures according to proteins in {}...'.format(self.protein_set))
                    targets, pdct_rev = self.uniprot_set_index(self.protein_set)
                    new_i = 0
                    matches = [0, 0]
                    for l_i in range(len(m_lines)):
                        vec = m_lines[l_i].strip().split('\t')
                        name = vec[0]
                        if name in targets:
                            scores = list(map(float, vec[1:]))
                            assert len(scores) == len(self.compounds), 'The number of compounds in {} does not match the '\
                                      'number of values in {}'.format(self.c_map, self.matrix)
                            #if len(scores) != len(self.compounds):
                            #    print('The number of compounds in {} does not match the '
                            #          'number of values in {} -- quitting.'.format(self.c_map, self.matrix))
                            #    quit()
                            p = Protein(name, scores)
                            try:
                                alt = pdct_rev[name]
                                p.alt_id = alt
                                matches[0] += 1
                            except KeyError:
                                matches[1] += 1
                            self.proteins.append(p)
                            self.protein_id_to_index[name] = new_i
                            for i in range(len(scores)):
                                s = scores[i]
                                self.compounds[i].sig.append(s)
                            new_i += 1
                        else:
                            continue
                    print('    Direct UniProt matches: {}\n    Direct PDB matches: {}'
                          '\n    New signature length: {}'.format(matches[1], matches[0], sum(matches)))
                    assert sum(matches), 'The input proteins did not match any proteins in the input matrix'
                    #if not sum(matches):
                    #   print('Sorry, the input proteins did not match any proteins in the input matrix -- quitting.')
                    #   quit()
                else:
                    for l_i in range(len(m_lines)):
                        vec = m_lines[l_i].strip().split('\t')
                        name = vec[0]
                        scores = list(map(float, vec[1:]))
                        assert len(scores) == len(self.compounds), 'The number of compounds in {} does not match the '\
                                  'number of values in {}'.format(self.c_map, self.matrix)
                        #if len(scores) != len(self.compounds):
                        #    print('The number of compounds in {} does not match the '
                        #          'number of values in {} -- quitting.'.format(self.c_map, self.matrix))
                        #    quit()
                        self.proteins.append(Protein(name, scores))
                        self.protein_id_to_index[name] = l_i
                        for i in range(len(scores)):
                            self.compounds[i].sig.append(scores[i])
            del m_lines, scores, vec
            print('Done reading signatures.\n')

        if pathways:
            print('Reading pathways...')
            if self.indication_pathways:
                print('Reading indication-pathway associations...')
                path_ind = {}
                with open(indication_pathways, 'r', encoding="utf8") as ipf:
                    for l in ipf:
                        ls = l.strip().split('\t')
                        pw = ls[0]
                        ind_ids = ls[1:]
                        path_ind[pw] = ind_ids

            with open(pathways, 'r', encoding="utf8") as pf:
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
                            pass

                    if self.indication_pathways:
                        try:
                            ind_ids = path_ind[pw]
                            for ind_id in ind_ids:
                                try:
                                    ind = self.get_indication(ind_id)
                                except LookupError:
                                    continue
                                PW.indications.append(ind)
                                ind.pathways.append(PW)
                        except KeyError:
                            continue
            if not indication_pathways:
                self.quantify_pathways()
            print('Done reading pathways.')

        # Create SQLITE db for CANDO libraries and data
        print("Building compound-indication tables...")

        check = False
        print("  Checking if data exists in tables...")
        try:
            db = create_engine(f'sqlite:///{self.db_name}')
            #db = create_engine('sqlite:///cando.db')
            df_inds = pd.read_sql("SELECT * FROM inds", db)
            df_cmpds = pd.read_sql("SELECT * FROM cmpds", db)
            if len(df_inds) == len(self.indications) and len(df_cmpds) == len(self.compounds):
                print("  Data already in tables.")
                check = True
        except:
            print("  Data does not exist OR data does not match.")
            #os.system(f"rm {self.db_name}")

        if not check:
            db = create_engine(f'sqlite:///{self.db_name}')
            #db = create_engine('sqlite:///cando.db')
            print("    indications")
            #d_inds = {str(x.id_): [str(x.name), str([cmpd.id_ for cmpd in x.compounds])] for x in tqdm(self.indications)}
            pbar = tqdm(self.indications) if self.pbar else self.indications
            d_inds = {str(x.id_): [str(x.name), str([cmpd for cmpd in x.compounds])] for x in pbar}
            df_inds = pd.DataFrame.from_dict(d_inds, orient='index')
            df_inds.index.names = ['id']
            df_inds.rename(columns={0: 'name', 1: 'cmpd_ids'}, inplace=True)
            df_inds.to_sql('inds', db, if_exists='replace')
            print("    compounds")
            #d_cmpds = {str(x.id_): [str(x.name), str([ind.id_ for ind in x.indications])] for x in tqdm(self.compounds)}
            pbar = tqdm(self.compounds) if self.pbar else self.compounds
            d_cmpds = {str(x.id_): [str(x.name), str([ind for ind in x.indications])] for x in pbar}
            df_cmpds = pd.DataFrame.from_dict(d_cmpds, orient='index')
            df_cmpds.index.names = ['id']
            df_cmpds.rename(columns={0: 'name', 1: 'ind_ids'}, inplace=True)
            df_cmpds.to_sql('cmpds', db, if_exists='replace')
            del d_inds, df_inds, d_cmpds, df_cmpds
            # Index on id for cmpds and inds tables
            conn = sqlite3.connect(f'{self.db_name}')
            #conn = sqlite3.connect('cando.db')
            cursor = conn.cursor()
            cursor.execute("CREATE INDEX c_id_idx ON cmpds(id)")
            cursor.execute("CREATE INDEX i_id_idx ON inds(id)")
            conn.commit()
            conn.close()
        print("Done building compound-indication tables.\n")


        if self.ddi_compounds:
            print("Reading compound-compound associations...")
            ddi = pd.read_csv(ddi_compounds, sep='\t')
            for x in ddi.index:
                c1 = self.get_compound(int(ddi.loc[x,'CANDO_ID-1']))
                c2 = self.get_compound(int(ddi.loc[x,'CANDO_ID-2']))
                if c2 not in c1.compounds:
                    c1.compounds.append(c2)
                if c1 not in c2.compounds:
                    c2.compounds.append(c1)
            print('Done reading compound-compound associations.\n')

        if self.ddi_adrs:
            print("Building compound pair-adverse events tables...")
            print("  Checking if data exists in tables...")
            check = False
            # Check
            # If pass, do not recreate but pull from db
            try:
                #db = create_engine('sqlite:///ddi.db')
                db = create_engine(f'sqlite:///{self.db_name}')
                # Pull ADRs from table
                df_adrs = pd.read_sql("SELECT * FROM adrs",db)
                # Get ADRs from inputted cmpd_pair-ind mapping
                ddi = pd.read_csv(self.ddi_adrs, sep='\t')
                l_adrs = ddi['COND_DB_ID'].drop_duplicates().to_list()
                # Pull cmpd_pairs from table
                df_cps = pd.read_sql("SELECT * FROM cmpd_pairs",db)
                # Get cmpd_pairs from inputted cmpd_pair-ind mapping
                l_cps = list(zip(ddi.loc[:,'CANDO_ID-1'].values.tolist(),ddi.loc[:,'CANDO_ID-2'].values.tolist()))
                l_cps = list(set(l_cps))
                #l_adrs = set(ddi.loc[:,'CONDITION_MESH_ID'].to_list())
                if len(df_adrs)==len(l_adrs) and len(df_cps)==len(l_cps):
                    print("  Data already in tables.")
                    check=True
                # Check other tables
                #l_cps = list(zip(ddi.loc[:,'CANDO_ID-1'].values.tolist(),ddi.loc[:,'CANDO_ID-2'].values.tolist()))
            except:
                print("  Data does not exist OR data does not match.")
                #os.system(f"rm {self.db_name}")

        if self.ddi_adrs and check==True:
            print("  Reading adverse drug reaction and compound pair associations from tables...")
            # Compound pairs
            db = create_engine(f'sqlite:///{self.db_name}')
            #db = create_engine('sqlite:///ddi.db')
            df_temp = pd.read_sql("SELECT * FROM cmpd_pairs",db)
            print("    compound pairs")
            pbar = tqdm(df_temp.index) if self.pbar else df_temp.index
            for i in pbar:
                cm_p = Compound_pair(ast.literal_eval(df_temp.loc[i,'names']), ast.literal_eval(df_temp.loc[i,'ids']), ast.literal_eval(df_temp.loc[i,'ids']))
                cm_p.adrs = ast.literal_eval(df_temp.loc[i,'adr_ids'])
                self.compound_pairs.append(cm_p)
                self.compound_pair_ids.append(cm_p.id_)

            # ADRs
            df_temp = pd.read_sql("SELECT * FROM adrs",db)
            print("    adverse drug reactions")
            pbar = tqdm(df_temp.index) if self.pbar else df_temp.index
            for i in pbar:
                adr = ADR(df_temp.loc[i,'id'],df_temp.loc[i,'name'])
                adr.compound_pairs = ast.literal_eval(df_temp.loc[i,'cmpd_pair_ids'])
                self.adrs.append(adr)
                self.adr_ids.append(adr.id_)
            print('  Done reading adverse drug reaction and compound pair associations.\n')

            '''
            # ADRS + Compound pairs
            df_temp = pd.read_sql("SELECT * FROM 'adr-cmpd_pairs'",db)
            # Add comppund pair to ADR and vice versa
            print("adr-compound pair associations")
            for i in tqdm(df_temp.index):
                adr = self.get_adr(df_temp.loc[i,'adrs'])
                for j in ast.literal_eval(df_temp.loc[i,'cmpd_pairs']):
                    cm_p = self.get_compound_pair(j)
                    adr.compound_pairs.append(cm_p)
                    cm_p.add_adr(adr)
            print('Done reading adverse drug reaction and compound pair associations.\n')
            '''

            print("Generating compound-compound signatures...")
            pbar = tqdm(self.compound_pairs) if self.pbar else self.compound_pairs
            for cm_p in pbar:
                c1 = self.get_compound(cm_p.id_[0])
                c2 = self.get_compound(cm_p.id_[1])
                # Add signatures??
                #cm_p.sig = [i+j for i,j in zip(c1.sig,c2.sig)]
                # max, min, mult?
                # ZF - mod 9/5/23
                # allow other types of vector combination
                if self.sig_fusion=='sum':
                    cm_p.sig = [i+j for i,j in zip(c1.sig,c2.sig)]
                elif self.sig_fusion=='product':
                    cm_p.sig = [i*j for i,j in zip(c1.sig,c2.sig)]
                elif self.sig_fusion=='min':
                    cm_p.sig = [min(i,j) for i,j in zip(c1.sig,c2.sig)]
                elif self.sig_fusion=='max':
                    cm_p.sig = [max(i,j) for i,j in zip(c1.sig,c2.sig)]
                elif self.sig_fusion=='average':
                    cm_p.sig = [np.mean([i,j]) for i,j in zip(c1.sig,c2.sig)]
 
            print("Done generating compound-compound signatures.\n")

        if self.ddi_adrs and check==False:
            #db = create_engine('sqlite:///ddi.db')
            db = create_engine(f'sqlite:///{self.db_name}')
            print("  Reading compound pair-adverse events associations...")
            ddi = pd.read_csv(ddi_adrs,sep='\t')
            # Create a unique set of tuples using CANDO IDs for compound pairs
            idss = list(zip(ddi.loc[:,'CANDO_ID-1'].values.tolist(),ddi.loc[:,'CANDO_ID-2'].values.tolist()))
            print("    {} compound pair-adverse event associations.".format(len(idss)))
            idss = list(set(idss))
            # Iterate through list of CANDO ID tuples
            pbar = tqdm(idss) if self.pbar else idss
            for ids in pbar: 
                if ids in self.compound_pair_ids:
                    cm_p = self.get_compound_pair(ids)
                elif (ids[1],ids[0]) in self.compound_pair_ids:
                    cm_p = self.get_compound_pair((ids[1],ids[0]))
                else:
                    names = (self.get_compound(ids[0]).name,self.get_compound(ids[1]).name)
                    cm_p = Compound_pair(names, ids, ids)
                    self.compound_pairs.append(cm_p)
                    self.compound_pair_ids.append(ids)
                # Pull list of ADRs for this compound pair
                adrs = ddi.loc[(ddi['CANDO_ID-1']==ids[0]) & (ddi['CANDO_ID-2']==ids[1])]
                # Iterate through ADRs for this compound pair 
                for x in adrs.index:
                    #ADRs
                    adr_name = ddi.loc[x,'COND_NAME']
                    adr_id = ddi.loc[x,'COND_DB_ID']
                    #adr_name = ddi.loc[x,'CONDITION_MESH_NAME']
                    #adr_id = ddi.loc[x,'CONDITION_MESH_ID']
                    if adr_id in self.adr_ids:
                        adr = self.get_adr(adr_id)
                    else:
                        adr = ADR(adr_id,adr_name)
                        self.adrs.append(adr)
                        self.adr_ids.append(adr.id_)
                    # Add comppund pair to ADR and vice versa
                    cm_p.add_adr(adr_id)
                    #cm_p.add_adr(adr)
                    adr.compound_pairs.append(ids)
                    #adr.compound_pairs.append(cm_p)
            print("    {} compound pairs.".format(len(self.compound_pairs)))
            print("    {} adverse events.".format(len(self.adrs)))
            print('  Done reading compound pair-adverse event associations.\n')

            print("  adverse drug reactions")
            pbar = tqdm(self.adrs) if self.pbar else self.adrs
            d_adrs = {str(x.id_): [str(x.name), str(x.compound_pairs)] for x in pbar}
            #d_adrs = {str(x.id_): str(x.name) for x in tqdm(self.adrs)}
            df_adrs = pd.DataFrame.from_dict(d_adrs,orient='index')
            df_adrs.index.names = ['id']
            df_adrs.rename(columns={0:'name',1:'cmpd_pair_ids'},inplace=True)
            #df_adrs.rename(columns={0:'name'},inplace=True)
            df_adrs.to_sql('adrs', db, if_exists='replace')
            
            print("  compound pairs")
            pbar = tqdm(self.compound_pairs) if self.pbar else self.compound_pairs
            d_cps = {str(x.id_): [str(x.name), str(x.adrs)] for x in pbar}
            #d_cps = {str(x.id_): str(x.name) for x in tqdm(self.compound_pairs)}
            df_cps = pd.DataFrame.from_dict(d_cps,orient='index')
            df_cps.index.names = ['ids']
            df_cps.rename(columns={0:'names', 1:'adr_ids'},inplace=True)
            #df_cps.rename(columns={0:'names'},inplace=True)
            df_cps.to_sql('cmpd_pairs', db, if_exists='replace')

            print("  Generating compound-compound signatures...")
            for cm_p in self.compound_pairs:
                c1 = self.get_compound(cm_p.id_[0])
                c2 = self.get_compound(cm_p.id_[1])
                # Add signatures??
                # max, min, mult?
                # ZF - mod 9/5/23
                # allow other types of vector combination
                if self.sig_fusion=='sum':
                    cm_p.sig = [i+j for i,j in zip(c1.sig,c2.sig)]
                elif self.sig_fusion=='product':
                    cm_p.sig = [i*j for i,j in zip(c1.sig,c2.sig)]
                elif self.sig_fusion=='min':
                    cm_p.sig = [min(i,j) for i,j in zip(c1.sig,c2.sig)]
                elif self.sig_fusion=='max':
                    cm_p.sig = [max(i,j) for i,j in zip(c1.sig,c2.sig)]
                elif self.sig_fusion=='average':
                    cm_p.sig = [np.mean([i,j]) for i,j in zip(c1.sig,c2.sig)]
            print("  Done generating compound-compound signatures.\n")
            print("Done building compound pair-adverse events tables.\n")

        if self.indication_proteins:
            print('Reading indication-gene associations...')
            with open(indication_proteins, 'r', encoding="utf8") as igf:
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
                            pro.indications.append(ind)
                        except KeyError:
                            pass
                        except LookupError:
                            pass
            print('Done reading indication-gene associations.')

        if read_dists:
            print('Reading {} distances...'.format(self.dist_metric))
            with open(read_dists, 'r', encoding="utf8") as rrs:
                lines = rrs.readlines()
                for i in range(len(lines)):
                    c1 = self.compounds[i]
                    scores = lines[i].strip().split('\t')
                    assert len(scores) == len(self.compounds), '    The number of compounds in {} does not match the '\
                              'number of values in {} -- quitting.'.format(self.c_map, self.matrix)
                    #if len(scores) != len(self.compounds):
                    #    print('    The number of compounds in {} does not match the '
                    #         'number of values in {} -- quitting.'.format(self.c_map, self.matrix))
                    #   quit()
                    for j in range(len(scores)):
                        if i == j:
                            continue
                        else:
                            s = float(scores[j])
                            if similarity:
                                s = 1 - s
                            #c1.similar.append((self.compounds[j], s))
                            c1.similar.append((j, s))

            for c in self.compounds:
                sorted_scores = sorted(c.similar, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
                c.similar = sorted_scores
                c.similar_computed = True
                c.similar_sorted = True
            print('Done reading {} distances.\n'.format(self.dist_metric))

        # if compute distance is true, generate similar compounds for each
        if compute_distance and not read_dists:
            if self.pathways and not self.indication_pathways and not ddi_adrs:
                print('Computing distances using global pathway signatures...')
                for c in self.compounds:
                    self.generate_similar_sigs(c, aux=True)
                    c.similar = [(c.id_,dist)for c,dist in c.similar]
           
            # Still cleaning this code up.
            # Memory issues with full Twosides is a huge limitation
            ## Do not compute all distances, but rather generate_simialr on the fly
            ## do not populate the similar list for each Compound_pair object
            ## This will increase computing time, but decrease mem allocation
            elif ddi_adrs:
                #print('Will not compute {} distances for compound pairs due to memory issues...'.format(self.dist_metric))
                #print('Can compute individually on-the-fly for canpredict and/or canbennchmark.')
                print('Computing {} distances for compound pairs...'.format(self.dist_metric))
               
                try: 
                    #Connecting to sqlite
                    #conn = sqlite3.connect('ddi.db')
                    conn = sqlite3.connect(self.db_name)
                    #Creating a cursor object using the cursor() method
                    cursor = conn.cursor()
                    #Droping dists table if already exists
                    cursor.execute("DROP TABLE dists")
                    #print("dists table dropped.")
                    #Commit your changes in the database
                    conn.commit()
                    #Closing the connection
                    conn.close()
                except:
                    #print("dists table does not exist.")
                    pass

                '''
                try: 
                    #Connecting to sqlite
                    conn = sqlite3.connect('ddi.db')
                    #Creating a cursor object using the cursor() method
                    cursor = conn.cursor()
                    #Droping dists table if already exists
                    cursor.execute("DROP TABLE ranks")
                    print("ranks table dropped.")
                    #Commit your changes in the database
                    conn.commit()
                    #Closing the connection
                    conn.close()
                except:
                    print("ranks table does not exist.")
                '''

                #db = create_engine('sqlite:///ddi.db')
                db = create_engine(f'sqlite:///{self.db_name}')
                # put all compound_pair signatures into 2D-array
                snp = [self.compound_pairs[i].sig for i in range(0, len(self.compound_pairs))]
                snp = np.array(snp)  # convert to numpy form
                distance_matrix = []
                # call pairwise_distances, speed up with custom RMSD function and parallelism
                if self.dist_metric == "rmsd":
                    distance_matrix = pairwise_distances_chunked(snp, metric='euclidean', n_jobs=self.ncpus)
                    #distance_matrix = squareform(distance_matrix)
                elif self.dist_metric in ['cosine', 'correlation', 'euclidean', 'cityblock']:
                    #distance_matrix = pairwise_distances(snp, metric=self.dist_metric, n_jobs=self.ncpus)
                    #distance_matrix = squareform(distance_matrix)
                    distance_matrix = pairwise_distances_chunked(snp, metric=self.dist_metric, 
                            force_all_finite=False,
                            n_jobs=self.ncpus)
                l = [cpp.id_ for cpp in self.compound_pairs]
                d_similar = {}
                #r_similar = {}
                i = 0
                for chunk in distance_matrix:
                    pbar = tqdm(chunk) if self.pbar else chunk
                    for y in pbar: #TQDM3
                        #dists = cdist([snp[i]], snp, dist_metric)[0]
                        #self.compound_pairs[i].similar = dict(zip(self.compound_pairs, dists))
                        #self.compound_pairs[i].similar.pop(i)
                        cp = self.compound_pairs[i].id_
                        #d_temp = dict(zip(l, y))
                        #d_temp.pop(cp)
                        #d_temp = dict(sorted(d_temp.items(), key=lambda item: item[1] if not math.isnan(item[1]) else 100000))
                        if self.dist_metric == 'rmsd':
                            d_temp = [(l[i], y[i]/(len(self.proteins)**0.5)) for i in range(len(y))]
                        else:
                            d_temp = list(zip(l, y))
                        d_temp.pop(i)
                        #l2 = l.copy()
                        #l2.pop(i)
                        #y = np.delete(y, i)
                        #d_temp = sorted(d_temp, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
                            
                        # Calculate ranks for each compound pair
                        #rank_sorted = stats.rankdata(y, method='max')
                        #rank_sorted = stats.rankdata(list(zip(*d_temp))[1], method='max')

                        # Save as a list
                        #d_temp = list(zip(l2,y,rank_sorted))
                        #d_temp = list(zip(list(zip(*d_temp))[0],list(zip(*d_temp))[1],rank_sorted))
                        # Save as a dict
                        #d_temp = {sctr(cp2): [dist2,rank2] for cp2,dist2,rank2 in d_temp}
                        d_temp = {str(cp2): dist2 for cp2,dist2 in d_temp}
                        #d_similar[str(cp)] = d_temp
                        d_similar[str(cp)] = str(d_temp)
                        if i%1000==0 or i==len(self.compound_pairs)-1:
                            df_temp = pd.DataFrame.from_dict(d_similar,orient='index')
                            df_temp.index.names = ['id']
                            df_temp.rename(columns={0:'dists'},inplace=True)
                            df_temp.to_sql('dists', db, if_exists='append')
                            d_similar = {}
                                
                            #df_temp = pd.DataFrame.from_dict(r_similar,orient='index')
                            #df_temp.index.names = ['id']
                            #df_temp.to_sql('ranks', db, if_exists='append')
                            #r_similar = {}
                        i+=1
                        #df_temp = pd.DataFrame.from_dict(d_similar, index=[str(self.compound_pairs[i].id_)], columns=['dists'])
                    #df_temp.to_sql('dists', db, if_exists='append')
                    #self.compound_pairs[i].similar_computed = True
                    #self.compound_pairs[i].similar_sorted = True
                print('Done computing {} distances for compound pairs.\n'.format(self.dist_metric))

            # Compute distances -- just compounds (not compound pairs)
            else:
                print('Building {} distance table...'.format(self.dist_metric))

                print("  Checking if data exists in table...")
                check = False
                try:
                    db = create_engine(f'sqlite:///{self.db_name}')
                    df_dists = pd.read_sql(f"SELECT * FROM {self.dist_metric}", db)
                    if len(df_dists) == len(self.compounds):
                        print("  Data already exists in table.")
                        check = True
                except:
                    pass
                if not check:
                    print("  Data does not exist in table OR data does not match.")
                    try:
                        conn = sqlite3.connect(f'{self.db_name}')
                        cursor = conn.cursor()
                        cursor.execute(f"DROP TABLE {self.dist_metric}")
                        conn.commit()
                        conn.close()
                    except:
                        pass

                    print('  Calculating {} distances...'.format(self.dist_metric))
                    db = create_engine(f'sqlite:///{self.db_name}')
                    # put all compound_pair signatures into 2D-array
                    snp = [self.compounds[i].sig for i in range(len(self.compounds))]
                    snp = np.array(snp)  # convert to numpy form

                    # Remove sigs from Compounds to reduce mem
                    #for i in range(len(self.compounds)):
                    #    self.compounds[i].sig = []

                    # call pairwise_distances, speed up with custom RMSD function and parallelism
                    distance_matrix = []
                    if self.dist_metric == "rmsd":
                        distance_matrix = pairwise_distances_chunked(snp, metric='euclidean',
                                                                     #working_memory=512,
                                                                     n_jobs=self.ncpus)
                    elif self.dist_metric in ['cosine', 'correlation', 'euclidean', 'cityblock']:
                        distance_matrix = pairwise_distances_chunked(snp, metric=self.dist_metric,
                                                                     force_all_finite=False,
                                                                     #working_memory=512,
                                                                     n_jobs=self.ncpus)
                    print(f'  Done calculating {self.dist_metric} distances.')

                    l = [c.id_ for c in self.compounds]
                    d_similar = {}
                    i = 0
                    for chunk in distance_matrix:
                        pbar = tqdm(chunk) if self.pbar else chunk
                        for y in pbar: ##TQDM1
                            c1 = str(self.compounds[i].id_)
                            if self.dist_metric == 'rmsd':
                                d_temp = [(l[i], y[i]/(len(self.proteins)**0.5)) for i in range(len(y))]
                            else:
                                d_temp = list(zip(l, y))
                            d_temp.pop(i)
                            # This is to keep the old versions of canbenchmark and canpredict working
                            # Need to update those functions to work with sqlite
                            self.compounds[i].similar = d_temp
                            self.compounds[i].similar_computed = True
                            d_temp = {str(c2): dist2 for c2, dist2 in d_temp}
                            d_similar[str(c1)] = str(d_temp)
                            if i % 1000 == 0 or i == len(self.compounds)-1:
                                df_temp = pd.DataFrame.from_dict(d_similar, orient='index')
                                df_temp.index.names = ['id']
                                df_temp.rename(columns={0: 'dists'}, inplace=True)
                                df_temp.to_sql(self.dist_metric, db, if_exists='append')
                                del df_temp
                                d_similar = {}
                            i += 1

                    # Speed the table up with indexes
                    conn = sqlite3.connect(f'{self.db_name}')
                    cursor = conn.cursor()
                    cursor.execute(f"CREATE INDEX {self.dist_metric}_id_idx ON {self.dist_metric}(id)")
                    conn.commit()
                    conn.close()
                    del distance_matrix, d_temp, snp, d_similar
                    
                    for c in self.compounds:
                        c_sorted = c.similar
                        c_sorted = sorted(c_sorted, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
                        c.similar = c_sorted
                        c.similar_sorted = True
                elif check:
                    #conn = f'sqlite://{self.db_name}'
                    with sqlite3.connect(f'{self.db_name}') as conn:
                        df_dists = pl.read_database(query=f"SELECT * FROM {self.dist_metric}",
                                                    connection=conn)
                    for c in self.compounds:
                        c_sorted = json.loads(df_dists.filter(pl.col('id') == str(c.id_)).select(['dists']).item().replace("'", '"'))
                        c_sorted = [(int(c),c_sorted[c]) for c in c_sorted.keys()]
                        c_sorted = sorted(c_sorted, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
                        c.similar = c_sorted
                        c.similar_computed = True
                        c.similar_sorted = True
                    del df_dists
                print(f'Done building {self.dist_metric} distance table.\n')


            if self.save_dists:
                def dists_to_str(cmpd, ci):
                    o = []
                    for si in range(len(cmpd.similar)):
                        if ci == si:
                            if self.similarity:
                                o.append('1.0')
                            else:
                                o.append('0.0')
                        s = cmpd.similar[si]
                        o.append(str(s[1]))
                    if len(o) < len(self.compounds):
                        o.append('0.0')
                    o = "\t".join(o)
                    o = o + '\n'
                    return o

                print('Saving {} distances...'.format(self.dist_metric))
                '''
                if adr_ddi:
                    with open(self.save_dists, 'w') as srf:
                        for ci in range(len(self.compound_pairs)):
                            c = self.compound_pairs[ci]
                            srf.write(dists_to_str(c, ci))
                else:
                    with open(self.save_dists, 'w') as srf:
                        for ci in range(len(self.compounds)):
                            c = self.compounds[ci]
                            srf.write(dists_to_str(c, ci))
                '''
                with open(self.save_dists, 'w', encoding="utf8") as srf:
                    for ci in range(len(self.compounds)):
                        c = self.compounds[ci]
                        srf.write(dists_to_str(c, ci))
                print('Done saving {} distances.\n'.format(self.dist_metric))

        if rm_compounds:
            print('Removing undesired compounds in {}...'.format(rm_compounds))
            with open(rm_compounds, 'r', encoding="utf8") as rcf:
                self.rm_cmpds = [int(line.strip().split('\t')[0]) for line in rcf]
            self.compounds = [c for c in self.compounds if c.id_ not in self.rm_cmpds]
            for c in self.compounds:
                c.similar = [s for s in c.similar if s[0].id_ not in self.rm_cmpds]
                c.compounds = [s for s in c.compounds if s.id_ not in self.rm_cmpds]
            if self.matrix:
                for p in self.proteins:
                    p.sig = [y for x, y in enumerate(p.sig) if x not in self.rm_cmpds]
            print('Done removing undesired compounds.\n')

            # sort the RMSDs after saving (if desired)
            for c in self.compounds:
                sorted_scores = sorted(c.similar, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
                c.similar = sorted_scores
                c.similar_computed = True
                c.similar_sorted = True

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
            print('Done removing compounds with all-zero signatures.\n')

        if self.rm_zeros or self.rm_compounds:
            print('Filtering indication mapping...')
            for ind in self.indications:
                ind.compounds = [cmpd for cmpd in ind.compounds if cmpd.id_ not in self.rm_cmpds]
            print('Done filtering indication mapping.\n')

        if compute_distance and not read_dists and (rm_compounds or rm_zeros):
            if self.pathways and not self.indication_pathways:
                print('Recomputing distances using global pathway signatures...')
                for c in self.compounds:
                    self.generate_similar_sigs(c, aux=True)
            else:
                print('Recomputing {} distances...'.format(self.dist_metric))
                # put all compound signatures into 2D-array
                signatures = []
                for i in range(0, len(self.compounds)):
                    signatures.append(self.compounds[i].sig)
                snp = np.array(signatures)  # convert to numpy form
                # call pairwise_distances, speed up with custom RMSD function and parallelism
                if self.dist_metric == "rmsd":
                    distance_matrix = pairwise_distances(snp, metric=lambda u, v: np.sqrt(np.mean((u - v)**2)),
                                                         n_jobs=self.ncpus)
                    distance_matrix = squareform(distance_matrix)
                elif self.dist_metric in ['correlation', 'euclidean', 'cityblock', 'cosine']:
                    #distance_matrix = pairwise_distances_chunked(snp, metric=self.dist_metric, 
                    #        force_all_finite=False,
                    #        n_jobs=self.ncpus)
                    #distance_matrix = np.concatenate(list(distance_matrix), axis=0)                    
                    distance_matrix = pairwise_distances(snp, metric=self.dist_metric, force_all_finite=False,
                                                         n_jobs=self.ncpus)
                    # Removed checks in case the diagonal is very small (close to zero) but not zero.
                    distance_matrix = squareform(distance_matrix, checks=False)
                else:
                    print("Incorrect distance metric - {}".format(self.dist_metric))
                    exit()

                # step through the condensed matrix - add RMSDs to Compound.similar lists
                nc = len(self.compounds)
                n = 0
                for i in range(nc):
                    for j in range(i, nc):
                        c1 = self.compounds[i]
                        c2 = self.compounds[j]
                        if i == j:
                            continue
                        r = distance_matrix[n]
                        c1.similar.append((c2, r))
                        c2.similar.append((c1, r))
                        n += 1
                print('Done recomputing {} distances.\n'.format(self.dist_metric))

        if adr_map:
            print('Reading ADR mapping file...')
            with open(adr_map, 'r', encoding="utf8") as amf:
                lines = amf.readlines()
                header = lines[0]
                h2i = {}
                for i, h in enumerate(header.strip().split('\t')):
                    h2i[h] = i
                prev_id = -1
                lcount = 0
                for l in lines[1:]:
                    ls = l.strip().split('\t')
                    adr_name = ls[h2i['CONDITION_MESH_NAME']]
                    adr_id = ls[h2i['CONDITION_MESH_ID']]
                    c_id = int(ls[h2i['CANDO_ID']])
                    #adr_name = ls[h2i['condition_concept_name']]
                    #c_id = int(ls[h2i['drug_cando_id']])
                    #adr_id = ls[h2i['condition_meddra_id']]

                    if c_id == -1:
                        continue
                    if prev_id == c_id:
                        pass
                    else:
                        cmpd = self.get_compound(c_id, quiet=True)
                        if cmpd is not None:
                            prev_id = c_id
                        else:
                            # cmpd is not in CANDO - prevents from crashing
                            continue
                    try:
                        adr = self.get_adr(adr_id)
                        if adr_id == adr.id_:
                            cmpd.adrs.append(adr.id_)
                            adr.compounds.append(c_id)
                    except LookupError:
                        adr = ADR(adr_id, adr_name)
                        adr.compounds.append(c_id)
                        cmpd.adrs.append(adr_id)
                        self.adrs.append(adr)
            print('Read {} ADRs.'.format(len(self.adrs)))

        if protein_map:
            print('Reading Protein mapping file...')
            prot_df = pd.read_csv(protein_map,sep='\t',index_col=0)
            for i in prot_df.index:
                p = self.get_protein(i)
                p.name = prot_df['uniprotRecommendedName'][i]
                p.gene = prot_df['geneName'][i]
                p.method = prot_df['method'][i]

    def search_compound(self, name, n=5):
        """!
        Print closest Compound names/IDs for input search str

        @param name str: Compound name
        @param n int: Number of outputted compounds
        @return Returns None
        """
        id_d = {}

        def return_names(x):
            id_d[x.name] = x.id_
            return x.name

        name = name.strip().lower().replace(' ', '_')
        cando_drugs = list(map(return_names, self.compounds))
        nms = difflib.get_close_matches(name, cando_drugs, n=n, cutoff=0.5)
        print('id\tname')
        for nm in nms:
            print("{}\t{}".format(id_d[nm], nm))

    def get_compound(self, cmpd_id, quiet=False):
        """!
        Get Compound object from Compound id or fuzzy match to Compound name

        @param cmpd_id int or str: Compound id or Compound name
        @return Returns object: Compound object or None if no exact match is found
        """
        if type(cmpd_id) is int:
            for c in self.compounds:
                if c.id_ == cmpd_id:
                    return c
            if not quiet:
                print("{0} not in {1}".format(cmpd_id, self.c_map))
            return None
        elif type(cmpd_id) is str:
            id_d = {}

            def return_names(x):
                id_d[x.name] = x.id_
                return x.name

            cando_drugs = list(map(return_names, self.compounds))
            name = cmpd_id.strip().lower().replace(' ', '_')
            if name not in cando_drugs:
                print('"{}" is not in our mapping, here are the 5 closest results:'.format(name))
                self.search_compound(name, n=5)
                return None
            else:
                return self.get_compound(id_d[name])

    def get_compound_pair(self, ids):
        """!
        Get Compound_pair object from Compound_pair id

        @param id_ int: Compound_pair id
        @return Returns object: Compound_pair object
        """
        for c in self.compound_pairs:
            if c.id_ == ids:
                return c
            elif c.id_ == (ids[1],ids[0]):
                return c
        print("{0} not in {1}".format(ids, self.ddi_adrs))
        return None

    def get_protein(self, protein_id):
        """!
        Get Protein object from Protein id

        @param protein_id str: Protein name
        @return Returns object: Protein object
        """
        assert len(self.proteins) > 0 and self.matrix, 'No matrix/proteins loaded'
        #if len(self.proteins) == 0 or not self.matrix:
        #    print('No matrix/proteins loaded -- quitting.')
        #    quit()
        for p in self.proteins:
            if p.id_ == protein_id:
                return p

    def get_indication(self, ind_id):
        """!
        Get Indication object from Indication id

        @param ind_id str: Indication id
        @return Returns object: Indication object
        """
        for i in self.indications:
            if i.id_ == ind_id:
                return i
        print('{} not in {}'.format(ind_id, self.i_map))
        raise LookupError

    def get_pathway(self, id_):
        """!
        Get Pathway object from Pathway id

        @param id_ str: Pathway id
        @return Returns object: Pathway object
        """
        for p in self.pathways:
            if p.id_ == id_:
                return p
        raise LookupError

    def get_adr(self, id_):
        """!
        Get ADR (adverse drug reaction) from ADR id
        
        @param id_ str: ADR id
        @return Returns object: ADR object
        """
        for a in self.adrs:
            if a.id_ == id_:
                return a
        raise LookupError

    def search_indication(self, name, n=5):
        """!
        Print closest MeSH IDs for Indication name

        @param name str: Indication name
        @param n int: Number of outputted indications
        @return Returns None
        """
        id_d = {}

        def return_names(x):
            id_d[x.name] = x.id_
            return x.name

        name = name.strip()
        cando_inds = list(map(return_names, self.indications))
        exact_matches = []
        for ci in cando_inds:
            if name in ci:
                exact_matches.append(ci)
        if exact_matches:
            print('Matches exactly containing {}:'.format(name))
            print('id             \tname')
            for em in exact_matches:
                print("{}\t{}".format(id_d[em], em))
            print()
        nms = difflib.get_close_matches(name, cando_inds, n=n, cutoff=0.3)
        print('Matches using string distance:')
        print('id             \tname')
        for nm in nms:
            print("{}\t{}".format(id_d[nm], nm))

    def top_targets(self, cmpd, n=10, negative=False, save_file=''):
        """!
        Get the top scoring protein targets for a given compound

        @param cmpd Compound or int: Compound object or int id_ for which to print targets
        @param n int: number of top targets to print/return
        @param negative int: if the interaction scores are negative (stronger) energies
        @param save_file str: output file for results
        @return Returns list: list of tuples (protein id_, score)
        """
        assert type(cmpd) is Compound or type(cmpd) is int, 'Please enter a Compound object or integer id_ for a compound'
        # print the list of the top targets
        if type(cmpd) is Compound:
            pass
        elif type(cmpd) is int:
            cmpd = self.get_compound(cmpd)
        #else:
        #    print('Please enter a Compound object or integer id_ for a compound -- quitting.')
        #    quit()
        all_interactions = []
        sig = cmpd.sig
        for i in range(len(sig)):
            s = sig[i]
            p = self.proteins[i]
            all_interactions.append((p, s))
        if negative:
            interactions_sorted = sorted(all_interactions, key=lambda x: x[1])
        else:
            interactions_sorted = sorted(all_interactions, key=lambda x: x[1])[::-1]
        if save_file:
            o = open(save_file,'w', encoding="utf8")
            o.write('rank\tscore\tindex\tid\tgene\tname\n')
        print('Compound is {}'.format(cmpd.name))
        print('rank\tscore\tindex\tid\tgene\tname')
        for si in range(n):
            pr = interactions_sorted[si][0]
            print('{}\t{}\t{}\t{}\t{}\t{}'.format(si+1, round(interactions_sorted[si][1], 3),
                                          self.proteins.index(pr), pr.id_, pr.gene, pr.name))
            if save_file:
                o.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(si+1, round(interactions_sorted[si][1], 3),
                                                self.proteins.index(pr), pr.id_, pr.gene, pr.name))
        print()
        if save_file:
            o.close()
        return interactions_sorted[0:n]

    def common_targets(self, cmpds_file, n=10, negative=False, save_file=''):
        """!
        Get the consensus top scoring protein targets for a set of compounds

        @param cmpds_file str: File containing a list of Compound IDs for which to search common targets
        @param n int: number of top targets to print/return
        @param negative int: if the interaction scores are negative (stronger) energies
        @param save_file str: save results to file name
        @return Returns list: list of tuples (protein id_, score)
        """
        cs_df = pd.read_csv(cmpds_file,sep='\t',header=None)
        sum_sig = [0]*len(self.get_compound(0).sig)
        for ci in cs_df.itertuples(index=False):
            try:
                s = self.get_compound(int(ci[0])).sig
            except:
                print("{} does not exist in the current drug library.\n".format(ci[0]))
                continue
            sum_sig = [i+j for i,j in zip(sum_sig,s)]
        # print the list of the top targets
        all_interactions = []
        for i in range(len(sum_sig)):
            s = sum_sig[i]
            p = self.proteins[i]
            all_interactions.append((p, s))
        if negative:
            interactions_sorted = sorted(all_interactions, key=lambda x: x[1])
        else:
            interactions_sorted = sorted(all_interactions, key=lambda x: x[1])[::-1]
        if save_file:
            o = open(save_file,'w', encoding="utf8")
            o.write('rank\tscore\tindex\tid\tgene\tname\n')
        print('rank\tscore\tindex\tid\tgene\tname')
        for si in range(n):
            pr = interactions_sorted[si][0]
            print('{}\t{}\t{}\t{}\t{}\t{}'.format(si+1, round(interactions_sorted[si][1], 3),
                                          self.proteins.index(pr), pr.id_, pr.gene, pr.name))
            if save_file:
                o.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(si+1, round(interactions_sorted[si][1], 3),
                                                self.proteins.index(pr), pr.id_, pr.gene, pr.name))
        print()
        if save_file:
            o.close()
        return interactions_sorted[0:n]

    def virtual_screen(self, protein, n=10, negative=False, compound_set='all', save_file=''):
        """!
        Get the top scoring compounds for a given protein

        @param protein Protein int or str: Protein (object, int index, or str id_) of which to screen for top scores
        @param n int: number of top compounds to print/return
        @param negative int: if the interaction scores are negative (stronger) energies
        @param compound_set str: use all Compounds ('all') or only approved Compounds ('approved')
        @param save_file str: save results to file name
        @return Returns None
        """
        if type(protein) is Protein:
            prot = protein
        elif type(protein) is int:
            prot = self.proteins[protein]
        elif type(protein) is str:
            for p in self.proteins:
                if p.id_ == protein:
                    prot = p

        # print the list of the top targets
        all_interactions = []
        sig = prot.sig
        for i in range(len(sig)):
            s = sig[i]
            c_id = self.compounds[i].id_
            #if c_id in self.rm_cmpds:
            #    continue
            all_interactions.append((c_id, s))
        if negative:
            interactions_sorted = sorted(all_interactions, key=lambda x: x[1])
        else:
            interactions_sorted = sorted(all_interactions, key=lambda x: x[1])[::-1]
        print('Protein is {}'.format(prot.id_))
        if save_file:
            o = open(save_file,'w', encoding="utf8")
            o.write('rank\tscore\tid\tapproved\tname\n')
        print('rank\tscore\tid\tapproved\tname')
        printed = 0
        si = 0
        while printed < n:
            c = self.get_compound(interactions_sorted[si][0])
            #c = self.compounds[interactions_sorted[si][0]]
            if compound_set == 'approved':
                if c.status == 'approved':
                    print('{}\t{}\t{}\t{}    \t{}'.format(printed+1, round(interactions_sorted[si][1], 3), c.id_,
                                                          'true', c.name))
                    if save_file:
                        o.write('{}\t{}\t{}\t{}\t{}\n'.format(printed+1, round(interactions_sorted[si][1], 3), c.id_,
                                                                'true', c.name))
                    printed += 1
            else:
                print('{}\t{}\t{}\t{}    \t{}'.format(printed+1, round(interactions_sorted[si][1], 3),
                                                      c.id_, str(c.status == 'approved').lower(),
                                                      c.name))
                if save_file:
                    o.write('{}\t{}\t{}\t{}\t{}\n'.format(printed+1, round(interactions_sorted[si][1], 3),
                                                            c.id_, str(c.status == 'approved').lower(),
                                                            c.name))
                printed += 1
            si += 1
        print()
        if save_file:
            o.close()
        return

    def uniprot_set_index(self, prots):
        """!
        Gather proteins from input matrix that map to UniProt IDs from 'protein_set=' param

        @param prots list: UniProt IDs (str)
        @return Returns list: Protein chains (str) matching input UniProt IDs
        """
        pre = os.path.dirname(__file__) + "/data/v2.2+/"
        if not os.path.exists('{}/mappings/pdb_2_uniprot.csv'.format(pre)):
            print('Downloading UniProt to PDB mapping file...')
            url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/mappings/pdb_2_uniprot.csv'
            dl_file(url, '{}/mappings/pdb_2_uniprot.csv'.format(pre))
        pdct = {}
        pdct_rev = {}
        with open('{}/mappings/pdb_2_uniprot.csv'.format(pre), 'r', encoding="utf8") as u2p:
            for l in u2p.readlines()[1:]:
                spl = l.strip().split(',')
                pdb = spl[0] + spl[1]
                uni = spl[2]
                try:
                    if pdb not in pdct[uni]:
                        pdct[uni].append(pdb)
                except KeyError:
                    pdct[uni] = [pdb]
                pdct_rev[pdb] = uni
        targets = []
        with open(prots, 'r', encoding="utf8") as unisf:
            for lp in unisf:
                prot = lp.strip()
                targets.append(prot)
                #pdct_rev[prot] = lp.strip().upper()
                try:
                    targets += pdct[lp.strip().upper()]
                except KeyError:
                    pass
        return targets, pdct_rev

    def generate_similar_sigs(self, cmpd, sort=False, proteins=[], aux=False):
        """!
        For a given compound, generate the similar compounds using distance of sigs.

        @param cmpd object: Compound object
        @param sort bool: Sort the list of similar compounds
        @param proteins list: Protein objects to identify a subset of the Compound signature
        @param aux bool: Use an auxiliary signature (default: False)

        @return Returns list: Similar Compounds to the given Compound
        """
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
        c_sig = [float(score) for score in c_sig]
        ca = np.array([c_sig])

        other_sigs = []
        for ci in range(len(self.compounds)):
            c = self.compounds[ci]
            if cmpd.id_ == c.id_:
                q = ci
            other = []
            if proteins is None:
                c_sig = [float(score) for score in c.sig]
                other_sigs.append(c_sig)
            elif proteins:
                for pro in proteins:
                    index = self.protein_id_to_index[pro.id_]
                    other.append(c.sig[index])
                    other = [float(score) for score in other]
                other_sigs.append(other)
            else:
                if aux:
                    c_sig = [float(score) for score in c.aux_sig]
                    other_sigs.append(c_sig)
                else:
                    c_sig = [float(score) for score in c.sig]
                    other_sigs.append(c_sig)
        oa = np.array(other_sigs)
        
        # call pairwise_distance to enable parallel computing
        # custom RMSD function
        if self.dist_metric == "rmsd":
            distances = pairwise_distances(ca, oa, lambda u, v: np.sqrt(np.mean((u - v) ** 2)), n_jobs=self.ncpus)
        elif self.dist_metric in ['cosine', 'correlation', 'euclidean', 'cityblock']:
            distances = pairwise_distances(ca, oa, self.dist_metric, force_all_finite=False, n_jobs=self.ncpus)
        else:
            print("Incorrect distance metric - {}".format(self.dist_metric))

        cmpd.similar = []
        # step through the cdist list - add dists to Compound.similar list
        n = len(self.compounds)
        for i in range(n):
            c2 = self.compounds[i]
            if i == q:
                continue
            d = int(distances[0][i])
            cmpd.similar.append((c2, d))
            n += 1

        if sort:
            sorted_scores = sorted(cmpd.similar, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
            cmpd.similar = sorted_scores
            cmpd.similar_computed = True
            cmpd.similar_sorted = True
            return sorted_scores
        else:
            cmpd.similar_computed = True
            return cmpd.similar

    def generate_similar_sigs_cp(self, cmpd_pair, sort=False, proteins=[], aux=False, ncpus=1):
        """!
        For a given compound pair, generate the similar compound pairs using distance of sigs.

        @param cmpd_pair object: Compound_pair object
        @param sort bool: Sort the list of similar compounds
        @param proteins list: Protein objects to identify a subset of the Compound signature
        @param aux bool: Use an auxiliary signature (default: False)

        @return Returns list: Similar Compounds to the given Compound
        """
        # find index of query compound, collect signatures for both
        try:
            q = self.compound_pair_ids.index(cmpd_pair.id_)
        except:
            q = None

        cp_sig = cmpd_pair.sig
        ca = np.array([cp_sig])

        other_sigs = [cp.sig for cp in self.compound_pairs]
        oa = np.array(other_sigs)
        
        # call cdist, speed up with custom RMSD function
        if self.dist_metric == "rmsd":
            distances = pairwise_distances(ca, oa, lambda u, v: np.sqrt(np.mean((u - v) ** 2)), n_jobs=ncpus)
        elif self.dist_metric in ['cosine', 'correlation', 'euclidean', 'cityblock']:
            distances = pairwise_distances(ca, oa, self.dist_metric, n_jobs=ncpus)
        else:
            print("Incorrect distance metric - {}".format(self.dist_metric))

        # DO NOT populate the similar list in the Compound_pair object.
        # This will cause memory issues with large sets of compounds (TWOSIDES)
        # Output just the list of similar compounds
        '''
        cmpd_pair.similar = []
        # step through the cdist list - add RMSDs to Compound.similar list
        n = len(self.compound_pairs)
        for i in range(n):
            c2 = self.compound_pairs[i]
            if i == q:
                continue
            d = distances[0][i]
            cmpd_pair.similar.append((c2, d))
            n += 1

        if sort:
            sorted_scores = sorted(cmpd_pair.similar, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
            cmpd_pair.similar = sorted_scores
            cmpd_pair.similar_computed = True
            cmpd_pair.similar_sorted = True
            return sorted_scores
        else:
            cmpd_pair.similar_computed = True
            return cmpd_pair.similar
        '''
        cps = self.compound_pairs
        scores = list(zip(cps, distances[0]))
        if q:
            del scores[q]

        if sort:
            sorted_scores = sorted(scores, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
            return sorted_scores
        else:
            return scores


    def generate_similar_sigs_cps(self, cmpd_pairs, sort=False, proteins=[], aux=False, ncpus=1):
        """!
        For a given compound pair, generate the similar compound pairs using distance of sigs.

        @param cmpd_pair object: Compound_pair object
        @param sort bool: Sort the list of similar compounds
        @param proteins list: Protein objects to identify a subset of the Compound signature
        @param aux bool: Use an auxiliary signature (default: False)

        @return Returns list: Similar Compounds to the given Compound
        """
        # find index of query compound, collect signatures for both
        q = None
        cp_sigs = []
        for cmpd_pair in cmpd_pairs:
            cp_sig = cmpd_pair.sig
            cp_sigs.append(cp_sig)
        ca = np.array(cp_sigs)

        other_sigs = []
        for ci in range(len(self.compound_pairs)):
            cp = self.compound_pairs[ci]
            if cmpd_pair.id_ == cp.id_:
                q = ci
            other_sigs.append(cp.sig)
        oa = np.array(other_sigs)
        
        # call cdist, speed up with custom RMSD function
        if self.dist_metric == "rmsd":
            distances = pairwise_distances(ca, oa, lambda u, v: np.sqrt(np.mean((u - v) ** 2)), n_jobs=ncpus)
        elif self.dist_metric in ['cosine', 'correlation', 'euclidean', 'cityblock']:
            distances = pairwise_distances(ca, oa, self.dist_metric, n_jobs=ncpus)
        else:
            print("Incorrect distance metric - {}".format(self.dist_metric))

        cps = self.compound_pairs
        scores = list(zip(cps, distances[0]))
        if q:
            del scores[q]

        if sort:
            sorted_scores = [sorted(y, key=lambda x: x[1] if not math.isnan(x[1]) else 100000) for y in scores]
            #sorted_scores = sorted(scores, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
            return sorted_scores
        else:
            return scores


    def generate_some_similar_sigs(self, cmpds, sort=False, proteins=[], aux=False):
        """!
        For a given list of compounds, generate the similar compounds based on dist of sigs
        This is pathways/genes for all intents and purposes

        @param cmpds list: Compound objects
        @param sort bool: Sort similar compounds for each Compound
        @param proteins list: Protein objects to identify a subset of the Compound signature
        @param aux bool: Use an auxiliary signature (default: False)

        @return Returns list: Similar Compounds to the given Compound

        """
        #q = [cmpd.id_ for cmpd in cmpds]
        q = cmpds
        cmpds = [self.get_compound(c) for c in cmpds]

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
        
        # call cdist, speed up with custom RMSD function
        if self.dist_metric == "rmsd":
            distances = pairwise_distances(ca, oa, lambda u, v: np.sqrt(np.mean((u - v) ** 2)), n_jobs=self.ncpus)
        elif self.dist_metric in ['cosine', 'correlation', 'euclidean', 'cityblock']:
            distances = pairwise_distances(ca, oa, self.dist_metric, n_jobs=self.ncpus)
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
                cmpds[j].similar = [(c.id_,dist) for c,dist in sorted_scores]
                cmpds[j].similar_computed = True
                cmpds[j].similar_sorted = True
            else:
                cmpds[j].similar_computed = True

    def quantify_pathways(self, indication=None):
        """!
        Uses the pathway quantifier defined in the CANDO instantiation to make a
        pathway signature for all pathways in the input file (NOTE: does not compute distances)

        @param indication object: Indication object
        @return Returns None
        """
        pq = self.pathway_quantifier
        assert pq in ('max', 'sum', 'avg', 'proteins'), 'Please enter a proper pathway quantify method'
        if pq == 'max':
            func = max
        elif pq == 'sum':
            func = sum
        elif pq == 'avg':
            func = np.average
        elif pq == 'proteins':
            assert self.indication_pathways, 'Pathway quantifier "proteins" should only be used in combination with a '\
                      'pathway-disease mapping (indication_pathways).'
            #if not self.indication_pathways:
            #    print('Pathway quantifier "proteins" should only be used in combination with a '
            #          'pathway-disease mapping (indication_pathways), quitting.')
            #    quit()
            func = None
        #else:
        #    print('Please enter a proper pathway quantify method, quitting.')
        #    func = None
        #    quit()

        # this is a recursive function for checking if the pathways have proteins
        def check_proteins(paths):
            pl = []  # list of pathways with >1 protein
            n = 0
            for path in paths:
                if len(path.proteins) > 0:
                    pl.append(path)
                    n += 1
            if n > 0:
                return pl
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

    def results_analysed(self, f, metrics, effect_type):
        """!
        Creates the results analysed named file for the benchmarking and
        computes final avg indication accuracies

        @param f str: File path for results analysed named
        @param metrics list: Cutoffs used for the benchmarking protocol
        @param effect_type str: Defines the effect as either an Indication (disease) or ADR (adverse reaction)
        @return Returns dct: dict of accuracies at each cutoff
        """
        fo = open(f, 'w', encoding="utf8")
        effects = list(self.accuracies.keys())
        # Write header
        fo.write("{0}_id\tcmpds_per_{0}\ttop10\ttop25\ttop50\ttop100\ttopAll\ttop1%\t"
                 "top5%\ttop10%\ttop50%\ttop100%\t{0}_name\n".format(effect_type))
        effects_sorted = sorted(effects, key=lambda x: (len(x[0].compounds), x[0].id_))[::-1]
        l = len(effects)
        final_accs = {}
        for m in metrics:
            final_accs[m] = 0.0
        for effect, c in effects_sorted:
            fo.write("{0}\t{1}\t".format(effect.id_, c))
            accs = self.accuracies[(effect, c)]
            for m in metrics:
                n = accs[m]
                y = str(n / c * 100)[0:4]
                fo.write("{}\t".format(y))

                final_accs[m] += n / c / l
            fo.write("{}\n".format(effect.name))
        fo.close()
        return final_accs

    def canbenchmark(self, file_name, indications=[], continuous=False, bottom=False,
                     ranking='standard', adrs=False):
        """!
        Benchmarks the platform based on compound similarity of those approved for the same diseases

        @param file_name str: Name to be used for the various results files (e.g. file_name=test --> summary_test.tsv)
        @param indications list or str: List of Indication ids to be benchmarked, otherwise all will be used.
        @param continuous bool: Use the percentile of distances from the similarity matrix as the benchmarking cutoffs
        @param bottom bool: Reverse the ranking (descending) for the benchmark
        @param ranking str: What ranking method to use for the compounds. This really only affects ties. (standard,
        modified, and ordinal)
        @param adrs bool: ADRs are used as the Compounds' phenotypic effects instead of Indications
        @return Returns None
        """
        assert not continuous or not (self.indication_pathways or self.indication_proteins),\
               'Continuous benchmarking and indication-based signatures are not compatible'
               
        #if (continuous and self.indication_pathways) or (continuous and self.indication_proteins):
        #    print('Continuous benchmarking and indication-based signatures are not compatible, quitting.')
        #    exit()

        if not self.indication_proteins and not self.indication_pathways:
            if not self.compounds[0].similar_sorted:
                for c in self.compounds:
                    sorted_scores = sorted(c.similar, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
                    c.similar = sorted_scores
                    c.similar_sorted = True

        if not os.path.exists('./results_analysed_named'):
            print("Directory 'results_analysed_named' does not exist, creating directory")
            os.mkdir('results_analysed_named')
            #os.system('mkdir results_analysed_named')
        if not os.path.exists('./raw_results'):
            print("Directory 'raw_results' does not exist, creating directory")
            os.mkdir('raw_results')
            #os.system('mkdir raw_results')

        ra_named = 'results_analysed_named/results_analysed_named-' + file_name + '.tsv'
        ra = 'raw_results/raw_results-' + file_name + '.csv'
        summ = 'summary-' + file_name + '.tsv'
        ra_out = open(ra, 'w', encoding="utf8")

        def effect_type():
            if adrs:
                return 'ADR'
            else:
                return 'disease'

        def competitive_standard_bottom(sims, r):
            rank = 0
            for sim in sims:
                if sim[1] > r:
                    rank += 1.0
                else:
                    return rank
            return len(sims)

        def competitive_modified_bottom(sims, r):
            rank = 0
            for sim in sims:
                if sim[1] >= r:
                    rank += 1.0
                else:
                    return rank
            return len(sims)

        # Competitive modified ranking code
        def competitive_modified(sims, r):
            rank = 0
            for sim in sims:
                if sim[1] <= r:
                    rank += 1.0
                else:
                    return rank
            return len(sims)

        # Competitive standard ranking code
        def competitive_standard(sims, r):
            rank = 0
            for sim in sims:
                if sim[1] < r:
                    rank += 1.0
                else:
                    return rank
            return len(sims)

        def filter_indications(ind_set):
            if not os.path.exists('v2.0/mappings/group_disease-top_level.tsv'):
                url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2/mappings/group_disease-top_level.tsv'
                dl_file(url, 'v2.0/mappings/group_disease-top_level.tsv')
            path_ids = ['C01', 'C02', 'C03']
            with open('v2.0/mappings/group_disease-top_level.tsv', 'r', encoding="utf8") as fgd:
                for l in fgd:
                    ls = l.strip().split('\t')
                    if ls[1] in path_ids:
                        ind = self.get_indication(ls[0])
                        ind.pathogen = True
            assert ind_set in ('pathogen', 'human'), 'Please enter proper indication set; options include "pathogen" or "human"'
            if ind_set == 'pathogen':
                return [indx for indx in self.indications if indx.pathogen]
            elif ind_set == 'human':
                return [indx for indx in self.indications if not indx.pathogen]
            #else:
            #    print('Please enter proper indication set, options include "pathogen", "human", or "all".')
            #    quit()

        effect_dct = {}
        ss = []
        c_per_effect = 0

        if isinstance(indications, list) and len(indications) >= 1:
            effects = list(map(self.get_indication, indications))
        elif isinstance(indications, list) and len(indications) == 0 and not adrs:
            effects = self.indications
        elif adrs:
            effects = self.adrs
        else:
            if isinstance(indications, str):
                if indications == 'all':
                    effects = self.indications
                else:
                    effects = filter_indications(indications)

        def cont_metrics():
            all_v = []
            for c in self.compounds:
                for s in c.similar:
                    if s[1] != 0.0:
                        all_v.append(s[1])
            avl = len(all_v)
            all_v_sort = sorted(all_v)
            # for tuple 10, have to add the '-1' for index out of range reasons
            metrics = [(1, all_v_sort[int(avl/1000.0)]), (2, all_v_sort[int(avl/400.0)]), (3, all_v_sort[int(avl/200.0)]),
                       (4, all_v_sort[int(avl/100.0)]), (5, all_v_sort[int(avl/20.0)]), (6, all_v_sort[int(avl/10.0)]),
                       (7, all_v_sort[int(avl/5.0)]), (8, all_v_sort[int(avl/3.0)]), (9, all_v_sort[int(avl/2.0)]),
                       (10, all_v_sort[int(avl/1.0)-1])]
            return metrics

        x = (len(self.compounds)) / 100.0  # changed this...no reason to use similar instead of compounds
        # had to change from 100.0 to 100.0001 because the int function
        # would chop off an additional value of 1 for some reason...
        if continuous:
            metrics = cont_metrics()
        else:
            metrics = [(1, 10), (2, 25), (3, 50), (4, 100), (5, int(x*100.0001)),
                       (6, int(x*1.0001)), (7, int(x*5.0001)), (8, int(x*10.0001)),
                       (9, int(x*50.0001)), (10, int(x*100.0001))]

        if continuous:
            ra_out.write("compound_id,{}_id,0.1%({:.3f}),0.25%({:.3f}),0.5%({:.3f}),"
                         "1%({:.3f}),5%({:.3f}),10%({:.3f}),20%({:.3f}),33%({:.3f}),"
                         "50%({:.3f}),100%({:.3f}),value\n".format(effect_type(), metrics[0][1], metrics[1][1],
                                                                     metrics[2][1], metrics[3][1], metrics[4][1],
                                                                     metrics[5][1], metrics[6][1], metrics[7][1],
                                                                     metrics[8][1], metrics[9][1]))
        else:
            ra_out.write("compound_id,{}_id,top10,top25,top50,top100,"
                         "topAll,top1%,top5%,top10%,top50%,top100%,rank\n".format(effect_type()))

        pbar = tqdm(effects) if self.pbar else effects
        for effect in pbar:
            count = len(effect.compounds)
            if count < 2:
                continue
            if not adrs:
                if self.indication_pathways:
                    if len(effect.pathways) == 0:
                        print('No associated pathways for {}, skipping'.format(effect.id_))
                        continue
                    elif len(effect.pathways) < 1:
                        #print('Less than 5 associated pathways for {}, skipping'.format(effect.id_))
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

            # Retrieve the appropriate protein indices here, should be
            # incorporated as part of the ind object during file reading
            if self.indication_proteins:
                dg = []
                for p in effect.proteins:
                    if p not in dg:
                        dg.append(p)

            cmpds = effect.compounds
            if self.pathways:
                if self.indication_pathways:
                    if self.pathway_quantifier == 'proteins':
                        if not vs:
                            print('Warning: protein list empty for {}, using all proteins'.format(effect.id_))
                            self.generate_some_similar_sigs(cmpds, sort=True, proteins=None, aux=True)
                        else:
                            self.generate_some_similar_sigs(cmpds, sort=True, proteins=vs, aux=True)
                    else:
                        self.generate_some_similar_sigs(cmpds, sort=True, aux=True)
            elif self.indication_proteins:
                if len(dg) < 2:
                    self.generate_some_similar_sigs(cmpds, sort=True, proteins=None)
                else:
                    self.generate_some_similar_sigs(cmpds, sort=True, proteins=dg)
            # call c.generate_similar_sigs()
            # use the proteins/pathways specified above

            for cmpd in effect.compounds:
                c = self.get_compound(cmpd)
                for cmpd2 in c.similar:
                    cs = self.get_compound(cmpd2[0])
                    if adrs:
                        #if effect in cs.adrs:
                        if cs.id_ in effect.compounds:
                            cs_dist = cmpd2[1]
                        else:
                            continue
                    else:
                        #if effect in cs.indications:
                        if cs.id_ in effect.compounds:
                            cs_dist = cmpd2[1]
                        else:
                            continue

                    value = 0.0 
                    if continuous:
                        value = cs_dist
                    elif bottom:
                        if ranking == 'modified':
                            value = competitive_modified_bottom(c.similar, cs_dist)
                        elif ranking == 'standard':
                            value = competitive_standard_bottom(c.similar, cs_dist)
                        elif ranking == 'ordinal':
                            value = list(zip(*c.similar))[0].index(cs.id_)
                        else:
                            print("Ranking function {} is incorrect.".format(ranking))
                            exit()
                    elif ranking == 'modified':
                        value = competitive_modified(c.similar, cs_dist)
                    elif ranking == 'standard':
                        value = competitive_standard(c.similar, cs_dist)
                    elif ranking == 'ordinal':
                        value = c.similar.index(cs.id_)
                    else:    
                        print("Ranking function {} is incorrect.".format(ranking))
                        exit()
                   
                    if adrs:
                        s = [str(c.index), effect.name]
                    else:
                        s = [str(c.index), effect.id_]
                    for x in metrics:
                        if value <= x[1]:
                            effect_dct[(effect, count)][x] += 1.0
                            s.append('1')
                        else:
                            s.append('0')
                    if continuous:
                        s.append(str(value))
                    else:
                        s.append(str(int(value)))
                    ss.append(s)
                    break

        self.accuracies = effect_dct
        final_accs = self.results_analysed(ra_named, metrics, effect_type())
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
        for effect, c in list(self.accuracies.keys()):
            accs = self.accuracies[effect, c]
            for m_i in range(len(metrics)):
                v = accs[metrics[m_i]]
                if v > 0.0:
                    cov[m_i] += 1

        if continuous:
            headers = ['0.1%ile', '.25%ile', '0.5%ile', '1%ile', '5%ile',
                       '10%ile', '20%ile', '33%ile', '50%ile', '100%ile']
        else:
            headers = ['top10', 'top25', 'top50', 'top100', 'top{}'.format(len(self.compounds)),
                       'top1%', 'top5%', 'top10%', 'top50%', 'top100%']
        # Create average indication accuracy list in percent
        ia = []
        for m in metrics:
            ia.append(final_accs[m] * 100.0)
        # Create average pairwise accuracy list in percent
        pa = [(x * 100.0 / len(ss)) for x in top_pairwise]
        # Indication coverage
        cov = map(int, cov)
        # Append 3 lists to df and write to file
        with open(summ, 'w', encoding="utf8") as sf:
            sf.write("\t" + '\t'.join(headers) + '\n')
            iast = "\t".join(map(str, [format(x, ".3f") for x in ia]))
            pwst = "\t".join(map(str, [format(x, ".3f") for x in pa]))
            icst = "\t".join(map(str, cov)) + '\n'
            sf.write('aia\t{}\napa\t{}\nic\t{}\n'.format(iast, pwst, icst))
       
        # pretty print the average indication accuracies
        cut = 0
        print("\taia")
        for m in metrics:
            print("{}\t{:.3f}".format(headers[cut], final_accs[m] * 100.0))
            cut += 1
        print('\n')

    def canbenchmark_new(self, file_name, n=10, indications=[], continuous=False, bottom=False,
                         ranking='standard', adrs=False, associated=True, addl_metrics=[], write_addl=False,\
                         exclude_indic=False, tierank='min'):
        """!
        Benchmarks the platform based on consensus compound similarity of those approved for the same diseases
        This function tests the performance of canpredict_compounds

        @param file_name str: Name to be used for the various results files (e.g. file_name=test --> summary_test.tsv)
        @param indications list of str: List of Indication ids to be benchmarked, otherwise all will be used.
        @param continuous bool: Use the percentile of distances from the similarity matrix as the benchmarking cutoffs
        @param bottom bool: Reverse the ranking (descending) for the benchmark
        @param ranking str: What ranking method to use for the compounds. This really only affects ties. (standard,
        modified, and ordinal)
        @param adrs bool: ADRs are used as the Compounds' phenotypic effects instead of Indications
        @param addl_metrics list of func: functions to compute additional metrics
            input - iterable of ranks, (iterable of out_ofs), thresholds
            output - either single # result (eg AUROC) or tuple of one # per threshold
        @param write_addl bool: If True, will create a results_analysed_named file for each addl_metrics func (default: False)
        @param exlude_indic: If True, will only rank left out compound against non-indicated compounds (default: False)
        @param tierank str: rank that tied compounds will take; see polars.Expr.rank for full options (default: min)
            options - min, max, average, ordinal, random (for 3 tied compounds, could result in: 1/1/1, 3/3/3, 2/2/2, 1/2/3, 3/1/2)
        @return Returns None
        """

        def print_time(s):
            if s >= 60:
                m = s / 60.0
                s %= 60.0
                if m >= 60.0:
                    h = m / 60.0
                    m %= 60.0
                    return f"{h:.0f} hr {m:.0f} min {s:.0f} s"
                else:
                    return f"{m:.0f} min {s:.0f} s"
            else:
                return f"{s:.0f} s"

        print("Begin running canbenchmark_new...")
        start = time.time()

        #ZF - 1/19/24 - check if this is needed later
        assert not continuous or not (self.indication_pathways or self.indication_proteins),\
               'Continuous benchmarking and indication-based signatures are not compatible'
        #if (continuous and self.indication_pathways) or (continuous and self.indication_proteins):
        #    print('Continuous benchmarking and indication-based signatures are not compatible, quitting.')
        #   exit()

        if not self.indication_proteins and not self.indication_pathways:
            if not self.compounds[0].similar_sorted:
                for c in self.compounds:
                    sorted_scores = sorted(c.similar, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
                    c.similar = sorted_scores
                    c.similar_sorted = True

        associated_str = '-associated' if associated else ''
        ra_named = f'results_analysed_named/results_analysed_named-{file_name}-{n}{associated_str}.tsv'
        ra_named_ndcg = f"results_analysed_named/results_analysed_named-ndcg-{file_name}-{n}{associated_str}.tsv"
        ra = f'raw_results/raw_results-{file_name}-{n}{associated_str}.csv'
        pwr = f'pairwise_results/pairwise_results-{file_name}-{n}{associated_str}.csv'
        summ = f'summary-{file_name}-{n}{associated_str}.tsv'
        benchmark_name = f"canbenchmark-{file_name}-{n}{associated_str}"
        t_name = f"time-{file_name}-{n}{associated_str}.txt"
        if write_addl:
            addl_named = f'results_analysed_named/results_analysed_named-%s-{file_name}-{n}{associated_str}.tsv'

        os.makedirs('results_analysed_named', exist_ok=True)
        os.makedirs('raw_results', exist_ok=True)
        os.makedirs('pairwise_results', exist_ok=True)
        os.makedirs(benchmark_name, exist_ok=True)

        ra_out = open(ra, 'w')
        pwr_out = open(pwr, 'w')

        def dcg(l,k):
            dcg = [((2**x)-1)/(math.log2(i+1)) for i,x in enumerate(l[:k],1)]
            return np.sum(dcg)

        effect_type = 'ADR' if adrs else 'disease'

        def filter_indications(ind_set):
            if not os.path.exists('v2.0/mappings/group_disease-top_level.tsv'):
                url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2/mappings/group_disease-top_level.tsv'
                dl_file(url, 'v2.0/mappings/group_disease-top_level.tsv')
            path_ids = ['C01', 'C02', 'C03']
            with open('v2.0/mappings/group_disease-top_level.tsv', 'r') as fgd:
                for l in fgd:
                    ls = l.strip().split('\t')
                    if ls[1] in path_ids:
                        ind = self.get_indication(ls[0])
                        ind.pathogen = True
            assert ind_set in ('pathogen', 'human'), 'Please enter proper indication set; options include "pathogen" or "human"'
            if ind_set == 'pathogen':
                return [indx for indx in self.indications if indx.pathogen]
            elif ind_set == 'human':
                return [indx for indx in self.indications if not indx.pathogen]
            #else:
            #    print('Please enter proper indication set, options include "pathogen", "human", or "all".')
            #   quit()

        effect_dct = {}
        ss = []
        c_per_effect = 0

        if associated:
            cmpd_lib = [str(self.get_compound(c).id_) for effect in self.indications for c in effect.compounds]
            #cmpd_lib = [str(c.id_) for c in self.compounds if len(c.indications)>=1]
        else:
            cmpd_lib = [str(c.id_) for c in self.compounds]

        cmpd_lib = list(set(cmpd_lib))

        if isinstance(indications, list) and len(indications) >= 1:
            effects = list(map(self.get_indication, indications))
        elif isinstance(indications, list) and len(indications) == 0 and not adrs:
            effects = self.indications
        elif adrs:
            effects = self.adrs
        else:
            if isinstance(indications, str):
                if indications == 'all':
                    effects = self.indications
                else:
                    effects = filter_indications(indications)
        
        effects = [effect for effect in effects if len(effect.compounds) > 1]
        def cont_metrics():
            all_v = []
            for c in self.compounds:
                for s in c.similar:
                    if s[1] != 0.0:
                        all_v.append(s[1])
            avl = len(all_v)
            all_v_sort = sorted(all_v)
            # for tuple 10, have to add the '-1' for index out of range reasons
            metrics = [(1, all_v_sort[int(avl / 1000.0)]), (2, all_v_sort[int(avl / 400.0)]),
                       (3, all_v_sort[int(avl / 200.0)]),
                       (4, all_v_sort[int(avl / 100.0)]), (5, all_v_sort[int(avl / 20.0)]),
                       (6, all_v_sort[int(avl / 10.0)]),
                       (7, all_v_sort[int(avl / 5.0)]), (8, all_v_sort[int(avl / 3.0)]),
                       (9, all_v_sort[int(avl / 2.0)]),
                       (10, all_v_sort[int(avl / 1.0) - 1])]
            return metrics

        x = (len(cmpd_lib)) / 100.0  # changed this...no reason to use similar instead of compounds
        #x = (len(self.compounds)) / 100.0  # changed this...no reason to use similar instead of compounds
        # had to change from 100.0 to 100.0001 because the int function
        # would chop off an additional value of 1 for some reason...
        if continuous:
            metrics = cont_metrics()
        else:
            metrics = [(1, 10), (2, 25), (3, 50), (4, 100), (5, int(x * 100.0001)),
                       (6, int(x * 1.0001)), (7, int(x * 5.0001)), (8, int(x * 10.0001)),
                       (9, int(x * 50.0001)), (10, int(x * 100.0001))]

        if continuous:
            ra_out.write("compound_id,{}_id,0.1%({:.3f}),0.25%({:.3f}),0.5%({:.3f}),"
                         "1%({:.3f}),5%({:.3f}),10%({:.3f}),20%({:.3f}),33%({:.3f}),"
                         "50%({:.3f}),100%({:.3f}),value\n".format(effect_type, metrics[0][1], metrics[1][1],
                                                                   metrics[2][1], metrics[3][1], metrics[4][1],
                                                                   metrics[5][1], metrics[6][1], metrics[7][1],
                                                                   metrics[8][1], metrics[9][1]))
        else:
            ra_out.write(f"compound_id,effect_id,top10,top25,top50,top100,"
                         f"top{len(cmpd_lib)},top1%,top5%,top10%,top50%,top100%,"
                         f"rank,score,avg_rank,avg_dist,conf\n")
            pwr_out.write(f"cmpd_id-1,cmpd_id-2,effect_id,top10,top25,top50,top100,"
                          f"top{len(cmpd_lib)},top1%,top5%,top10%,top50%,top100%,"
                          f"rank,dist\n")
        print("  Calculating scores...")
        start_calc = time.time()
        if self.ncpus > 1:
            # tqdm progressbar is not working for multiprocessing
            pool = mp.Pool(processes=self.ncpus)
            pool.starmap_async(ind_accuracies, [(effect.id_, effect.compounds, cmpd_lib, benchmark_name, metrics,\
                                                 associated, n, self.db_name, self.dist_metric, exclude_indic, tierank) for effect in effects], chunksize=20).get()
            pool.close
            pool.join
        else:
            pbar = tqdm(effects) if self.pbar else effects
            [ind_accuracies(effect.id_, effect.compounds, cmpd_lib, benchmark_name, metrics, associated, n, self.db_name, self.dist_metric, exclude_indic, tierank)\
             for effect in pbar] #TQDM2
        t_calc = print_time(time.time() - start_calc)
        print("  Done calculating scores.")
        print(f"  Time to calculate scores: {t_calc}")
        with open(t_name,'w') as tw:
            tw.write(f"Number of CPUs: {self.ncpus}\n")
            tw.write(f"Time to calculate scores: {t_calc}\n")

        print("  Compiling and saving results...")
        # Average Indication Accuracy
        aia_accs = []
        # Pairwise Accuracy
        pwa_count = [0.0]*len(metrics)
        pwa_tot = 0
        top_pairwise = [0.0] * 10
        # Coverage
        cov_count = [0]*len(metrics)
        # NDCG
        ndcg = {}
        for m in metrics:
            ndcg[m[0]] = {}
        # Results_analysed_named
        #for effect_id in ranks.keys():
        pd.options.display.float_format='{:.3f}'.format

        addl_results = {}
        pbar = tqdm(range(len(effects))) if self.pbar else range(len(effects)) #TQDM4
        for i in pbar:
            effect = effects[i]
            effect_id = effect.id_
            addl_results[effect_id] = [[] for met in addl_metrics]
            addl_list = addl_results[effect_id]
            #pbar.set_description(f"    {effect_id}")
            #print(effect_id)
            db_benchmark = create_engine(f'sqlite:///{benchmark_name}/{effect_id}.db')
            df_ia = pd.read_sql("SELECT * FROM ia_results",db_benchmark)
            df_pa = pd.read_sql("SELECT * FROM pa_results",db_benchmark)
            #effect = self.get_adr(effect_id)
            count = [0.0]*len(metrics)
            tot = len(effect.compounds)
            effect_dct[(effect, tot)] = {}
            ndcg_l = {}
            for m in metrics:
                effect_dct[(effect, tot)][m] = 0.0
                ndcg_l[m[0]] = []
            #pwa_tot += tot
            c_ideal = [0]*len(cmpd_lib)
            c_ideal[0] = 1

            ranks = []
            for c_idx in df_ia.index:
                rank = int(df_ia.loc[c_idx,'rank'])
                ranks.append(rank)
                c_rank = [0]*len(cmpd_lib)
                c_rank[rank-1] = 1
                # Check ranks against each top metric
                for idx,m in enumerate(metrics):
                    #if ranks[effect_id][cp_query][1] <= m[1]:
                    if rank <= m[1]:
                        effect_dct[(effect, tot)][m] += 1.0
                        count[idx]+=1.0
                        #pwa_count[idx]+=1.0
                    # NDCG
                    ndcg_l[m[0]].append(dcg(c_rank, int(m[1])) / dcg(c_ideal, int(m[1])))

                # compound-indication results
                ra_out.write(','.join(df_ia.loc[c_idx,:].values.tolist()) + '\n')

            for i in range(len(addl_metrics)):
                func = addl_metrics[i]
                results = func(thresholds=metrics, ranks=ranks, out_of=len(cmpd_lib) - ((tot - 1) if exclude_indic else 0))
                addl_list[i] = results

            # Pairwise accuracy
            pwa_tot += len(df_pa)
            for c_idx in df_pa.index:
                rank = int(df_pa.loc[c_idx,'rank'])
                for idx,m in enumerate(metrics):
                    #if ranks[effect_id][cp_query][1] <= m[1]:
                    if rank <= m[1]:
                        pwa_count[idx]+=1.0
                pwr_out.write(','.join(df_pa.loc[c_idx,:].values.tolist()) + '\n')

            # NDCG
            for m in metrics:
                ndcg[m[0]][effect] = np.mean(ndcg_l[m[0]])

            # Indication coverage
            count = [(i/tot)*100.0 for i in count]
            for idx,i in enumerate(count):
                if i>0.0:
                    cov_count[idx]+=1
            aia_accs.append(count)
            #print(effect_id,effect.name,count)
        ra_out.close()

        addl_names = [x.__name__ for x in addl_metrics]
        addl_results['overall'] = [None for met in addl_metrics]
        for i in range(len(addl_metrics)):
            avg_list = [0]*len(metrics)

            for effect in addl_results.keys():
                if effect == 'overall':
                    continue
                for j in range(len(metrics)):
                    avg_list[j] += addl_results[effect][i][j]

            avg_list = [x/(len(addl_results) - 1) for x in avg_list]
            addl_results['overall'][i] = avg_list

        top_metrics = [str(j) for i,j in metrics]
        aia_accs = [str(sum(sub_list) / len(sub_list)) for sub_list in zip(*aia_accs)]
        pwa_accs = [str((i/pwa_tot)*100.0) for i in pwa_count]
        cov_count = [str(i) for i in cov_count]

        #print("\t".join(top_metrics))
        #print("\t".join(aia_accs))
        #print("\t".join(pwa_accs))
        #print("\t".join(cov_count))
        # Fix later
        self.accuracies = effect_dct
        final_accs = self.results_analysed(ra_named, metrics, effect_type)
        #ss = sorted(ss, key=lambda xx: xx[0])
        #ss = sorted(ss, key=lambda xx: int(xx[0]))

        # NDCG - results_analysed_named
        with open(ra_named_ndcg, 'w') as o:
            o.write(f"effect_id\tcmpds_per_effect\ttop10\ttop25\ttop50\ttop100\ttop{len(cmpd_lib)}\ttop1%\ttop5%\ttop10%\ttop50%\ttop100%\teffect_name\n")
            for effect in effects:
                o.write(f"{effect.id_}\t{len(effect.compounds)}")
                for m in metrics:
                    o.write(f"\t{ndcg[m[0]][effect]:.5f}")
                o.write(f"\t{effect.name}\n")

        if write_addl:
            format_str = '%s\t%d\t' + '\t'.join(['%.3f' for x in metrics]) + '\t%s\n'
            for i in range(len(addl_metrics)):
                out = ['effect_id\tcmpds_per_effect\t' + '\t'.join(['top' + str(x[1]) for x in metrics]) + '\teffect_name\n']
                for effect in effects:
                    effect_id = effect.id_
                    out.append(format_str % \
                               ((effect_id, len(effect.compounds)) + tuple(addl_results[effect_id][i]) + (effect.name,)))
                with open(addl_named % addl_names[i], 'w') as f:
                    f.writelines(out)


        pa = [(i/pwa_tot)*100.0 for i in pwa_count]

        cov = [0] * 10
        for effect, c in list(self.accuracies.keys()):
            accs = self.accuracies[effect, c]
            for m_i in range(len(metrics)):
                if accs[metrics[m_i]] > 0.0:
                    cov[m_i] += 1
        cov = map(int, cov)

        nndcg = [0.0] * 10
        for idx, m in enumerate(metrics):
            l = []
            for effect in effects:
                l.append(ndcg[m[0]][effect])
            nndcg[idx] = np.mean(l)

        # Create average indication accuracy list in percent
        ia = []
        for m in metrics:
            ia.append(final_accs[m] * 100.0)
        # Create average pairwise accuracy list in percent
        # Old method - pairwise compound-indication
        #pa = [(x * 100.0 / pwa_tot) for x in top_pairwise]

        # Control calculation
        control_aia = []
        for n in range(len(metrics)):
            p = hypergeom.pmf(1, len(cmpd_lib), 1, metrics[n][1])
            control_aia.append(p * 100.0)
        print("  Done compiling and saving results.")

        if continuous:
            headers = ['0.1%ile', '.25%ile', '0.5%ile', '1%ile', '5%ile',
                       '10%ile', '20%ile', '33%ile', '50%ile', '100%ile']
        else:
            headers = ['top10', 'top25', 'top50', 'top100', f'top{len(cmpd_lib)}',
                       'top1%', 'top5%', 'top10%', 'top50%', 'top100%']
        # pretty print the average indication accuracies
        if addl_metrics:
            col_names = ['nAIA','control-nAIA','PA','IC','nNDCG'] + addl_names
            overalls = addl_results['overall']
            content = list(zip(ia, control_aia, pa, cov, nndcg, *overalls))
            df_summ = pd.DataFrame(content, columns=col_names, index=[headers])
        else:
            df_summ = pd.DataFrame(list(zip(ia, control_aia, pa, cov, nndcg)), columns=['nAIA','control-nAIA','PA','IC','nNDCG'], index=[headers])
        print("\nSummary")
        print(df_summ.T)
        print()
        df_summ.T.to_csv(summ, float_format='%.5f', sep='\t')
        t_tot = print_time(time.time()-start)
        print("Done running canbenchmark_new.")
        print(f"Total time to run canbenchmark_new: {t_tot}\n")
        with open(t_name,'a') as tw:
            tw.write(f"Total time to run canbenchmark_new: {t_tot}")

        return df_summ

    def canbenchmark_associated(self, file_name, indications=[], continuous=False, ranking='standard'):
        """!
        Benchmark only the compounds in the indication mapping, aka get rid of "noisy" compounds.
        This function returns the filtered CANDO object in the event that you want to explore further.

        @param file_name str: Name to be used for the variosu results files (e.g. file_name=test --> summary_test.tsv)
        @param indications list: List of Indication ids to be used for this benchmark, otherwise all will be used.
        @param continuous bool: Use the percentile of distances from the similarity matrix as the benchmarking cutoffs
        @param ranking str: What ranking method to use for the compounds. This really only affects ties.
        (standard, modified, and ordinal)
        @return Returns None
        """
        print("Making CANDO copy with only benchmarking-associated compounds")
        cp = CANDO(self.c_map, self.i_map, matrix=self.matrix, compound_set=self.compound_set, dist_metric=self.dist_metric)
        good_cs = []
        good_ids = []
        for ind in cp.indications:
            if len(ind.compounds) >= 2:
                for c in ind.compounds:
                    if c not in good_ids:
                        good_cs.append(self.get_compound(c))
                        good_ids.append(c)
        cp.compounds = good_cs
        
        print('Computing {} distances...'.format(self.dist_metric))

        for c in cp.compounds:
            cp.generate_similar_sigs(c, sort=True)
            good_sims = []
            for s in c.similar:
                if s[0].id_ not in good_ids:
                    pass
                else:
                    good_sims.append(s)
            good_sims = [(c.id_,dist) for c,dist in good_sims]
            c.similar = sorted(good_sims, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
            c.similar_computed = True
            c.similar_sorted = True
        
        print('Done computing {} distances.\n'.format(self.dist_metric))
        
        cp.canbenchmark(file_name=file_name, indications=indications, continuous=continuous, ranking=ranking)

    def canbenchmark_bottom(self, file_name, indications=[], ranking='standard'):
        """!
        Benchmark the reverse ranking of similar compounds as a control.

        @param file_name str: Name to be used for the variosu results files (e.g. file_name=test --> summary_test.tsv)
        @param indications list: List of Indication ids to be used for this benchmark, otherwise all will be used.
        @param ranking str: What ranking method to use for the compounds. This really only affects ties. (standard,
        modified, and ordinal)
        @return Returns None
        """
        print("Making CANDO copy with reversed compound ordering")
        cp = CANDO(self.c_map, self.i_map, self.matrix)
        
        print('Computing {} distances...'.format(self.dist_metric))
        
        for ic in range(len(cp.compounds)):
            cp.generate_similar_sigs(cp.compounds[ic], sort=True)
            sorted_scores = sorted(cp.compounds[ic].similar, key=lambda x: x[1])[::-1]
            cp.compounds[ic].similar = [(c.id_,dist) for c,dist in sorted_scores]
            cp.compounds[ic].similar_computed = True
            cp.similar_sorted = True
        
        print('Done computing {} distances.\n'.format(self.dist_metric))
        
        cp.canbenchmark(file_name=file_name, indications=indications, ranking=ranking, bottom=True)

    def canbenchmark_ndcg(self, file_name):
        """!
        Benchmark using the normalized discounted cumulative gain metric

        @param file_name str: Name to be used for the results files (file_name=test --> summary_ndcg-test.tsv)
        @return Returns None
        """
        def dcg(l,k):
            dcg = [((2**x)-1)/(math.log2(i+1)) for i,x in enumerate(l[:k],1)]
            return np.sum(dcg)

        k_s = [10,25,50,100,len(self.compounds),0.01*len(self.compounds),0.05*len(self.compounds),0.10*len(self.compounds),0.50*len(self.compounds),len(self.compounds)]
        i_accs = {}
        c_accs = {}
        nz_counts = {}
        for k in range(len(k_s)):
            i_accs[k] = {}
            c_accs[k] = []
            nz_counts[k] = 0
        for ind in self.indications:
            if len(ind.compounds) < 2:
                continue
            #approved_ids = [i.id_ for i in ind.compounds]
            approved_ids = ind.compounds
            acc = {}
            for k in range(len(k_s)):
                acc[k] = []
            for c_id in ind.compounds:
                c = self.get_compound(c_id)
                if not c.similar_sorted:
                    sorted_scores = sorted(c.similar, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
                    c.similar = sorted_scores
                    c.similar_sorted = True
                c_ideal = [0]*len(c.similar)
                for x in range(len(approved_ids)):
                    c_ideal[x] = 1
                c_rank = []
                for x in c.similar:
                    if x[0] in approved_ids:
                        c_rank.append(1)
                    else:
                        c_rank.append(0)
                for k in range(len(k_s)):
                    acc[k].append(dcg(c_rank,int(k_s[k]))/dcg(c_ideal,int(k_s[k])))
                    c_accs[k].append((c.id_,ind.id_,dcg(c_rank,int(k_s[k]))/dcg(c_ideal,int(k_s[k]))))
            for k in range(len(k_s)):
                i_accs[k][ind.id_] = (ind,np.mean(acc[k]))
        for k in range(len(k_s)):
            # Non-zero ndcg
            i_accs_nz = [i_accs[k][x][1] for x in i_accs[k] if i_accs[k][x][1] > 0.0]
            nz_counts[k] = len(i_accs_nz)
        # Write NDCG results per indication in results_analysed_named
        if not os.path.exists('./results_analysed_named/'):
            os.mkdir('results_analysed_named')
            #os.system('mkdir results_analysed_named')
        with open("results_analysed_named/results_analysed_named_ndcg-{}.tsv".format(file_name), 'w', encoding="utf8") as o:
            o.write("disease_id\tcmpds_per_disease\ttop10\ttop25\ttop50\ttop100\ttop{}\ttop1%\ttop5%\ttop10%\ttop50%\ttop100%\tdisease_name\n".format(len(self.compounds)))
            for x in i_accs[0]:
                o.write("{}\t{}".format(i_accs[0][x][0].id_,len(i_accs[0][x][0].compounds)))
                for k in range(len(k_s)):
                    o.write("\t{:.3f}".format(i_accs[k][x][1]))
                o.write("\t{}\n".format(i_accs[0][x][0].name))
        # Write NDCG results per compound-indication pair in raw_results
        if not os.path.exists('./raw_results/'):
            os.mkdir('raw_results')
            #os.system('mkdir raw_results')
        with open("raw_results/raw_results_ndcg-{}.csv".format(file_name), 'w', encoding="utf8") as o:
            o.write("compound_id,disease_id,top10,top25,top50,top100,top{},top1%,top5%,top10%,top50%,top100%\n".format(len(self.compounds)))
            for x in range(len(c_accs[0])):
                o.write("{},{}".format(c_accs[0][x][0],c_accs[0][x][1]))
                for k in range(len(k_s)):
                    o.write(",{:.3f}".format(c_accs[k][x][2]))
                o.write("\n")
        # Write a summary file for NDCG
        with open("summary_ndcg-{}.tsv".format(file_name), 'w', encoding="utf8") as o:
            o.write("\ttop10\ttop25\ttop50\ttop100\ttop{}\ttop1%\ttop5%\ttop10%\ttop50%\ttop100%\n".format(len(self.compounds)))
            o.write("ai-ndcg")
            for k in range(len(k_s)):
                o.write("\t{:.3f}".format(np.mean(list(zip(*i_accs[k].values()))[1])))
            o.write("\n")
            o.write("ap-ndcg")
            for k in range(len(k_s)):
                o.write("\t{:.3f}".format(np.mean(list(zip(*c_accs[k]))[2])))
            o.write("\n")
            o.write("ic-ndcg")
            for k in range(len(k_s)):
                o.write("\t{}".format(int(nz_counts[k])))
            o.write("\n")
        #print("NDCG averaged over {} indications = {}".format(len(i_accs),np.mean(list(zip(*i_accs.values()))[1])))
        #print("Pairwise NDCG averaged over {} compound-indication pairs = {}".format(len(c_accs),np.mean(list(zip(*c_accs))[3])))

    def canbenchmark_cluster(self, n_clusters=5):
        """!
        Benchmark using k-means clustering

        @param n_clusters int: Number of clusters for k-means
        @return Returns None
        """
        def cluster_kmeans(cmpds):
            def f(x):
                return x.sig

            def g(x):
                return tuple(x.indications)

            def h(x):
                return x.id_

            sigs = np.array(list(map(f, cmpds)))
            pca = PCA(n_components=10).fit(sigs)
            sigs = pca.transform(sigs)
            inds = np.array(map(g, cmpds))
            ids = np.array(list(map(h, cmpds)))
            sigs_train, sigs_test, inds_train, inds_test, ids_train, ids_test = train_test_split(sigs, inds, ids,
                                                                                                 test_size=0.20,
                                                                                                 random_state=1)
            clusters = KMeans(n_clusters, random_state=1).fit(sigs_train)
            return clusters, sigs_test, inds_train, inds_test, ids_train, ids_test

        # Calculate the K means clusters for all compound signatures
        cs, sigs_test, inds_train, inds_test, ids_train, ids_test = cluster_kmeans(self.compounds)
        labels = cs.labels_

        # Determine how many compounds are in each cluster
        # Plot the results and output the mean, median, and range
        c_clusters = [0] * n_clusters
        for l in labels:
            c_clusters[l] += 1
        '''
        all_clusters = range(n_clusters)
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
                    if done:
                        break
                total_count += 1

        print("Number of cluster = {}".format(n_clusters))
        print("Mean cluster size = {}".format(np.mean(c_clusters)))
        print("Median cluster size = {}".format(np.median(c_clusters)))
        print("Range of cluster sizes = [{},{}]".format(np.min(c_clusters), np.max(c_clusters)))
        print("% Accuracy = {}".format(total_acc / total_count * 100.0))

    def compounds_analysed(self, f, metrics):
        fo = open(f, 'w', encoding="utf8")
        cmpds = list(self.accuracies.keys())
        cmpds_sorted = sorted(cmpds, key=lambda x: (len(x[0].compounds), x[0].id_))[::-1]
        l = len(cmpds)
        final_accs = {}
        for m in metrics:
            final_accs[m] = 0.0
        for cmpd, c in cmpds_sorted:
            fo.write("{0}\t{1}\t".format(cmpd.id_, c))
            accs = self.accuracies[(cmpd,c)]
            for m in metrics:
                n = accs[m]
                y = str(n / c * 100)[0:4]
                fo.write("{}\t".format(y))

                final_accs[m] += n / c / l
            fo.write("|\t{}\n".format(cmpd.name))
        fo.close()
        return final_accs

    def canbenchmark_compounds(self, file_name, adrs=[], continuous=False,
                          bottom=False, ranking='standard'):
        """!
        Benchmarks the platform based on compound similarity of those known to interact with other compounds.

        @param file_name str: Name to be used for the various results files (e.g. file_name=test --> summary_test.tsv)
        @param adrs list: List of ADR ids to be used for this benchmark, otherwise all will be used.
        @param continuous bool: Use the percentile of distances from the similarity matrix as the cutoffs for
        benchmarking
        @param bottom bool: Reverse the ranking (descending) for the benchmark
        @param ranking str: What ranking method to use for the compounds. This really only affects ties. (standard,
        modified, and ordinal)
        @return Returns None
        """
        assert not continuous or not (self.indication_pathways or self.indication_proteins),\
               'Continuous benchmarking and indication-based signatures are not compatible'
        #if (continuous and self.indication_pathways) or (continuous and self.indication_proteins):
        #   print('Continuous benchmarking and indication-based signatures are not compatible, quitting.')
        #   exit()

        if not self.indication_proteins and not self.indication_pathways:
            if not self.compounds[0].similar_sorted:
                for c in self.compounds:
                    sorted_scores = sorted(c.similar, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
                    c.similar = sorted_scores
                    c.similar_sorted = True

        if not os.path.exists('./results_analysed_named'):
            print("Directory 'results_analysed_named' does not exist, creating directory")
            os.mkdir('results_analysed_named')
            #os.system('mkdir results_analysed_named')
        if not os.path.exists('./raw_results'):
            print("Directory 'raw_results' does not exist, creating directory")
            os.mkdir('raw_results')
            #os.system('mkdir raw_results')

        ra_named = 'results_analysed_named/results_analysed_named_' + file_name + '-cmpds.tsv'
        ra = 'raw_results/raw_results_' + file_name + '-cmpds.csv'
        summ = 'summary_' + file_name + '-cmpds.tsv'
        ra_out = open(ra, 'w', encoding="utf8")

        def effect_type():
            if adrs:
                return 'ADR'
            else:
                return 'disease'

        def competitive_standard_bottom(sims, r):
            rank = 0
            for sim in sims:
                if sim[1] > r:
                    rank += 1.0
                else:
                    return rank
            return len(sims)

        def competitive_modified_bottom(sims, r):
            rank = 0
            for sim in sims:
                if sim[1] >= r:
                    rank += 1.0
                else:
                    return rank
            return len(sims)

        # Competitive modified ranking code
        def competitive_modified(sims, r):
            rank = 0
            for sim in sims:
                if sim[1] <= r:
                    rank += 1.0
                else:
                    return rank
            return len(sims)

        # Competitive standard ranking code
        def competitive_standard(sims, r):
            rank = 0
            for sim in sims:
                if sim[1] < r:
                    rank += 1.0
                else:
                    return rank
            return len(sims)

        cmpd_dct = {}
        ss = []
        c_per_cmpd = 0

        def cont_metrics():
            all_v = []
            for c in self.compounds:
                for s in c.similar:
                    if s[1] != 0.0:
                        all_v.append(s[1])
            avl = len(all_v)
            all_v_sort = sorted(all_v)
            # for tuple 10, have to add the '-1' for index out of range reasons
            metrics = [(1, all_v_sort[int(avl/1000.0)]), (2, all_v_sort[int(avl/400.0)]), (3, all_v_sort[int(avl/200.0)]),
                       (4, all_v_sort[int(avl/100.0)]), (5, all_v_sort[int(avl/20.0)]), (6, all_v_sort[int(avl/10.0)]),
                       (7, all_v_sort[int(avl/5.0)]), (8, all_v_sort[int(avl/3.0)]), (9, all_v_sort[int(avl/2.0)]),
                       (10, all_v_sort[int(avl/1.0)-1])]
            return metrics

        x = (len(self.compounds)) / 100.0  # changed this...no reason to use similar instead of compounds
        # had to change from 100.0 to 100.0001 because the int function
        # would chop off an additional value of 1 for some reason...
        if continuous:
            metrics = cont_metrics()
        else:
            metrics = [(1, 10), (2, 25), (3, 50), (4, 100), (5, int(x*100.0001)),
                       (6, int(x*1.0001)), (7, int(x*5.0001)), (8, int(x*10.0001)),
                       (9, int(x*50.0001)), (10, int(x*100.0001))]

        if continuous:
            ra_out.write("compound_id,compound_id,0.1%({:.3f}),0.25%({:.3f}),0.5%({:.3f}),"
                         "1%({:.3f}),5%({:.3f}),10%({:.3f}),20%({:.3f}),33%({:.3f}),"
                         "50%({:.3f}),100%({:.3f}),value\n".format(metrics[0][1], metrics[1][1],
                                                                     metrics[2][1], metrics[3][1], metrics[4][1],
                                                                     metrics[5][1], metrics[6][1], metrics[7][1],
                                                                     metrics[8][1], metrics[9][1]))
        else:
            ra_out.write("compound_id,compound_id,top10,top25,top50,top100,"
                         "topAll,top1%,top5%,top10%,top50%,top100%,rank\n")

        for cmpd in self.compounds:
            count = len(cmpd.compounds)
            if count < 2:
                continue
            c_per_cmpd += count
            cmpd_dct[(cmpd, count)] = {}
            for m in metrics:
                cmpd_dct[(cmpd, count)][m] = 0.0
            # retrieve the appropriate proteins/pathway indices here, should be
            # incorporated as part of the ind object during file reading
            vs = []

            for c in cmpd.compounds:
                for cs in c.similar:
                    if cs[0] in cmpd.compounds:
                    #if cmpd in cs[0].compounds:
                        cs_rmsd = cs[1]
                    else:
                        continue

                    value = 0.0
                    if continuous:
                        value = cs_rmsd
                    elif bottom:
                        if ranking == 'modified':
                            value = competitive_modified_bottom(c.similar, cs_rmsd)
                        elif ranking == 'standard':
                            value = competitive_standard_bottom(c.similar, cs_rmsd)
                        elif ranking == 'ordinal':
                            value = c.similar.index(cs)
                        else:
                            print("Ranking function {} is incorrect.".format(ranking))
                            exit()
                    elif ranking == 'modified':
                        value = competitive_modified(c.similar, cs_rmsd)
                    elif ranking == 'standard':
                        value = competitive_standard(c.similar, cs_rmsd)
                    elif ranking == 'ordinal':
                        value = c.similar.index(cs)
                    else:
                        print("Ranking function {} is incorrect.".format(ranking))
                        exit()

                    s = [str(c.index), str(cmpd.id_)]
                    for x in metrics:
                        if value <= x[1]:
                            cmpd_dct[(cmpd, count)][x] += 1.0
                            s.append('1')
                        else:
                            s.append('0')
                    if continuous:
                        s.append(str(value))
                    else:
                        s.append(str(value))
                    ss.append(s)
                    break

        self.accuracies = cmpd_dct
        final_accs = self.compounds_analysed(ra_named, metrics)
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
        for cmpd, c in list(self.accuracies.keys()):
            accs = self.accuracies[cmpd, c]
            for m_i in range(len(metrics)):
                v = accs[metrics[m_i]]
                if v > 0.0:
                    cov[m_i] += 1

        if continuous:
            headers = ['0.1%ile', '.25%ile', '0.5%ile', '1%ile', '5%ile',
                       '10%ile', '20%ile', '33%ile', '50%ile', '100%ile']
        else:
            headers = ['top10', 'top25', 'top50', 'top100', 'top{}'.format(len(self.compounds)),
                       'top1%', 'top5%', 'top10%', 'top50%', 'top100%']
        # Create average indication accuracy list in percent
        ia = []
        for m in metrics:
            ia.append(final_accs[m] * 100.0)
        # Create average pairwise accuracy list in percent
        pa = [(x * 100.0 / len(ss)) for x in top_pairwise]
        # Indication coverage
        cov = map(int, cov)
        # Append 3 lists to df and write to file
        with open(summ, 'w', encoding="utf8") as sf:
            sf.write("\t" + '\t'.join(headers) + '\n')
            iast = "\t".join(map(str, [format(x, ".3f") for x in ia]))
            pwst = "\t".join(map(str, [format(x, ".3f") for x in pa]))
            icst = "\t".join(map(str, cov)) + '\n'
            sf.write('aia\t{}\napa\t{}\nic\t{}\n'.format(iast, pwst, icst))

        # pretty print the average indication accuracies
        cut = 0
        print("\taia")
        for m in metrics:
            print("{}\t{:.3f}".format(headers[cut], final_accs[m] * 100.0))
            cut += 1
        print('\n')

    def canbenchmark_ddi(self, file_name, adrs=[], continuous=False, n=100, approved=False,
                          bottom=False, ranking='standard'):
        """!
        Benchmarks the platform based on compound pairs known to cause ADRs

        @param file_name str: Name to be used for the results files (file_name=test --> summary_test-ddi_adr.tsv)
        @param continuous bool: Use the percentile of distances from the similarity matrix as the cutoffs for
        benchmarking
        @param bottom bool: Reverse the ranking (descending) for the benchmark
        @param ranking str: What ranking method to use for the compounds. This really only affects ties. (standard,
        modified, and ordinal)
        @return Returns None
        """

        def print_time(s):
            if s >= 60:
                m = s / 60.0
                s %= 60.0
                if m >= 60.0:
                    h = m / 60.0
                    m %= 60.0
                    return f"{h:.0f} hr {m:.0f} min {s:.0f} s"
                else:
                    return f"{m:.0f} min {s:.0f} s"
            else:
                return f"{s:.0f} s"

        '''
        if (continuous and self.indication_pathways) or (continuous and self.indication_proteins):
            print('Continuous benchmarking and indication-based signatures are not compatible, quitting.')
            exit()
        '''
        '''
        if not self.indication_proteins and not self.indication_pathways:
            if not self.compound_pairs[0].similar_sorted:
            #if not self.compound_pairs[0].similar_sorted and not associated:
                for cm_p in self.compound_pairs:
                    cm_p.similar = {k: v for k, v in sorted(cm_p.similar.items(), key=lambda item: item[1])} 
                    #sorted_scores = sorted(cm_p.similar, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
                    #cm_p.similar = sorted_scores
                    cm_p.similar_sorted = True
        '''
        print("Begin running canbenchmark_ddi...")
        start = time.time()

        adrs = True

        approved_str = '-approved' if approved else ''
        ra_named = f'results_analysed_named/results_analysed_named-ddi_adr-{file_name}-{n}{approved_str}.tsv'
        ra_named_ndcg = f"results_analysed_named/results_analysed_named-ddi_adr-ndcg-{file_name}-{n}{approved_str}.tsv"
        ra = f'raw_results/raw_results-ddi_adr-{file_name}-{n}{approved_str}.csv'
        pwr = f'pairwise_results/pairwise_results-ddi_adr-{file_name}-{n}{approved_str}.csv'
        summ = f'summary-ddi_adr-{file_name}-{n}{approved_str}.tsv'
        benchmark_name = f"canbenchmark-ddi_adr-{file_name}-{n}{approved_str}"
        t_name = f"time-ddi_adr-{file_name}-{n}{approved_str}.txt"

        '''
        os.makedirs('results_analysed_named', exist_ok=True)
        os.makedirs('raw_results', exist_ok=True)
        os.makedirs('pairwise_results', exist_ok=True)
        os.makedirs(benchmark_name, exist_ok=True)
        if not os.path.exists('./results_analysed_named'):
            print("Directory 'results_analysed_named' does not exist, creating directory")
            #os.system('mkdir results_analysed_named')
            os.mkdir('results_analysed_named')
        if not os.path.exists('./raw_results'):
            print("Directory 'raw_results' does not exist, creating directory")
            #os.system('mkdir raw_results')
            os.mkdir('raw_results')

        ra_named = 'results_analysed_named/results_analysed_named_' + file_name + '-ddi_adr.tsv'
        ra = 'raw_results/raw_results_' + file_name + '-ddi_adr.csv'
        summ = 'summary_' + file_name + '-ddi_adr.tsv'
        '''

        ra_out = open(ra, 'w')
        pwr_out = open(pwr, 'w')

        def dcg(l,k):
            dcg = [((2**x)-1)/(math.log2(i+1)) for i,x in enumerate(l[:k],1)]
            return np.sum(dcg)

        def effect_type():
            if adrs:
                return 'ADR'
            else:
                return 'disease'

        def competitive_standard_bottom(sims, r):
            rank = 0
            for sim in sims:
                if sim[1] > r:
                    rank += 1.0
                else:
                    return rank
            return len(sims)

        def competitive_modified_bottom(sims, r):
            rank = 0
            for sim in sims:
                if sim[1] >= r:
                    rank += 1.0
                else:
                    return rank
            return len(sims)

        # Competitive modified ranking code
        def competitive_modified(sims, r):
            rank = 0
            for sim in sims:
                if sim[1] <= r:
                    rank += 1.0
                else:
                    return rank
            return len(sims)

        # Competitive standard ranking code
        def competitive_standard(sims, r):
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

        if adrs:
            #effects = self.adrs
            effects = [effect for effect in self.adrs if len(effect.compound_pairs)>=2]
            #effect_ids = [effect.id_ for effect in effects if len(effect.compound_pairs)>=2]
            effect_ids = [effect.id_ for effect in effects]
        else:
            effects = self.indications


        effects = [effect for effect in self.adrs if len(effect.compound_pairs) > 1]
        if approved:
            cp_lib = [str(cp.id_) for cp in self.compound_pairs if len(cp.adrs)>=1]
        else:
            cp_lib = [str(cp.id_) for cp in self.compound_pairs]

        def cont_metrics():
            all_v = []
            for cp in self.compound_pairs:
                # Create a sorted similar list on-the-fly for each compound pair
                cp_sorted = self.generate_similar_sigs_cp(cp, sort=True, ncpus=self.ncpus)
                # Iterate through the sorted list
                for cp_sim in cp_sorted:
                    cp_dist = cp_sorted[cp_sim]
                    if cp_dist != 0.0:
                        all_v.append(cp_dist)
            avl = len(all_v)
            all_v_sort = sorted(all_v)
            # for tuple 10, have to add the '-1' for index out of range reasons
            metrics = [(1, all_v_sort[int(avl/1000.0)]), (2, all_v_sort[int(avl/400.0)]), (3, all_v_sort[int(avl/200.0)]),
                       (4, all_v_sort[int(avl/100.0)]), (5, all_v_sort[int(avl/20.0)]), (6, all_v_sort[int(avl/10.0)]),
                       (7, all_v_sort[int(avl/5.0)]), (8, all_v_sort[int(avl/3.0)]), (9, all_v_sort[int(avl/2.0)]),
                       (10, all_v_sort[int(avl/1.0)-1])]
            return metrics

        x = (len(self.compound_pairs)) / 100.0  # changed this...no reason to use similar instead of compounds
        # had to change from 100.0 to 100.0001 because the int function
        # would chop off an additional value of 1 for some reason...
        if continuous:
            metrics = cont_metrics()
        else:
            metrics = [(1, 10), (2, 25), (3, 50), (4, 100), (5, int(x*100.0001)),
                       (6, int(x*1.0001)), (7, int(x*5.0001)), (8, int(x*10.0001)),
                       (9, int(x*50.0001)), (10, int(x*100.0001))]

        if continuous:
            ra_out.write("compound_id,{}_id,0.1%({:.3f}),0.25%({:.3f}),0.5%({:.3f}),"
                         "1%({:.3f}),5%({:.3f}),10%({:.3f}),20%({:.3f}),33%({:.3f}),"
                         "50%({:.3f}),100%({:.3f}),value\n".format(effect_type(), metrics[0][1], metrics[1][1],
                                                                     metrics[2][1], metrics[3][1], metrics[4][1],
                                                                     metrics[5][1], metrics[6][1], metrics[7][1],
                                                                     metrics[8][1], metrics[9][1]))
        else:
            ra_out.write(f"cmpd_pair_id,effect_id,top10,top25,top50,top100,"
                         f"top{len(self.compound_pairs)},top1%,top5%,top10%,top50%,top100%,rank\n")
        

        # Calculate all similar first
        # But do not populate compound_pair objects with similar
        #print("Calculating all compound pair distances...")
        '''
        signatures = [self.compound_pairs[i].sig for i in range(len(self.compound_pairs))]
        signatures = np.array(signatures)  # convert to numpy form
        # call pairwise_distances, speed up with custom RMSD function and parallelism
        if self.dist_metric == "rmsd":
            distance_matrix = pairwise_distances(signatures, metric=lambda u, v: np.sqrt(((u - v) ** 2).mean()), n_jobs=self.ncpus)
        elif self.dist_metric in ['cosine', 'correlation', 'euclidean', 'cityblock']:
            distance_matrix = pairwise_distances(signatures, metric=self.dist_metric, n_jobs=self.ncpus)
        for i in distance_matrix:
            x = list(zip(self.compound_pairs,i))
            sorted_distances = sorted(x, key=lambda j: j[1] if not math.isnan(j[1]) else 100000)
            sorted_distances.append(sorted_distances[1:]) 
        cp_ids = [self.compound_pairs[i].id_ for i in range(len(self.compound_pairs))]
        sorted_distances = {cp_ids[i]: sorted_distances[i] for i in range(len(sorted_distances))}
        '''
        #sorted_distances = {self.compound_pairs[i].id_: self.generate_similar_sigs_cp(self.compound_pairs[i], sort=True) for i in range(len(self.compound_pairs))}
        #print("Calculated distances.")

        # ZF - edit 02//06/23
        # Fixed to work with the sqlite
        print("  Calculating scores...")
        start_calc = time.time()
        os.makedirs(benchmark_name,exist_ok=True)
        if self.ncpus > 1:
            pool = mp.Pool(self.ncpus)
            #accs = pool.starmap_async(self.calc_accuracies_cmpd_pairs, [(effect.id_, effect.compound_pairs, metrics) for effect in effects]).get()
            pool.starmap_async(ind_accuracies_cmpd_pair, [(effect.id_, effect.compound_pairs, cp_lib, benchmark_name, metrics, approved, n, self.db_name) for effect in effects]).get()
            pool.close
            pool.join
        else:
            [ind_accuracies_cmpd_pair(effect.id_, effect.compound_pairs, cp_lib, benchmark_name, metrics, approved, n, self.db_name) for effect in effects]
            #accs = [self.calc_accuracies_cmpd_pairs_new(effect.id_, effect.compound_pairs, metrics) for effect in tqdm(effects)]
        t_calc = print_time(time.time() - start_calc)
        print("  Done calculating scores.")
        print(f"  Time to calculate scores: {t_calc}")
        with open(t_name,'w') as tw:
            tw.write(f"Number of CPUs: {self.ncpus}\n")
            tw.write(f"Time to calculate scores: {t_calc}\n")

        print("  Compiling and saving results...")
        # Average Indication Accuracy
        aia_accs = []
        # Pairwise Accuracy
        pwa_count = [0.0]*len(metrics)
        pwa_tot = 0
        top_pairwise = [0.0] * 10
        # Coverage
        cov_count = [0]*len(metrics)
        # NDCG
        ndcg = {}
        for m in metrics:
            ndcg[m[0]] = {}
        # Results_analysed_named
        #for effect_id in ranks.keys():
        #for effect in effects:
        #    effect_id = effect.id_
        #    print(effect_id)

        pd.options.display.float_format='{:.3f}'.format

        #for effect in tqdm(effects):
        pbar = tqdm(range(len(effects))) if self.pbar else range(len(effects))
        for i in pbar:
            effect = effects[i]
            effect_id = effect.id_
            pbar.set_description(f"    {effect_id}")
            db_benchmark = create_engine(f'sqlite:///{benchmark_name}/{effect_id}.db')
            df_ia = pd.read_sql(f"SELECT * FROM ia_results",db_benchmark)
            df_pa = pd.read_sql(f"SELECT * FROM pa_results",db_benchmark)
            #effect = self.get_adr(effect_id)
            count = [0.0]*len(metrics)
            tot = len(effect.compound_pairs)
            effect_dct[(effect, tot)] = {}
            ndcg_l = {}
            for m in metrics:
                effect_dct[(effect, tot)][m] = 0.0
                ndcg_l[m[0]] = []
            #pwa_tot += tot
            c_ideal = [0]*len(cp_lib)
            c_ideal[0] = 1
            #for cp_query in ranks[effect_id].keys():
            for cp_idx in df_ia.index:
                rank = int(df_ia.loc[cp_idx,'rank'])
                c_rank = [0]*len(cp_lib)
                c_rank[rank-1] = 1
                # Check ranks against each top metric
                for idx,m in enumerate(metrics):
                    #if ranks[effect_id][cp_query][1] <= m[1]:
                    if rank <= m[1]:
                        effect_dct[(effect, tot)][m] += 1.0
                        count[idx]+=1.0
                        #pwa_count[idx]+=1.0
                    # NDCG
                    ndcg_l[m[0]].append(dcg(c_rank, int(m[1])) / dcg(c_ideal, int(m[1])))
                '''
                # pairwise results
                if str(df_benchmark.iloc[cp_idx,2]) == '1':
                #if s[2] == '1':
                    top_pairwise[0] += 1.0
                if str(df_benchmark.iloc[cp_idx,3]) == '1':
                #if s[3] == '1':
                    top_pairwise[1] += 1.0
                if str(df_benchmark.iloc[cp_idx,4]) == '1':
                #if s[4] == '1':
                    top_pairwise[2] += 1.0
                if str(df_benchmark.iloc[cp_idx,5]) == '1':
                #if s[5] == '1':
                    top_pairwise[3] += 1.0
                if str(df_benchmark.iloc[cp_idx,6]) == '1':
                #if s[6] == '1':
                    top_pairwise[4] += 1.0
                if str(df_benchmark.iloc[cp_idx,7]) == '1':
                #if s[7] == '1':
                    top_pairwise[5] += 1.0
                if str(df_benchmark.iloc[cp_idx,8]) == '1':
                #if s[8] == '1':
                    top_pairwise[6] += 1.0
                if str(df_benchmark.iloc[cp_idx,9]) == '1':
                #if s[9] == '1':
                    top_pairwise[7] += 1.0
                if str(df_benchmark.iloc[cp_idx,10]) == '1':
                #if s[10] == '1':
                    top_pairwise[8] += 1.0
                if str(df_benchmark.iloc[cp_idx,11]) == '1':
                #if s[11] == '1':
                    top_pairwise[9] += 1.0
                sj = ','.join(df_benchmark.loc[cp_idx,:].values.tolist())
                sj += '\n'
                ra_out.write(sj)
                '''
                ra_out.write(','.join(df_ia.loc[cp_idx,:].values.tolist()) + '\n')

            # NDCG
            for m in metrics:
                ndcg[m[0]][effect] = np.mean(ndcg_l[m[0]])

            count = [(i/tot)*100.0 for i in count]
            for idx,i in enumerate(count):
                if i>0.0:
                    cov_count[idx]+=1
            aia_accs.append(count)
            #print(effect_id,effect.name,count)
        ra_out.close()

        # Pairwise accuracy
        pwa_tot += len(df_pa)
        for c_idx in df_pa.index:
            rank = int(df_pa.loc[c_idx, 'rank'])
            for idx, m in enumerate(metrics):
                # if ranks[effect_id][cp_query][1] <= m[1]:
                if rank <= m[1]:
                    pwa_count[idx] += 1.0
            pwr_out.write(','.join(df_pa.loc[c_idx, :].values.tolist()) + '\n')
        pa = [(i/pwa_tot)*100.0 for i in pwa_count]

        # NDCG - results_analysed_named
        with open(ra_named_ndcg, 'w') as o:
            o.write(
                f"effect_id\tcmpd_pairs_per_effect\ttop10\ttop25\ttop50\ttop100\ttop{len(cp_lib)}\ttop1%\ttop5%\ttop10%\ttop50%\ttop100%\teffect_name\n")
            for effect in effects:
                o.write(f"{effect.id_}\t{len(effect.compounds)}")
                for m in metrics:
                    o.write(f"\t{ndcg[m[0]][effect]:.3f}")
                o.write(f"\t{effect.name}\n")

        # Create average NDCG
        nndcg = [0.0] * 10
        for idx, m in enumerate(metrics):
            l = []
            for effect in effects:
                l.append(ndcg[m[0]][effect])
            nndcg[idx] = np.mean(l)

        # Indication coverage
        cov = [0] * 10
        for effect, c in list(self.accuracies.keys()):
            accs = self.accuracies[effect, c]
            for m_i in range(len(metrics)):
                v = accs[metrics[m_i]]
                if v > 0.0:
                    cov[m_i] += 1
        cov = map(int, cov)

        #top_metrics = [str(j) for i,j in metrics]
        #aia_accs = [str(sum(sub_list) / len(sub_list)) for sub_list in zip(*aia_accs)]
        #pwa_accs = [str((i/pwa_tot)*100.0) for i in pwa_count]
        #cov_count = [str(i) for i in cov_count]

        #print("\t".join(top_metrics))
        #print("\t".join(aia_accs))
        #print("\t".join(pwa_accs))
        #print("\t".join(cov_count))
        # Fix later
        self.accuracies = effect_dct
        final_accs = self.results_analysed(ra_named, metrics, effect_type())
        #ss = sorted(ss, key=lambda xx: xx[0])
        #ss = sorted(ss, key=lambda xx: int(xx[0]))

        # Create average indication accuracy list in percent
        ia = []
        for m in metrics:
            ia.append(final_accs[m] * 100.0)
        # Create average pairwise accuracy list in percent
        #pa = [(x * 100.0 / pwa_tot) for x in top_pairwise]

        # Control calculation
        control_aia = []
        for n in range(len(metrics)):
            p = hypergeom.pmf(1, len(cp_lib), 1, metrics[n][1])
            control_aia.append(p * 100.0)

        print("  Done compiling and saving results.")

        '''
        if continuous:
            headers = ['0.1%ile', '.25%ile', '0.5%ile', '1%ile', '5%ile',
                       '10%ile', '20%ile', '33%ile', '50%ile', '100%ile']
        else:
            headers = ['top10', 'top25', 'top50', 'top100', f'top{len(cp_lib)}',
                       'top1%', 'top5%', 'top10%', 'top50%', 'top100%']
                               
        # Append 3 lists to df and write to file
        with open(summ, 'w') as sf:
            sf.write("\t" + '\t'.join(headers) + '\n')
            iast = "\t".join(map(str, [format(x, ".3f") for x in ia]))
            pwst = "\t".join(map(str, [format(x, ".3f") for x in pa]))
            icst = "\t".join(map(str, cov)) + '\n'
            sf.write('aia\t{}\napa\t{}\nic\t{}\n'.format(iast, pwst, icst))
        '''

        # pretty print the average indication accuracies
        if continuous:
            headers = ['0.1%ile', '.25%ile', '0.5%ile', '1%ile', '5%ile',
                       '10%ile', '20%ile', '33%ile', '50%ile', '100%ile']
        else:
            headers = ['top10', 'top25', 'top50', 'top100', f'top{len(cp_lib)}',
                       'top1%', 'top5%', 'top10%', 'top50%', 'top100%']
        df_summ = pd.DataFrame(list(zip(ia, control_aia, pa, cov, nndcg)), columns=['nAIA','control-nAIA','PA','IC','nNDCG'], index=[headers])
        print("\nSummary")
        print(df_summ.T)
        print()
        df_summ.T.to_csv(summ, float_format='%.3f', sep='\t')
        t_tot = print_time(time.time()-start)
        print("Done running canbenchmark_ddi.")
        print(f"Total time to run canbenchmark_ddi: {t_tot}\n")
        with open(t_name,'a') as tw:
            tw.write(f"Total time to run canbenchmark_ddi: {t_tot}")


    def calc_accuracies_cmpd_pairs(self, effect_id, cmpd_pairs, metrics):
        if os.path.exists(f'ddi_benchmark/{effect_id}.db'):
            return
        try: 
            #Connecting to sqlite
            conn = sqlite3.connect(f'ddi_benchmark/{effect_id}.db')
            #Creating a cursor object using the cursor() method
            cursor = conn.cursor()
            #Droping dists table if already exists
            cursor.execute("DROP TABLE results")
            #print("results table dropped.")
            #Commit your changes in the database
            conn.commit()
            #Closing the connection
            conn.close()
        except:
            #print("dists table does not exist.")
            pass


        #conn = 'sqlite://ddi.db'
        conn = f'sqlite://{self.db_name}'
        #db = create_engine('sqlite:///ddi.db')
        db_benchmark = create_engine(f'sqlite:///ddi_benchmark/{effect_id}.db')
        #df_cp = pd.read_sql(f"SELECT * FROM 'adrs'",db)
        benchmark_cols = ['cmpd_pair_ids','effect_id','top10','top25','top50','top100','topAll','top1%','top5%','top10%','top50%','top100%','hit_cmpd_pair_ids','rank']

        #effect = self.get_adr(effect_id)
        cmpd_pairs = [str(c) for c in cmpd_pairs]
        
        #df_cp = pd.read_sql(f"SELECT cmpd_pair_ids FROM 'adrs' WHERE id='{effect_id}'",db)
        #cmpd_pairs = ast.literal_eval(df_cp.loc[0,'cmpd_pair_ids'])
        #cmpd_pairs = [str(c) for c in cmpd_pairs]
        #count = len(cmpd_pairs)
        #print(f"{effect_id} has {count} cmpd pairs")

        #count = len(effect.compound_pairs)
        #if count < 2:
        #    return
        '''
        if not adrs:
            if self.indication_pathways:
                if len(effect.pathways) == 0:
                    print('No associated pathways for {}, skipping'.format(effect.id_))
                    continue
                elif len(effect.pathways) < 1:
                    #print('Less than 5 associated pathways for {}, skipping'.format(effect.id_))
                    continue
        '''
        #c_per_effect += count
        
        #effect_dct[(effect, count)] = {}
        #for m in metrics:
        #    effect_dct[(effect, count)][m] = 0.0
       
        accs = [0.0]*len(metrics)
        ss = []
        
        #print(effect_id)
        #df_dists = pd.read_sql(f"SELECT {str(cmpd_pairs).replace('[','').replace(']','')} FROM ranks WHERE id IN {tuple(cmpd_pairs)}",db)
        df_dists = pl.read_sql(f"SELECT * FROM dists WHERE id IN {tuple(cmpd_pairs)}", conn)
        #df_dists = pd.read_sql(f"SELECT * FROM dists WHERE id IN {tuple(cmpd_pairs)}",db)
        #all_dists = {c: ast.literal_eval(df_dists.loc[(df_dists['id']==str(c)),'dists'].values.tolist()[0]) for c in cmpd_pairs}
        for c in cmpd_pairs:
            #c_sorted = self.generate_similar_sigs_cp(c, sort=True)
            #df_dists = pd.read_sql(f"SELECT dists FROM dists WHERE id='{c}'",db)
            #c_sorted = [[key, value] for key, value in ast.literal_eval(df_dists.loc[0,'dists']).items()]
            #c_sorted = ast.literal_eval(df_dists.loc[0,'dists'])
            
            #c_sorted = ast.literal_eval(df_dists.loc[(df_dists['id']==str(c)),'dists'].values.tolist()[0])
            #c_sorted = all_dists[c]
            
            c_sorted = json.loads(df_dists.filter(pl.col('id')==str(c)).select(['dists']).item().replace("'",'"'))
            c_sorted = pl.DataFrame(c_sorted)
            #c_df = c_sorted.transpose().with_columns(pl.Series(name='ids',values=c_sorted.columns)).sort('column_1')
            c_df = c_sorted.transpose().with_columns(pl.Series(name='ids',values=c_sorted.columns)).sort('column_0')
            c_df = c_df.with_columns(pl.Series(name='ranks',values=list(range(1,len(c_sorted.columns)+1))))

            # Polars way of doing it
            best_value = c_df.filter(pl.col('ids').is_in(cmpd_pairs))[0,'ranks']
            best_cp = c_df.filter(pl.col('ids').is_in(cmpd_pairs))[0,'ids']

            s = [c, effect_id]
            for x in metrics:
                if best_value <= x[1]:
                    # Comment out for now to see if we can just use s to calculate benchmarks
                    #effect_dct[effect][x] += 1.0
                    s.append('1')
                else:
                    s.append('0')
            s.append(str(best_cp))
            s.append(str(int(best_value)))
            ss.append(s)
        df_temp = pd.DataFrame(ss,columns=benchmark_cols)
        #df_temp = pd.DataFrame.from_dict(effect_dct,orient='index')
        #df_temp.index.names = ['id']
        #df_temp.rename(columns={benchmark_cols},inplace=True)
        df_temp.to_sql('results', db_benchmark, if_exists='append', index=False)
        #df_temp = None
        return

        '''
            for cs in c_sorted:
                if str(cs[0]) not in cmpd_pairs:
                    continue
                value = cs[2]
                s = [c, effect_id]
                #s = [str(c.index), effect.name]
                for x in range(len(metrics)):
                    if value <= metrics[x][1]:
                        accs[x] += 1.0
                        #effect_dct[(effect, count)][x] += 1.0
                        s.append('1')
                    else:
                        s.append('0')
                s.append(str(int(value)))
                ss.append(s)
                break
        '''
            
            #best_value = 10000000000
            #for c2 in cmpd_pairs:
            #    if str(c2) == str(c):
            #        continue
            #    #c2 = tuple(map(int, c2.replace("(",'').replace(")",'').split(', ')))
            #    temp_value = c_sorted[c2][1]
            #    if temp_value < best_value:
            #        best_value = temp_value
            #        #cp2 = cp

            
            #s = [c, effect_id]
            #for x in range(len(metrics)):
            #    if best_value <= metrics[x][1]:
            #        accs[x] += 1.0
            #        #effect_dct[(effect, count)][x] += 1.0
            #        s.append('1')
            #    else:
            #        s.append('0')
            #s.append(str(int(best_value)))
            #ss.append(s)

        '''
            for cs in c_sorted:
                if str(cs[0]) not in cmpd_pairs:
                    continue

                value = cs[2]
                s = [c, effect_id]
                #s = [str(c.index), effect.name]
                for x in range(len(metrics)):
                    if value <= metrics[x][1]:
                        accs[x] += 1.0
                        #effect_dct[(effect, count)][x] += 1.0
                        s.append('1')
                    else:
                        s.append('0')
                s.append(str(int(value)))
                ss.append(s)
                break
        '''

        ''' 
        # This has been updated to work with compound_pairs
        for c in effect.compound_pairs:
            c_sorted = self.generate_similar_sigs_cp(c, sort=True, ncpus=1)
        '''
        '''
            if bottom:
                if ranking == 'modified':
                    rank_sorted = abs(stats.rankdata(list(zip(*c_sorted))[1], method='min') - len(c_sorted))+1
                elif ranking == 'standard':
                    rank_sorted = abs(stats.rankdata(list(zip(*c_sorted))[1], method='max') - len(c_sorted))+1
                elif ranking == 'ordinal':
                    rank_sorted = abs(stats.rankdata(list(zip(*c_sorted))[1], method='ordinal') - len(c_sorted))+1        
                else:
                    print("Ranking function {} is incorrect.".format(ranking))
                    exit()
        '''
        '''
            if ranking == 'modified':
                rank_sorted = stats.rankdata(list(zip(*c_sorted))[1], method='min')
            elif ranking == 'standard':
                rank_sorted = stats.rankdata(list(zip(*c_sorted))[1], method='max')
            elif ranking == 'ordinal':
                rank_sorted = stats.rankdata(list(zip(*c_sorted))[1], method='ordinal')
            else:
                print("Ranking function {} is incorrect.".format(ranking))
                exit()

            for idx, cs in enumerate(c_sorted):
                c_sim = cs[0]
                c_dist = cs[1]
                if adrs:
                    if effect not in c_sim.adrs:
                        continue
                else:
                    if effect not in c_sim.indications:
                        continue

                s.append((effect_id, c.id_, c_sim.id_, rank_sorted[idx], c_dist))
                break
        '''
        '''
                value = 0.0
                if continuous:
                    value = c_dist
                else:
                    value = rank_sorted[idx]

                if adrs:
                    s = [str(c.index), effect.name]
                else:
                    s = [str(c.index), effect.id_]
                for x in range(len(metrics)):
                    if value <= metrics[x][1]:
                        accs[x] += 1.0
                        #effect_dct[(effect, count)][x] += 1.0
                        s.append('1')
                    else:
                        s.append('0')
                if continuous:
                    s.append(str(value))
                else:
                    s.append(str(int(value)))
                ss.append(s)
                break
        return [effect, count, accs, ss]
        '''
        #df_dists = None 
        #db = None 
        #c_sorted=None
        #return ss
        #return [ss, accs]


    # Indication accuracy metric w/ new protocol
    # ---
    # Leave one out method where a consensus list will be generated
    # using all but 1 approved drug (reverse of classic canbenchmark)
    # Accuracy will be determined by the presence or absence of the 
    # left out drug for each iteration and the average will be 
    # calculated for each indicaiton
    def calc_accuracies_cmpd_pairs_new(self, effect_id, cmpd_pairs, metrics):
        if os.path.exists(f'ddi_benchmark/{effect_id}.db'):
            return
        try: 
            #Connecting to sqlite
            conn = sqlite3.connect(f'ddi_benchmark/{effect_id}.db')
            #Creating a cursor object using the cursor() method
            cursor = conn.cursor()
            #Droping dists table if already exists
            cursor.execute("DROP TABLE results")
            #print("results table dropped.")
            #Commit your changes in the database
            conn.commit()
            #Closing the connection
            conn.close()
        except:
            #print("dists table does not exist.")
            pass


        #conn = 'sqlite://ddi.db'
        conn = f'sqlite://{self.db_name}'
        #db = create_engine('sqlite:///ddi.db')
        db_benchmark = create_engine(f'sqlite:///ddi_benchmark/{effect_id}.db')
        #df_cp = pd.read_sql(f"SELECT * FROM 'adrs'",db)
        benchmark_cols = ['cmpd_pair_ids','effect_id','top10','top25','top50','top100','topAll','top1%','top5%','top10%','top50%','top100%','hit_cmpd_pair_ids','rank']

        #effect = self.get_adr(effect_id)
        cmpd_pairs = [str(c) for c in cmpd_pairs]
        
        #df_cp = pd.read_sql(f"SELECT cmpd_pair_ids FROM 'adrs' WHERE id='{effect_id}'",db)
        #cmpd_pairs = ast.literal_eval(df_cp.loc[0,'cmpd_pair_ids'])
        #cmpd_pairs = [str(c) for c in cmpd_pairs]
        #count = len(cmpd_pairs)
        #print(f"{effect_id} has {count} cmpd pairs")

        #count = len(effect.compound_pairs)
        #if count < 2:
        #    return
        '''
        if not adrs:
            if self.indication_pathways:
                if len(effect.pathways) == 0:
                    print('No associated pathways for {}, skipping'.format(effect.id_))
                    continue
                elif len(effect.pathways) < 1:
                    #print('Less than 5 associated pathways for {}, skipping'.format(effect.id_))
                    continue
        '''
        #c_per_effect += count
        
        #effect_dct[(effect, count)] = {}
        #for m in metrics:
        #    effect_dct[(effect, count)][m] = 0.0
       
        accs = [0.0]*len(metrics)
        ss = []
        
        #print(effect_id)
        #df_dists = pd.read_sql(f"SELECT {str(cmpd_pairs).replace('[','').replace(']','')} FROM ranks WHERE id IN {tuple(cmpd_pairs)}",db)
        df_dists = pl.read_database(query=f"SELECT * FROM dists WHERE id IN {tuple(cmpd_pairs)}", connection=conn)

        #df_dists = pd.read_sql(f"SELECT * FROM dists WHERE id IN {tuple(cmpd_pairs)}",db)
        #all_dists = {c: ast.literal_eval(df_dists.loc[(df_dists['id']==str(c)),'dists'].values.tolist()[0]) for c in cmpd_pairs}
        for c_loo in cmpd_pairs:
            c_df = pl.read_database(query=f"SELECT id FROM dists", connection=conn)
            for c in cmpd_pairs:
                if c_loo == c:
                    continue
                c_sorted = json.loads(df_dists.filter(pl.col('id')==str(c)).select(['dists']).item().replace("'",'"'))
                c_sorted = pl.DataFrame(c_sorted)
                #c_df = c_sorted.transpose().with_columns(pl.Series(name='ids',values=c_sorted.columns)).sort('column_1')
                c_df_temp = c_sorted.transpose().with_columns(pl.Series(name='id',values=c_sorted.columns)).sort('column_0')
                c_df_temp = c_df_temp.rename({'column_0':str(c)})
                c_df = c_df.join(c_df_temp, on='id', how='left')
            c_df = c_df.with_columns(sum=pl.sum_horizontal(c_df.columns[1:]))
            c_df = c_df.sort('sum')
            c_df = c_df.with_columns(pl.Series(name='rank',values=list(range(1,len(c_df)+1))))

            # Polars way of doing it
            best_rank = c_df.filter(pl.col('id')==str(c_loo))[0,'rank']
            best_dist = c_df.filter(pl.col('id')==str(c_loo))[0,'sum']

            s = [c_loo, effect_id]
            for x in metrics:
                if best_rank <= x[1]:
                    # Comment out for now to see if we can just use s to calculate benchmarks
                    #effect_dct[effect][x] += 1.0
                    s.append('1')
                else:
                    s.append('0')
            s.append(str(float(best_dist)))
            s.append(str(int(best_rank)))
            ss.append(s)
        df_temp = pd.DataFrame(ss,columns=benchmark_cols)
        #df_temp = pd.DataFrame.from_dict(effect_dct,orient='index')
        #df_temp.index.names = ['id']
        #df_temp.rename(columns={benchmark_cols},inplace=True)
        df_temp.to_sql('results', db_benchmark, if_exists='append', index=False)
        #df_temp = None
        return

        '''
            for cs in c_sorted:
                if str(cs[0]) not in cmpd_pairs:
                    continue
                value = cs[2]
                s = [c, effect_id]
                #s = [str(c.index), effect.name]
                for x in range(len(metrics)):
                    if value <= metrics[x][1]:
                        accs[x] += 1.0
                        #effect_dct[(effect, count)][x] += 1.0
                        s.append('1')
                    else:
                        s.append('0')
                s.append(str(int(value)))
                ss.append(s)
                break
        '''
            
            #best_value = 10000000000
            #for c2 in cmpd_pairs:
            #    if str(c2) == str(c):
            #        continue
            #    #c2 = tuple(map(int, c2.replace("(",'').replace(")",'').split(', ')))
            #    temp_value = c_sorted[c2][1]
            #    if temp_value < best_value:
            #        best_value = temp_value
            #        #cp2 = cp

            
            #s = [c, effect_id]
            #for x in range(len(metrics)):
            #    if best_value <= metrics[x][1]:
            #        accs[x] += 1.0
            #        #effect_dct[(effect, count)][x] += 1.0
            #        s.append('1')
            #    else:
            #        s.append('0')
            #s.append(str(int(best_value)))
            #ss.append(s)

        '''
            for cs in c_sorted:
                if str(cs[0]) not in cmpd_pairs:
                    continue

                value = cs[2]
                s = [c, effect_id]
                #s = [str(c.index), effect.name]
                for x in range(len(metrics)):
                    if value <= metrics[x][1]:
                        accs[x] += 1.0
                        #effect_dct[(effect, count)][x] += 1.0
                        s.append('1')
                    else:
                        s.append('0')
                s.append(str(int(value)))
                ss.append(s)
                break
        '''

        ''' 
        # This has been updated to work with compound_pairs
        for c in effect.compound_pairs:
            c_sorted = self.generate_similar_sigs_cp(c, sort=True, ncpus=1)
        '''
        '''
            if bottom:
                if ranking == 'modified':
                    rank_sorted = abs(stats.rankdata(list(zip(*c_sorted))[1], method='min') - len(c_sorted))+1
                elif ranking == 'standard':
                    rank_sorted = abs(stats.rankdata(list(zip(*c_sorted))[1], method='max') - len(c_sorted))+1
                elif ranking == 'ordinal':
                    rank_sorted = abs(stats.rankdata(list(zip(*c_sorted))[1], method='ordinal') - len(c_sorted))+1        
                else:
                    print("Ranking function {} is incorrect.".format(ranking))
                    exit()
        '''
        '''
            if ranking == 'modified':
                rank_sorted = stats.rankdata(list(zip(*c_sorted))[1], method='min')
            elif ranking == 'standard':
                rank_sorted = stats.rankdata(list(zip(*c_sorted))[1], method='max')
            elif ranking == 'ordinal':
                rank_sorted = stats.rankdata(list(zip(*c_sorted))[1], method='ordinal')
            else:
                print("Ranking function {} is incorrect.".format(ranking))
                exit()

            for idx, cs in enumerate(c_sorted):
                c_sim = cs[0]
                c_dist = cs[1]
                if adrs:
                    if effect not in c_sim.adrs:
                        continue
                else:
                    if effect not in c_sim.indications:
                        continue

                s.append((effect_id, c.id_, c_sim.id_, rank_sorted[idx], c_dist))
                break
        '''
        '''
                value = 0.0
                if continuous:
                    value = c_dist
                else:
                    value = rank_sorted[idx]

                if adrs:
                    s = [str(c.index), effect.name]
                else:
                    s = [str(c.index), effect.id_]
                for x in range(len(metrics)):
                    if value <= metrics[x][1]:
                        accs[x] += 1.0
                        #effect_dct[(effect, count)][x] += 1.0
                        s.append('1')
                    else:
                        s.append('0')
                if continuous:
                    s.append(str(value))
                else:
                    s.append(str(int(value)))
                ss.append(s)
                break
        return [effect, count, accs, ss]
        '''
        #df_dists = None 
        #db = None 
        #c_sorted=None
        #return ss
        #return [ss, accs]

    def ml(self, method='rf', effect=None, benchmark=False, adrs=False, predict=[], threshold=0.5,
           negative='random', seed=42, out=''):
        """!
        Create an ML classifier for a specified indication to make drug-disease predictions or all inds for benchmarking

        @param method str: type of machine learning algorithm to use ('rf' or 'log')
        @param effect Indication or ADR: provide a specific Indication or ADR object to train a classifer
        @param benchmark bool: benchmark the ML pipeline by training a classifier with LOOCV for each Indication or ADR
        @param adrs bool: if the models are trained with ADRs instead of Indications
        @param predict list: provide a list of Compound objects to classify with the model (only used in
        combination with effect=Indication/ADR object)
        @param threshold float: decision threshold for positive vs negative classification
        @param negative str: choose random negative samples (default) or 'inverse' for most opposite signatures
        @param seed int: choose a seed for reproducibility
        @param out str: file name extension for the output of benchmark (note: must have benchmark=True)
        @return Returns None
        """
        assert method not in ('1csvm', 'svm'), 'SVMS are currently unsupported by this version of cando.py. Please choose "log" or "rf"'
        #if method in ['1csvm', 'svm']:
        #    print('SVMs are currently unsupported by this version of cando.py. Please choose "log" or "rf" - quitting.')
        #    quit()

        if out:
            if not os.path.exists('./raw_results/'):
                os.mkdir('raw_results')
                #os.system('mkdir raw_results')
            if not os.path.exists('./results_analysed_named/'):
                os.mkdir('results_analysed_named')
                #os.system('mkdir results_analysed_named')

        paired_negs = {}

        # gather approved compound signatures for training
        def split_cs(efct, cmpd=None):
            mtrx = []
            for c in efct.compounds:
                cm = self.get_compound(c)
                if cmpd:
                    if cm.id_ == cmpd.id_:
                        continue
                if self.indication_proteins:
                    if len(efct.proteins) >= 3:
                        eps = []
                        for ep in efct.proteins:
                            ep_index = self.protein_id_to_index[ep.id_]
                            eps.append(cm.sig[ep_index])
                        mtrx.append(eps)
                else:
                    mtrx.append(cm.sig)
            return mtrx, [1] * len(mtrx)

        def choose_negatives(efct, neg_set=negative, s=None, hold_out=None, avoid=[], test=None):
            if neg_set == 'inverse':
                assert self.compute_distance or self.read_dists, 'Please compute all compound-compound distances before using inverse_negatives().\n'\
                          'Re-run with "compute_distance=True" or read in pre-computed distance file "read_dists="'\
                          'in the CANDO object instantiation -- quitting.'
                #if not self.compute_distance and not self.read_dists:
                #   print('Please compute all compound-compound distances before using inverse_negatives().\n'
                #         'Re-run with "compute_distance=True" or read in pre-computed distance file "read_dists="'
                #         'in the CANDO object instantiation -- quitting.')
                #   quit()
            negatives = []
            used = avoid

            def pick_first_last(cmpd, s):
                if neg_set == 'inverse':
                    r = int(len(self.compounds) / 2)
                    shuffled = [cx[0].id_ for cx in cmpd.similar][::-1][0:r]
                else:
                    shuffled = [cx.id_ for cx in self.compounds]
                if s:
                    random.seed(s)
                    random.shuffle(shuffled)
                else:
                    s = random.randint(0, len(self.compounds) - 1)
                    random.seed(s)
                    random.shuffle(shuffled)
                for si in range(len(shuffled)):
                    n = shuffled[si]
                    if n in used:
                        continue
                    inv = self.get_compound(n)
                    if inv not in efct.compounds:
                        if n not in used:
                            paired_negs[cmpd] = inv
                            return inv

            if test:
                inv = pick_first_last(c, s)
                return inv

            for cmpd in efct.compounds:
                ce = self.get_compound(cmpd)
                if hold_out:
                    if ce.id_ == hold_out.id_:
                        continue
                inv = pick_first_last(ce, s)
                if self.indication_proteins:
                    if len(efct.proteins) >= 3:
                        eps = []
                        for ep in efct.proteins:
                            ep_index = self.protein_id_to_index[ep.id_]
                            eps.append(inv.sig[ep_index])
                        negatives.append(eps)
                else:
                    negatives.append(inv.sig)
                used.append(inv.id_)
            return negatives, [0] * len(negatives), used

        def model(meth, samples, labels, params=None, seed=None):
            assert meth in ('rf', 'svm', '1csvm', 'log'),\
                   "Please enter valid machine learning method ('rf', '1csvm', 'log', or 'svm')"
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
            #else:
            #   print("Please enter valid machine learning method ('rf', '1csvm', 'log', or 'svm')")
            #   quit()

        if benchmark:
            if adrs:
                effects = sorted(self.adrs, key=lambda x: (len(x.compounds), x.id_))[::-1]
            else:
                effects = sorted(self.indications, key=lambda x: (len(x.compounds), x.id_))[::-1]
            if out:
                frr = open('./raw_results/raw_results_ml_{}'.format(out), 'w', encoding="utf8")
                frr.write('Compound,Effect,Prob,Neg,Neg_prob\n')
                fran = open('./results_analysed_named/results_analysed_named_ml_{}'.format(out), 'w', encoding="utf8")
                fsum = open('summary_ml-{}'.format(out), 'w', encoding="utf8")
        else:
            assert len(effect.compounds) > 0, 'No compounds associated with {} ({})'.format(effect.name, effect.id_)
            assert not self.indication_proteins or len(effect.proteins) > 2, \
                   'Less than 3 proteins associated with {} ({})'.format(effect.name, effect.id_)
            #if len(effect.compounds) < 1:
            #    print('No compounds associated with {} ({}), quitting.'.format(effect.name, effect.id_))
            #   quit()
            #elif self.indication_proteins and len(effect.proteins) <= 2:
            #   print('Less than 3 proteins associated with {} ({}), quitting.'.format(effect.name, effect.id_))
            effects = [effect]

        rf_scores = []
        for e in effects:
            if len(e.compounds) < 2:
                continue
            if self.indication_proteins:
                if not len(e.proteins) >= 3:
                    continue
            tp_fn = [0, 0]
            fp_tn = [0, 0]
            for cmpd in e.compounds:
                c = self.get_compound(cmpd)
                pos = split_cs(e, cmpd=c)
                negs = choose_negatives(e, s=seed, hold_out=c, avoid=[])
                already_used = negs[2]
                train_samples = np.array(pos[0] + negs[0])
                train_labels = np.array(pos[1] + negs[1])
                mdl = model(method, train_samples, train_labels, seed=seed)
                test_neg = choose_negatives(e, s=seed, avoid=already_used, test=c)
                if self.indication_proteins:
                    eps_pos = []
                    eps_neg = []
                    for ep in e.proteins:
                        ep_index = self.protein_id_to_index[ep.id_]
                        eps_pos.append(c.sig[ep_index])
                        eps_neg.append(test_neg.sig[ep_index])
                    pred = mdl.predict_proba(np.array([eps_pos]))
                    pred_neg = mdl.predict_proba(np.array([eps_neg]))
                else:
                    pred = mdl.predict_proba(np.array([c.sig]))
                    pred_neg = mdl.predict_proba(np.array([test_neg.sig]))
                pos_class = list(mdl.classes_).index(1)

                if pred[0][pos_class] > threshold:
                    tp_fn[0] += 1
                else:
                    tp_fn[1] += 1
                if pred_neg[0][pos_class] > threshold:
                    fp_tn[0] += 1
                else:
                    fp_tn[1] += 1
                if benchmark and out:
                    frr.write('{},{},{},{},{}\n'.format(c.id_, e.id_, pred[0][pos_class],
                                                        test_neg.id_, pred_neg[0][pos_class]))

            # predict whether query drugs are associated with this indication
            if predict:
                print('Indication: {}'.format(e.name))
                print('Leave-one-out cross validation: TP={}, FP={}, FN={}, TN={}, Acc={:0.3f}'.format(
                    tp_fn[0], fp_tn[0], tp_fn[1], fp_tn[1], 100 * ((tp_fn[0]+fp_tn[1]) / (float(len(e.compounds))*2))))
                negs = choose_negatives(e, s=seed)
                pos = split_cs(e)
                train_samples = np.array(pos[0] + negs[0])
                train_labels = np.array(pos[1] + negs[1])
                mdl = model(method, train_samples, train_labels, seed=seed)
                print('\tCompound\tProb')
                for c in predict:
                    inv = choose_negatives(effect, s=seed, test=c, avoid=negs[2])
                    if self.indication_proteins:
                        eps_pos = []
                        eps_neg = []
                        for ep in e.proteins:
                            ep_index = self.protein_id_to_index[ep.id_]
                            eps_pos.append(c.sig[ep_index])
                            eps_neg.append(test_neg.sig[ep_index])
                        pred = mdl.predict_proba(np.array([eps_pos]))
                        pred_neg = mdl.predict_proba(np.array([test_neg.sig]))
                    else:
                        pred = mdl.predict_proba(np.array([c.sig]))
                        pred_inv = mdl.predict_proba(np.array([inv.sig]))
                    pos_class = list(mdl.classes_).index(1)

                    print('\t{}\t{:0.3f}'.format(c.name, pred[0][pos_class]))
                    #print('\t{}\t{:0.3f}\t(random negative of {})'.format(inv.name, pred_inv[0][pos_class], c.name))

            # append loocv results to combined list
            rf_scores.append((e, tp_fn, fp_tn))

        sm = [0, 0, 0, 0]
        if benchmark:
            for rf_score in rf_scores:
                efct = rf_score[0]
                tfp = rf_score[1]
                ffp = rf_score[2]
                acc = (tfp[0] + ffp[1]) / (float(len(efct.compounds) * 2))
                sm[0] += len(efct.compounds)
                sm[1] += acc
                sm[2] += (acc * len(efct.compounds))
                if acc > 0.5:
                    sm[3] += 1
                if out:
                    fran.write('{}\t{}\t{}\t{}\t{:0.3f}\t{}\n'.format(efct.id_, len(efct.compounds),
                                                                      tfp[0], tfp[1], 100 * acc, efct.name))
            if out:
                fsum.write('aia\t{:0.3f}\n'.format(100 * (sm[1]/len(rf_scores))))
                fsum.write('apa\t{:0.3f}\n'.format(100 * (sm[2] / sm[0])))
                fsum.write('ic\t{}\n'.format(sm[3]))

            print('aia\t{:0.3f}'.format(100 * (sm[1]/len(rf_scores))))
            print('apa\t{:0.3f}'.format(100 * (sm[2] / sm[0])))
            print('ic\t{}'.format(sm[3]))
        return

    def raw_results_roc(self, rr_files, labels, save='roc-raw_results.pdf'):
        assert len(labels) == len(rr_files), 'Please enter a label for each input raw results file '\
                  '({} files, {} labels).'.format(len(rr_files), len(labels))
        #if len(labels) != len(rr_files):
        #   print('Please enter a label for each input raw results file '
        #         '({} files, {} labels).'.format(len(rr_files), len(labels)))
        #   quit()

        n_per_d = {}
        dt = {}
        ds = {}
        metrics = {}
        truth = []
        scores = []
        for rr_file in rr_files:
            for l in open(rr_file, 'r', encoding="utf8").readlines()[1:]:
                ls = l.strip().split(',')
                pp = float(ls[2])
                truth.append(1)
                scores.append(pp)

                np = float(ls[4])
                truth.append(0)
                scores.append(np)
                if ls[1] not in n_per_d:
                    n_per_d[ls[1]] = 1
                else:
                    n_per_d[ls[1]] += 1
            pr = average_precision_score(truth, scores)
            fpr, tpr, thrs = roc_curve(truth, scores)
            area = roc_auc_score(truth, scores)
            dt[rr_file] = truth
            ds[rr_file] = scores
            metrics[rr_file] = [fpr, tpr, thrs, area, pr]

        plt.figure()
        lw = 2
        for rr_file in rr_files:
            i = rr_files.index(rr_file)
            [fpr, tpr, thrs, area, pr] = metrics[rr_file]
            plt.plot(fpr, tpr, lw=lw, label='{} (AUC-ROC={}, AUPR={})'.format(labels[i], format(area, '.3f'),
                                                                                         format(pr, '.3f')))
        plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right", prop={'size': 8})
        if save:
            plt.savefig(save, dpi=300)
        plt.show()

    def canpredict_denovo(self, method='count', threshold=0.0, topX=10, ind_id=None, proteins=None,
                          minimize=None, consensus=True, cmpd_set='all', save=''):
        """!
        This function is used for predicting putative therapeutics for an indication
        of interest by summing/counting the number of interactions above a certain input interaction
        threshold for all proteins or a specified subset of proteins. An indication can be specified to
        mark drugs associated with that indication in the output. The threshold will vary based on the
        values of the input matrix. Method can be 'count' (score1), which ranks compounds based on the
        number of interactions above the threshold, 'sum' (score2), which ranks the compounds based on the
        highest total sum for interaction scores above the threshold (these two are highly correlated but can
        differ for larger sets of proteins or lower thresholds), 'min', which first ranks by 'count' then re-ranks
        based on the summed interactions with the proteins in the input 'minimize' list - this list should contain
        proteins IDs towards which the user wants low interaction scores - or 'diff', which ranks by the difference of
        sums and the summed scores from off-targets in 'minimize'. A fifth option is 'targets', which inspects
        and outputs the top protein interactions on an individual basis without summing/counting per drug (the
        output format differs from the other two options). If indication_proteins flag is used for
        the CANDO object instantiation, the proteins associated with the input indication will automatically
        be used. Otherwise, the 'proteins=' input can be used. The output can be saved to a file specified
        by 'save='. If ind_id is used, compounds associated with the indication will be included and marked
        in the output for comparison.

        @param method str: 'sum', 'count', or 'targets'
        @param threshold float: a interaction score cutoff to use (ignores values for sum/count less than threshold)
        @param topX int: top number of predicted Compounds to be printed/saved
        @param ind_id str: an indication id for marking drug output/ specifying protein set
        @param proteins List str: list of protein IDs to use from the matrix
        @param minimize List str: list of protein IDs to treat as 'off targets' to avoid, ranking
        @param consensus bool: if True, only compounds with score1 >= 2 will be printed
        @param cmpd_set str: specify the compound set to use ('all', 'approved', or 'other')
        @param save str: name of a file to save results
        @return Returns None
        """

        if ind_id:
            ind = self.get_indication(ind_id)
        c_dct = {}
        top_hits = []
        min_hits = []
        if self.indication_proteins and ind_id:
            indices = []
            for p in ind.proteins:
                indices.append(self.protein_id_to_index[p.id_])
        elif proteins:
            indices = []
            for p in proteins:
                if type(p) is str:
                    indices.append(self.protein_id_to_index[p])
                elif type(p) is int:
                    indices.append(p)
                elif type(p) is Protein:
                    indices.append(self.protein_id_to_index[p.id_])
        else:
            indices = range(len(self.proteins))
        if minimize is None:
            minimize = []
        for c in self.compounds:
            ss = 0.0
            count = 0
            min_ss = 0.0
            min_count = 0
            for pi in indices:
                si = float(c.sig[pi])
                p = self.proteins[pi]
                if si >= threshold:
                    if p.id_ in minimize:
                        min_ss += si
                        min_count += 1
                        top_hits.append((p.id_, c, si, False))
                    else:
                        ss += si
                        count += 1
                        top_hits.append((p.id_, c, si, True))
            if ind_id:
                already_approved = ind.id_ in c.indications
            else:
                already_approved = False  # Not relevant since there is no indication
            c_dct[c.id_] = [ss, count, already_approved, min_ss, min_count]

        if method == 'sum':
            sorted_x = sorted(c_dct.items(), key=lambda x: (x[1][0], x[1][1]))[::-1]
        elif method == 'count':
            sorted_x = sorted(c_dct.items(), key=lambda x: (x[1][1], x[1][0]))[::-1]
        elif method == 'min':
            sorted_x = sorted(c_dct.items(), key=lambda x: (x[1][1], x[1][3]*-1))[::-1]
        elif method == 'diff':
            sorted_x = sorted(c_dct.items(), key=lambda x: (x[1][0] - x[1][3]))[::-1]
        elif method == 'targets':
            sp = sorted(top_hits, key=lambda x: x[2])[::-1]
            print('target  \tscore\toff_target\tid\tapproved\tname')
            if save:
                fo = open(save, 'w', encoding="utf8")
                fo.write('target  \tscore\toff_target\tid\tapproved\tname\n')
            for s in sp:
                co = s[1]
                if cmpd_set == 'approved':
                    if co.status == 'approved':
                        pass
                    elif ind_id:
                        if co in ind.compounds:
                            pass
                        else:
                            continue
                    else:
                        continue
                    st = '{}\t{}\t{}\t{}\t{}\t{}'.format(s[0].ljust(8), round(s[2], 3), co.id_,
                                                         str(s[3]).lower().ljust(10),
                                                         (str(co.status == 'approved').lower()).ljust(8), co.name)
                    print(st)
                    if save:
                        fo.write(st + '\n')
            return

        else:
            sorted_x = []
            raise ValueError('Please enter a valid ranking method')
            #quit()
        if save:
            fo = open(save, 'w', encoding="utf8")
            fo.write('rank\tscore1\tscore2\toffhits\tdiff\tid\tapproved\tname\n')
        print("Printing the {} highest predicted compounds...\n".format(topX))
        i = 0
        print('rank\tscore1\tscore2\toffhits\tdiff\tid\tapproved\tname')
        for p in enumerate(sorted_x):
            if i >= topX != -1:
                break
            else:
                if consensus and p[1][1][1] <= 1:
                    if i == 0:
                        print('\n\tFAILED - there are no compounds with score1 >= 2 -- change the\n'
                              '\targuments to include "consensus=False" to print results with\n'
                              '\tscore1 == 1, or lower the threshold.\n')
                    break
                co = self.get_compound(p[1][0])
                if cmpd_set == 'approved':
                    if co.status != 'approved':
                        if ind_id:
                            if co in ind.compounds:
                                pass
                            else:
                                continue
                        else:
                            continue
                if p[1][1][2]:
                    diff = str(round(p[1][1][0] - p[1][1][3], 3))[0:7].ljust(7)
                    st = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(i + 1, p[1][1][1], str(round(p[1][1][0], 3))[0:7],
                                                                 str(round(p[1][1][3], 3))[0:7].ljust(7), diff, co.id_,
                                                                 (str(co.status == 'approved').lower() + '+').ljust(8),
                                                                 co.name)
                else:
                    diff = str(round(p[1][1][0] - p[1][1][3], 3))[0:7].ljust(7)
                    st = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(i + 1, p[1][1][1], str(round(p[1][1][0], 3))[0:7],
                                                                 str(round(p[1][1][3], 3))[0:7].ljust(7), diff, co.id_,
                                                                 (str(co.status == 'approved').lower()).ljust(8),
                                                                 co.name)
                print(st)
                i += 1
                if save:
                    fo.write(st + '\n')
        return

    def canpredict_compounds(self, ind_id, n=10, topX=10, consensus=True, keep_associated=False, cmpd_set='all',
                             save=''):
        """!
        This function is used for predicting putative therapeutics for an indication
        of interest using a homology-based approach. Input an ind_id id and for each of the
        associated compounds, it will generate the similar compounds (based on distance) and add
        them to a dictionary with a value of how many times it shows up (enrichment). If a
        compound not approved for the indication of interest keeps showing
        up, that means it is similar in signature to the drugs that are
        ALREADY approved for the indication, so it may be a target for repurposing.
        Control how many similar compounds to consider with the argument 'n'. In the output, 'score1'
        refers to the number of times the compound shows up in the top 'n' drugs associated with
        the indication and 'score2' is the average of the ranks for 'score1' (note: 'score2' <= 'n').

        @param ind_id str: Indication id
        @param n int: top number of similar Compounds to be used for each Compound associated with the given Indication
        @param topX int: top number of predicted Compounds to be printed
        @param consensus bool: if True, only compounds with at least 2 votes will be printed
        @param keep_associated bool: Print Compounds that are already approved/associated for the Indication
        @param cmpd_set str: specify the compound set to use ('all', 'approved', or 'other')
        @param save str: name of a file to save results
        @return Returns None
        """

        if int(topX) == -1:
            topX = len(self.compounds)-1
        if int(n) == -1:
            n = len(self.compounds)-1

        i = self.indication_ids.index(ind_id)
        ind = self.indications[i]
        print("{0} compounds found for {1} --> {2}".format(len(ind.compounds), ind.id_, ind.name))

        if self.pathways:
            if self.indication_pathways:
                self.quantify_pathways(ind)
            else:
                self.quantify_pathways()
        for cmpd in ind.compounds:
            c = self.get_compound(cmpd)
            if c.similar_computed:
                if not c.similar_sorted:
                    sorted_scores = sorted(c.similar, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
                    c.similar = sorted_scores
                    c.similar_sorted = True
                continue
            if self.pathways:
                self.generate_similar_sigs(c, aux=True, sort=True)
            elif self.indication_proteins:
                self.generate_similar_sigs(c, sort=True, proteins=ind.proteins)
            else:
                self.generate_similar_sigs(c, sort=True)

        print("Generating compound predictions using top{} most similar compounds...\n".format(n))
        c_dct = {}
        for cmpd in ind.compounds:
            c = self.get_compound(cmpd)
            c2_i = 0
            c_count = 0
            while c_count < n:
                cmpd2 = c.similar[c2_i]
                c2 = self.get_compound(cmpd2[0])
                if c2.status != 'approved' and cmpd_set == 'approved':
                    c2_i += 1
                    continue
                elif c2.is_metabolite and cmpd_set == 'not_metabolite':
                    c2_i += 1
                    continue
                if cmpd2[1] == 0.0:
                    c2_i += 1
                    continue
                already_approved = ind.id_ in c2.indications
                k = c2.id_
                if k not in c_dct:
                    c_dct[k] = [1, already_approved, c_count]
                else:
                    c_dct[k][0] += 1
                    c_dct[k][2] += c_count
                c2_i += 1
                c_count += 1

        sorted_x = sorted(c_dct.items(), key=lambda x: (x[1][0], (-1 * (x[1][2] / x[1][0]))))[::-1]
        i = 0
        if save:
            fo = open(save, 'w', encoding="utf8")
            fo.write('rank\tscore1\tscore2\tprobability\tid\tapproved\tname\n')
        else:
            print('rank\tscore1\tscore2\tprobability\tid\tapproved\tname')
        hg_dct = {}
        for p in enumerate(sorted_x):
            if i >= topX != -1:
                break
            co = self.get_compound(p[1][0])
            if cmpd_set == 'approved':
                if co.status != 'approved':
                    continue
            elif cmpd_set == 'not_metabolite':
                if co.is_metabolite:
                    continue
            if not keep_associated and p[1][1][1]:
                continue
            if consensus and p[1][1][0] <= 1:
                if i == 0:
                    print('\n\tFAILED - there are no compounds with score1 >= 2 -- change the\n'
                          '\targuments to include "consensus=False" to print results with\n'
                          '\tscore1 == 1, and/or increase "n". \n')
                break
            if p[1][1][0] in hg_dct:
                prb = hg_dct[p[1][1][0]]
            else:
                prb_success = 1 / (len(self.compounds) - 1) * n
                prb = '%.2e' % Decimal(1.0 - stats.binom.cdf(p[1][1][0]-1, len(ind.compounds), prb_success))
                hg_dct[p[1][1][0]] = prb
            if p[1][1][1]:
                st = "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(i + 1, p[1][1][0], round(p[1][1][2] / p[1][1][0], 1),
                                                         prb.ljust(11), co.id_,
                                                         (str(co.status == 'approved').lower() + '*').ljust(8), co.name)
            else:
                st = "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(i + 1, p[1][1][0], round(p[1][1][2] / p[1][1][0], 1),
                                                         prb.ljust(11), co.id_,
                                                         (str(co.status == 'approved').lower()).ljust(8), co.name)
            if save:
                fo.write(st + '\n')
            else:
                print(st)
            i += 1
        print('\n')

    def canpredict_indications(self, cmpd, n=10, topX=10, consensus=True, sorting='prob', save=''):
        """!
        This function is the inverse of canpredict_compounds. Input a compound
        of interest cando_cmpd (or a novel protein signature of interest new_sig)
        and the most similar compounds to it will be computed. The indications
        associated with the top n most similar compounds to the query compound will
        be examined to see if any are repeatedly enriched.

        @param cmpd Compound: Compound object to be used
        @param n int: top number of similar Compounds to be used for prediction
        @param topX int: top number of predicted Indications to be printed
        @param consensus bool: if True, only indications with at least 2 votes will be printed
        @param sorting str: whether to sort the indications by probability ('prob') or score ('score')
        @param save str: path to file to save the output
        @return Returns None
        """
        if n == -1:
            n = len(self.compounds)-1
        if topX == -1:
            topX = len(self.indications)

        print(f"Generating indication predictions for {cmpd.name}...")

        if type(cmpd) is Compound:
            cmpd = cmpd
        elif type(cmpd) is int:
            cmpd = self.get_compound(cmpd)
        print(f"  Compound id = {cmpd.id_}")
        print(f"  Compound index = {cmpd.index}")
        print(f"  n = {n}")
        if not cmpd.similar_computed:
            print("  Comparing signature to all CANDO compound signatures...")
            self.generate_similar_sigs(cmpd, sort=True)
            cmpd.similar = [(c.id_,dist) for c,dist in cmpd.similar]
        i_dct = {}
        for c in cmpd.similar[:n]:
            cmpd2 = self.get_compound(c[0])
            for i in cmpd2.indications:
                ind = self.get_indication(i)
                if ind.id_ not in i_dct:
                    i_dct[ind.id_] = [1, len(ind.compounds)]
                else:
                    i_dct[ind.id_][0] += 1

        i2p_dct = {}
        for ik in i_dct:
            [k, n_app] = i_dct[ik]
            if consensus and k == 1:
                continue
            prb = 1.0 - stats.hypergeom.cdf(k, len(self.compounds) - 1, n_app, n)
            i2p_dct[ik] = (k, prb)

        assert not consensus or len(i2p_dct) > 0, '\n\tFAILED - there are no compounds with score1 >= 2 -- change the\n'\
                  '\targuments to include "consensus=False" to print results with\n'\
                  '\tscore1 == 1, and/or increase "n".\n'
        #if consensus and len(i2p_dct) == 0:
        #   print('\n\tFAILED - there are no compounds with score1 >= 2 -- change the\n'
        #         '\targuments to include "consensus=False" to print results with\n'
        #         '\tscore1 == 1, and/or increase "n".\n')
        #   quit()

        assert sorting in ('score', 'prob'), 'Please enter proper sorting method: "prob" or "score"'
        if sorting == 'score':
            sorted_x = sorted(list(i2p_dct.items()), key=lambda x: x[1][0], reverse=True)
        elif sorting == 'prob':
            sorted_x = sorted(list(i2p_dct.items()), key=lambda x: x[1][1], reverse=False)
        #else:
        #   sorted_x = []
        #   print('Please enter proper sorting method: "prob" or "score" -- quitting.')
        #   quit()

        if save:
            fo = open(save, 'w', encoding="utf8")
            print(f"  Saving the {top} highest predicted indications...")
            fo.write("rank\tprobability\tscore\tind_id\tindication\n")
        print(f"  Printing the {topX} highest predicted indications...\n")
        print("rank\tprobability\tscore\tind_id    \tindication")
        n_print = topX if len(sorted_x) >= topX else len(sorted_x)
        for i in range(n_print):
            indd = self.get_indication(sorted_x[i][0])
            prb = '%.2e' % Decimal(sorted_x[i][1][1])
            print("{}\t{}\t{}\t{}\t{}".format(i + 1, prb.ljust(11), sorted_x[i][1][0], indd.id_, indd.name))
            if save:
                fo.write("{}\t{}\t{}\t{}\t{}\n".format(i+1, prb, sorted_x[i][1][0], indd.id_, indd.name))
        if save:
            fo.close()
        print('')

    def canpredict_adr(self, cmpd, n=10, topX=10, consensus=True, sorting='prob', save=''):
        """!
        This function is the inverse of canpredict_compounds. Input a compound
        of interest cando_cmpd (or a novel protein signature of interest new_sig)
        and the most similar compounds to it will be computed. The ADRs
        associated with the top n most similar compounds to the query compound will
        be examined to see if any are repeatedly enriched.

        @param cmpd Compound: Compound object to be used
        @param n int: top number of similar Compounds to be used for prediction
        @param topX int: top number of predicted Indications to be printed
        @param consensus bool: if True, only ADRs with at least 2 votes will be printed
        @param sorting str: whether to sort the ADRs by probability ('prob') or score ('score')
        @param save str: path to file to save output
        @return Returns None
        """
        if n == -1:
            n = len(self.compounds)-1
        if topX == -1:
            topX = len(self.adrs)

        if type(cmpd) is Compound:
            cmpd = cmpd
        elif type(cmpd) is int:
            cmpd = self.get_compound(cmpd)
        print("Using CANDO compound {}".format(cmpd.name))
        print("Compound has id {} and index {}".format(cmpd.id_, cmpd.index))
        print("Comparing signature to all CANDO compound signatures...")
        self.generate_similar_sigs(cmpd, sort=True)
        print("Generating ADR predictions using top{} most similar compounds...".format(n))
        a_dct = {}
        for c in cmpd.similar[0:n]:
            for adr in c[0].adrs:
                if adr.id_ not in a_dct:
                    a_dct[adr.id_] = [1, len(adr.compounds)]
                else:
                    a_dct[adr.id_][0] += 1

        a2p_dct = {}
        for ik in a_dct:
            [k, n_app] = a_dct[ik]
            if consensus and k == 1:
                continue
            prb = 1.0 - stats.hypergeom.cdf(k, len(self.compounds) - 1, n_app, n)
            a2p_dct[ik] = (k, prb)

        assert not consensus or len(a2p_dct) > 0, '\n\tFAILED - there are no compounds with score1 >= 2 -- change the\n'\
                  '\targuments to include "consensus=False" to print results with\n'\
                  '\tscore1 == 1, and/or increase "n".\n'
        #if consensus and len(a2p_dct) == 0:
        #   print('\n\tFAILED - there are no compounds with score1 >= 2 -- change the\n'
        #         '\targuments to include "consensus=False" to print results with\n'
        #         '\tscore1 == 1, and/or increase "n".\n')
        #   quit()

        assert sorting in ('score', 'prob'), 'Please enter proper sorting method: "prob" or "score"'
        if sorting == 'score':
            sorted_x = sorted(list(a2p_dct.items()), key=lambda x: x[1][0], reverse=True)
        elif sorting == 'prob':
            sorted_x = sorted(list(a2p_dct.items()), key=lambda x: x[1][1], reverse=False)
        #else:
        #   sorted_x = []
        #   print('Please enter proper sorting method: "prob" or "score" -- quitting.')
        #   quit()

        if save:
            fo = open(save, 'w', encoding="utf8")
            print("Saving the {} highest predicted ADRs...\n".format(topX))
            fo.write("rank\tprobability\tscore\tadr_id\tadr\n")
        else:
            print("Printing the {} highest predicted ADRs...\n".format(topX))
            print("rank\tprobability\tscore\tadr_id    \tadr")
        n_print = topX if len(sorted_x) >= topX else len(sorted_x)
        for i in range(n_print):
            adrr = self.get_adr(sorted_x[i][0])
            prb = '%.2e' % Decimal(sorted_x[i][1][1])
            if save:
                fo.write("{}\t{}\t{}\t{}\t{}\n".format(i+1, prb, sorted_x[i][1][0], adrr.id_, adrr.name))
            else:
                print("{}\t{}\t{}\t{}\t{}".format(i+1, prb.ljust(11), sorted_x[i][1][0], adrr.id_, adrr.name))
        if save:
            fo.close()
        print('')

    def canpredict_ddi_cmpds(self, cmpd, n=10, topX=10, save=''):
        """!
        Input a compound of interest cando_cmpd and the most similar compounds to it will be computed
        and outputted as potential drug-drug-interactions.

        @param cmpd Compound: Compound object to be used
        @param n int: top number of similar Compounds to be used for prediction
        @param topX int: top number of predicted Drug-drug Interactions to be printed
        @return Returns None
        """
        if n == -1:
            n = len(self.compounds)-1
        if topX == -1:
            topX = len(self.compounds)-1

        if type(cmpd) is Compound:
            cmpd = cmpd
        elif type(cmpd) is int:
            cmpd = self.get_compound(cmpd)
        print("Using CANDO compound {}".format(cmpd.name))
        print("Compound has id {} and index {}".format(cmpd.id_, cmpd.index))
        print("Comparing signature to all CANDO compound signatures...")
        self.generate_similar_sigs(cmpd, sort=True)
        print("Generating interaction predictions using top{} most similar compounds...".format(n))
        i_dct = {}
        for c in cmpd.similar[0:n]:
            for itx in c[0].compounds:
                if itx.id_ not in i_dct:
                    i_dct[itx.id_] = 1
                else:
                    i_dct[itx.id_] += 1
        sorted_x = sorted(i_dct.items(), key=operator.itemgetter(1), reverse=True)
        if save:
            fo = open(save, 'w', encoding="utf8")
            print("Saving the {} highest predicted compounds...\n".format(topX))
            fo.write("rank\tscore\tcmpd_id\tcompound\n")
        else:
            print("Printing the {} highest predicted compounds...\n".format(topX))
            print("rank\tscore\tcmpd_id    \tcompound")
        topX = min(topX,len(sorted_x))
        for i in range(topX):
            itxd = self.get_compound(sorted_x[i][0])
            if save:
                fo.write("{}\t{}\t{}\t{}\n".format(i+1, sorted_x[i][1], itxd.id_, itxd.name))
            else:
                print("{}\t{}\t{}\t{}".format(i+1, sorted_x[i][1], itxd.id_, itxd.name))
        if save:
            fo.close()
        print('')

    def canpredict_ddi_adrs(self, cmpd_pair, n=10, topX=10, save=''):
        """!
        Similarly to canpredict_adrs(), input a compound pair of interest (cmpd_pair)
        and the most similar compound pairs to it will be computed. The ADRs associated
        with the top n most similar compound pairs to the query pair will be examined
        to see if any are repeatedly enriched.

        @param cmpd_pair Compound_pair: Compound_pair object to be used
        @param n int: top number of similar Compound_pairs to be used for prediction
        @param topX int: top number of predicted ADRs to be printed
        @return Returns None
        """
        if n == -1:
            n = len(self.compound_pairs)-1
        if topX == -1:
            topX = len(self.adrs)

        if type(cmpd_pair) is Compound_pair:
            cmpd_pair = cmpd_pair
        elif type(cmpd_pair) is tuple:
            try:
                cmpd = self.get_compound_pair(cmpd_pair)
            except:
                print("Compound pair does not exist.")
        '''
        if type(cmpd_pair) is tuple:
            c1 = self.get_compound(cmpd_pair[0])
            c2 = self.get_compound(cmpd_pair[1])
            cmpd_pair = Compound_pair((c1.name,c2.name),cmpd_pair,cmpd_pair)
            self.compound_pairs.append(cmpd_pair)
            self.compound_pair_ids.append(cmpd_pair.id_)
            cmpd_pair.sig = [i+j for i,j in zip(c1.sig,c2.sig)]
        '''
        print("Using CANDO compound pair {}".format(cmpd_pair.name))
        print("Compound pair has id {} and index {}".format(cmpd_pair.id_, cmpd_pair.index))
        #print("Comparing signature to all CANDO compound pair signatures...")
        #self.generate_similar_sigs_cp(cmpd_pair, sort=True, ncpus=self.ncpus)
        print("Generating ADR predictions using top{} most similar compound pairs...".format(n))
        #conn = 'sqlite://ddi.db'
        conn = f'sqlite://{self.db_name}'
        df_dists = pl.read_sql(f"SELECT * FROM dists WHERE id=={str(cmpd_pair.id_)}", conn)
        c_sorted = json.loads(df_dists.filter(pl.col('id')==str(c)).select(['dists']).item().replace("'",'"'))
        c_sorted = pl.DataFrame(c_sorted)
        #c_df = c_sorted.transpose().with_columns(pl.Series(name='ids',values=c_sorted.columns)).sort('column_1')
        c_df = c_sorted.transpose().with_columns(pl.Series(name='ids',values=c_sorted.columns)).sort('column_0')
        c_sorted = df.select(pl.col("ids")).to_list()
        a_dct = {}
        for cp2_id in c_sorted[:n]:
            cp2 = self.get_compound_pair(cp2_id)
            for adr in cp2.adrs:
                if adr.id_ not in a_dct:
                    a_dct[adr.id_] = 1
                else:
                    a_dct[adr.id_] += 1
        '''
        for i in cmpd_pair.similar[0:n]:
            cp2 = self.get_compound_pair(c)
            for adr in c[0].adrs:
                if adr.id_ not in a_dct:
                    a_dct[adr.id_] = 1
                else:
                    a_dct[adr.id_] += 1
        '''
        sorted_x = sorted(a_dct.items(), key=operator.itemgetter(1), reverse=True)
        if save:
            fo = open(save, 'w', encoding="utf8")
            print("Saving the {} highest predicted indications...\n".format(topX))
            fo.write("rank\tscore\tadr_id\tadverse_reaction\n")
        else:
            print("Printing the {} highest predicted ADRs...\n".format(topX))
            print("rank\tscore\tadr_id    \tadr_name")
        for i in range(topX):
            adr = self.get_adr(sorted_x[i][0])
            if save:
                fo.write("{}\t{}\t{}\t{}\n".format(i+1, sorted_x[i][1], adr.id_, adr.name))
            else:
                print("{}\t{}\t{}\t{}".format(i+1, sorted_x[i][1], adr.id_, adr.name))
        if save:
            fo.close()
        print('')


    def canpredict_ddi_adrs_newpair(self, cmpd1, cmpd2, n=10, topX=10, save='', verbose=True, printout=True):
        """!
        Similarly to canpredict_adrs(), input a compound pair of interest (cmpd_pair)
        and the most similar compound pairs to it will be computed. The ADRs associated
        with the top n most similar compound pairs to the query pair will be examined
        to see if any are repeatedly enriched.

        @param cmpd_pair Compound_pair: Compound_pair object to be used
        @param n int: top number of similar Compound_pairs to be used for prediction
        @param topX int: top number of predicted ADRs to be printed
        @return Returns None
        """
        if n == -1:
            n = len(self.compound_pairs)-1
        if topX == -1:
            topX = len(self.adrs)

        if type(cmpd1) is Compound:
            cmpd1 = cmpd1
        elif type(cmpd1) is int:
            cmpd1 = self.get_compound(cmpd1)

        if type(cmpd2) is Compound:
            cmpd2 = cmpd2
        elif type(cmpd2) is int:
            cmpd2 = self.get_compound(cmpd2)

        c1 = min(cmpd1.id_,cmpd2.id_)
        c2 = max(cmpd1.id_,cmpd2.id_)
        cmpd_pair_id = (c1, c2)
        known_cp = False
        if cmpd_pair_id in self.compound_pair_ids:
            cmpd_pair = self.get_compound_pair(cmpd_pair_id)
            print(cmpd_pair)
            known_cp = True
            print(f"Known compound pair -- {cmpd_pair_id}")
        else:
            cmpd_pair = Compound_pair((cmpd1.name,cmpd2.name),cmpd_pair_id,cmpd_pair_id)
            cmpd_pair.sig = [i+j for i,j in zip(cmpd1.sig,cmpd2.sig)]

        '''
        if type(cmpd_pair) is tuple:
            c1 = self.get_compound(cmpd_pair[0])
            c2 = self.get_compound(cmpd_pair[1])
            cmpd_pair = Compound_pair((c1.name,c2.name),cmpd_pair,cmpd_pair)
            self.compound_pairs.append(cmpd_pair)
            self.compound_pair_ids.append(cmpd_pair.id_)
            cmpd_pair.sig = [i+j for i,j in zip(c1.sig,c2.sig)]
        '''
        #conn = 'sqlite://ddi.db'
        conn = f'sqlite://{self.db_name}'
        if verbose:
            print("Using CANDO compound pair {}".format(cmpd_pair.name))
            print("Compound pair has id {} and index {}".format(cmpd_pair.id_, cmpd_pair.index))
            print("Comparing signature to all CANDO compound pair signatures...")
        c_sorted = []
        if known_cp:
            df_dists = pl.read_sql(f"SELECT * FROM dists WHERE id=='{cmpd_pair_id}'", conn)
            c_sorted = json.loads(df_dists.filter(pl.col('id')==str(cmpd_pair_id)).select(['dists']).item().replace("'",'"'))
            c_sorted = pl.DataFrame(c_sorted)
            c_df = c_sorted.transpose().with_columns(pl.Series(name='ids',values=c_sorted.columns)).sort('column_0')
            c_sorted = c_df.get_column("ids").to_list()
            c_sorted = [self.get_compound_pair(ast.literal_eval(x)) for x in c_sorted]
        else:
            dists_sorted = self.generate_similar_sigs_cp(cmpd_pair, sort=True, ncpus=self.ncpus)
            c_sorted = list(list(zip(*dists_sorted))[0])
        if verbose:
            print("Done comparing compound pair signatures.\n")
            print("Generating ADR predictions using top{} most similar compound pairs...".format(n))
        #conn = 'sqlite://ddi.db'
        #df_dists = pl.read_sql(f"SELECT * FROM dists WHERE id=={cmpd_pair.id_}", conn)
        #c_sorted = json.loads(df_dists.filter(pl.col('id')==str(c)).select(['dists']).item().replace("'",'"'))
        #c_sorted = pl.DataFrame(c_sorted)
        #c_df = c_sorted.transpose().with_columns(pl.Series(name='ids',values=c_sorted.columns)).sort('column_0')
        #c_sorted = df.select(pl.col("ids")).to_list()
        '''
        a_dct = {}
        for cp2 in c_sorted[:n]:
            #cp2 = self.get_compound_pair(cp2_id)
            for adr_id in cp2.adrs:
                if adr_id not in a_dct:
                    a_dct[adr_id] = 1
                else:
                    a_dct[adr_id] += 1
        sorted_x = sorted(a_dct.items(), key=operator.itemgetter(1), reverse=True)
        if save:
            fo = open(save, 'w')
            print("Saving the {} highest predicted ADRs...\n".format(topX))
            fo.write("rank\tscore\tadr_id\tadr\n")
        else:
            print("Printing the {} highest predicted ADRs...\n".format(topX))
            print("rank\tscore\tadr_id    \tadr_name")
        tot = min(topX,len(sorted_x))
        for i in range(tot):
            adr = self.get_adr(sorted_x[i][0])
            if save:
                fo.write("{}\t{}\t{}\t{}\n".format(i+1, sorted_x[i][1], adr.id_, adr.name))
            else:
                print("{}\t{}\t{}\t{}".format(i+1, sorted_x[i][1], adr.id_, adr.name))
        if save:
            fo.close()
        print('')
        '''
        cols = ['score','adr_id','adr_name']
        adr_df = pd.DataFrame(columns=cols)
        for cp2 in c_sorted[:n]:
            for adr_id in cp2.adrs:
                if adr_id not in adr_df.adr_id.values:
                    adr_df = pd.concat([adr_df,pd.DataFrame([[1,adr_id,self.get_adr(adr_id).name]],columns=cols)],ignore_index=True)
                else:
                    adr_df.loc[(adr_df['adr_id']==adr_id),'score'] += 1
        adr_df.sort_values(by=['score'],ascending=False,inplace=True)
        adr_df['rank'] = list(range(1,len(adr_df)+1))
        adr_df['score_norm'] = adr_df.apply(lambda x: x['score']/n, axis=1)
        adr_df = adr_df[['rank','score','score_norm','adr_id','adr_name']]
        tot = min(topX,len(adr_df))
        if save:
            if verbose: print("Saving the {} highest predicted ADRs...\n".format(tot))
            adr_df.iloc[:tot,:].to_csv(save,sep='\t',index=False)
            if verbose: print("Predicted ADRs saved.\n")
        else:
            if verbose: print("Printing the {} highest predicted ADRs...\n".format(tot))
            if printout: print(adr_df.iloc[:tot,:].to_string(index=False))
            if verbose: print()
        return adr_df.iloc[:tot,:]


    def canpredict_ddi_adrs_newpairs(self, cp_list, n=10, topX=10, save='', verbose=True, printout=True):
        """!
        Similarly to canpredict_adrs(), input a compound pair of interest (cmpd_pair)
        and the most similar compound pairs to it will be computed. The ADRs associated
        with the top n most similar compound pairs to the query pair will be examined
        to see if any are repeatedly enriched.

        @param cmpd_pair Compound_pair: Compound_pair object to be used
        @param n int: top number of similar Compound_pairs to be used for prediction
        @param topX int: top number of predicted ADRs to be printed
        @return Returns None
        """
        if n == -1:
            n = len(self.compound_pairs)-1
        if topX == -1:
            topX = len(self.adrs)

        cmpd_pair_list = []
        for x in cp_list:
            if x in self.compound_pair_ids:
                cmpd_pair_list.append(self.get_compound_pair(x))
            else:
                cmpd1 = self.get_compound(cmpd1)
                cmpd2 = self.get_compound(cmpd2)
                c1 = min(cmpd1.id_,cmpd2.id_)
                c2 = max(cmpd1.id_,cmpd2.id_)
                cmpd_pair = (c1,c2)
                cmpd_pair = Compound_pair((cmpd1.name,cmpd2.name),cmpd_pair,cmpd_pair)
                cmpd_pair.sig = [i+j for i,j in zip(cmpd1.sig,cmpd2.sig)]
                cmpd_pair_list.append(cmpd_pair)

        if verbose:
            print("Using CANDO compound pair {}".format(cmpd_pair.name))
            print("Compound pair has id {} and index {}".format(cmpd_pair.id_, cmpd_pair.index))
            print("Comparing signature to all CANDO compound pair signatures...")
        dists_sorted = self.generate_similar_sigs_cps(cmpd_pair_list, sort=True, ncpus=self.ncpus)
        if verbose:
            print("Done comparing compound pair signatures.\n")
            print("Generating ADR predictions using top{} most similar compound pairs...".format(n))
        
        cols = ['score','adr_id','adr_name']
        for idx in range(len(dists_sorted)):
            c_sorted = list(list(zip(*dists_sorted[idx]))[0])
            adr_df = pd.DataFrame(columns=cols)
            for cp2 in c_sorted[:n]:
                for adr_id in cp2.adrs:
                    if adr_id not in adr_df.adr_id.values:
                        adr_df = pd.concat([adr_df,pd.DataFrame([[1,adr_id,self.get_adr(adr_id).name]],columns=cols)],ignore_index=True)
                    else:
                        adr_df.loc[(adr_df['adr_id']==adr_id),'score'] += 1
            adr_df.sort_values(by=['score'],ascending=False,inplace=True)
            adr_df['rank'] = list(range(1,len(adr_df)+1))
            adr_df['score_norm'] = adr_df.apply(lambda x: x['score']/n, axis=1)
            adr_df = adr_df[['rank','score','score_norm','adr_id','adr_name']]
            tot = min(topX,len(adr_df))
            if save:
                if verbose: print("Saving the {} highest predicted ADRs...\n".format(tot))
                adr_df.iloc[:tot,:].to_csv(save,sep='\t',index=False)
                if verbose: print("Predicted ADRs saved.\n")
            else:
                if verbose: print("Printing the {} highest predicted ADRs...\n".format(tot))
                if printout: print(adr_df.iloc[:tot,:].to_string(index=False))
                if verbose: print()
            return adr_df.iloc[:tot,:]




    def similar_compounds(self, cmpd, n=10, save=''):
        """!
        Computes and prints the top n most similar compounds to an input
        Compound object cando_cmpd or input novel signature new_sig

        @param cmpd Compound: Compound object
        @param n int: top number of similar Compounds to be used for prediction
        @param save str: path to the file where the results will be written. If empty then results will only be printed. Default: ''.
        @return Returns None
        """
        print(f"Generating most similar compounds for {cmpd.name}...")

        if type(cmpd) is Compound:
            cmpd = cmpd
        elif type(cmpd) is int:
            cmpd = self.get_compound(cmpd)
        # Add ability to generate simialr list from an iputted signature
        #elif type(cmpd) is list:
        print(f"Compound id = {cmpd.id_}")
        print(f"Compound index = {cmpd.index}")
        if not cmpd.similar_computed:
            print("Comparing signature to all CANDO compound signatures...")
            self.generate_similar_sigs(cmpd, sort=True)
            cmpd.similar = [(c.id_,dist) for c,dist in cmpd.similar]
        print(f"  Printing {n} most similar compounds...\n")
        print("  rank\tdist\tid\tname")
        for i in range(n+1):
            c = self.get_compound(cmpd.similar[i][0])
            print("  {}\t{:.3f}\t{}\t{}".format(i+1, cmpd.similar[i][1], c.id_, c.name))
        print('\n')
        if save:
            print(f"  Saving {n} most similar compounds...")
            with open(save, 'w', encoding="utf8") as o:
                o.write("rank\tdist\tid\tname")
                for i in range(n+1):
                    o.write("{}\t{:.3f}\t{}\t{}\n".format(i+1, cmpd.similar[i][1], cmpd.similar[i][0].id_, cmpd.similar[i][0].name))
            print(f"  Results saved to {save}.\n")
        return

    def add_cmpd(self, new_sig, new_name=''):
        """!
        Add a new Compound object to the platform
        
        @param new_sig str: Path to the tab-separated interaction scores
        @param new_name str: Name for the new Compound
        @return Returns None
        """
        #with open(new_sig, 'r', encoding="utf8") as nsf:
        with open(new_sig, 'r') as nsf:
            n_sig = [0.00] * len(self.proteins)
            for l in nsf:
                [pr, sc] = l.strip().split('\t')
                pr_i = self.protein_id_to_index[pr]
                n_sig[pr_i] = sc
        i = max([cm.id_ for cm in self.compounds]) + 1
        if not new_name:
            new_name = 'compound_{}'.format(i)
        cmpd = Compound(new_name, i, i)
        cmpd.sig = n_sig
        self.compounds.append(cmpd)
        self.compound_ids.append(cmpd.id_)

        if self.compounds[0].similar_computed or len(self.compounds[0].similar) > 1:
            dists = self.generate_similar_sigs(cmpd, sort=True)
            for c, dist in dists:
                c.similar.append((cmpd.id_, dist))
                c.similar = sorted(c.similar, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
            dists = sorted(dists, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
        cmpd.similar = [(c.id_,dist) for c,dist in cmpd.similar]
        #print(cmpd.similar)
        print("New compound is " + cmpd.name)
        print(f"New compound has id {cmpd.id_} and index {cmpd.index}.\n")

    def sigs(self, rm):
        """!
        Return a list of all signatures, rm is a list of compound ids you do not want in the list

        @param rm list: List of compound ids to remove from list of signatures
        @return list: List of all signatures
        """
        return [x.sig for x in self.proteins if x.id_ not in rm]

    def save_dists_to_file(self, f):
        """!
        Write calculated distances of all compounds to all compounds to file

        @param f File name to save distances
        """
        def dists_to_str(cmpd):
            o = ''
            for s in cmpd.similar:
                o += '{}\t'.format(s[1])
            o = o + '\n'
            return o

        with open(f, 'w', encoding="utf8") as srf:
            for c in self.compounds:
                srf.write(dists_to_str(c))

    def fusion(self, cando_objs, out_file='', method='sum'):
        """!
        This function re-ranks the compounds according to the desired comparison specified by
        'method' -> currently supports 'min', 'avg', 'mult', and 'sum'

        @param cando_objs list: List of CANDO objects
        @param out_file str: Path to where the result will be written
        @param method str: Method of fusion to be used (e.g., sum, mult, etc.)
        @return Returns CANDO object
        """
        print(f"Fusing CANDO objects using {method}...")
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
                cid_to_ranks[c.id_][sims[i][0]] = [i]
        for cando_obj in cando_objs:
            for c2 in cando_obj.compounds:
                sims2 = c2.similar
                for j in range(len(sims2)):
                    cid_to_ranks[c2.id_][sims2[j][0]].append(j)
        for c3 in cnd.compounds:
            ranks_dct = cid_to_ranks[c3.id_]
            for c4 in cnd.compounds:
                if c4.id_ == c3.id_:
                    continue
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
            with open(out_file, 'w', encoding="utf8") as fo:
                for co in cnd.compounds:
                    s = list(map(str, [x[1] for x in co.similar]))
                    fo.write('\t'.join(s) + '\n')
        for cf in cnd.compounds:
            sorted_scores = sorted(cf.similar, key=lambda x: x[1] if not math.isnan(x[1]) else 100000)
            cf.similar = sorted_scores
            cf.similar_computed = True
            cf.similar_sorted = True
        print(f"Fusion of CANDO objects complete.\n")
        return cnd

    def normalize(self):
        """!
        Normalize the distance scores to between [0,1]. Simply divides all scores by the largest distance
        between any two compounds.

        @return Returns None
        """
        assert len(self.compounds[0].similar) > 0, 'Similar scores not computed yet'
        #if len(self.compounds[0].similar) == 0:
        #   print('Similar scores not computed yet')
        #   return

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
        """!
        Print stats about the CANDO object

        """
        nc = len(self.compounds)
        b = self.compounds[0].similar_computed
        ni = len(self.indications)
        np = len(self.proteins)
        if np:
            return 'CANDO: {0} compounds, {1} proteins, {2} indications\n' \
                   '\tMatrix - {3}\nIndication mapping - {4}\n' \
                   '\tDistances computed - {5}'.format(nc, np, ni, self.matrix, self.i_map, b)
        elif self.read_dists:
            return 'CANDO: {0} compounds, {1} indications\n' \
                   '\tCompound comparison file - {2}\n' \
                   '\tIndication mapping - {3}'.format(nc, ni, self.read_dists, self.i_map)
        else:
            return 'CANDO: {0} compounds, {1} indications\n' \
                   '\tIndication mapping - {2}'.format(nc, ni, self.i_map)


class Matrix(object):
    """!
    An object to represent a matrix

    Intended for easier handling of matrices.
    Convert between fpt and tsv, as well as distance to similarity (and vice versa)
    """
    def __init__(self, matrix_file, dist=False, convert_to_tsv=False):
        ## @var matrix_file
        # str: Path to file with interaction scores
        self.matrix_file = matrix_file
        ## @var dist
        # bool: if the matrix_file is an dist file
        self.dist = dist
        ## @var convert_to_tsv
        # bool: Convert old matrix format (.fpt) to .tsv
        self.convert_to_tsv = convert_to_tsv
        ## @var proteins
        # list: Proteins in the Matrix
        self.proteins = []
        ## @var values
        # list: Values in the Matrix
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

        if not dist:
            with open(matrix_file, 'r', encoding="utf8") as f:
                lines = f.readlines()
                if convert_to_tsv:
                    if matrix_file[-4:] == '.fpt':
                        out_file = '.'.join(matrix_file.split('.')[:-1]) + '.tsv'
                    else:
                        out_file = matrix_file + '.tsv'
                    of = open(out_file, 'w', encoding="utf8")
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
                        assert len(vec) > 1, 'The matrix file {} is in the old fpt format -- please '\
                                  'convert to tsv with the following line of code:\n-> Matrix("{}", convert_to_tsv=True) <-'.format(self.matrix_file, self.matrix_file)
                        #if len(vec) < 2:
                        #   print('The matrix file {} is in the old fpt format -- please '
                        #         'convert to tsv with the following line of code:'.format(self.matrix_file))
                        #   print('-> Matrix("{}", convert_to_tsv=True) <-'.format(self.matrix_file))
                        #   quit()
                        name = vec[0]
                        scores = vec[1:]
                        self.proteins.append(name)
                        self.values.append(list(map(float, scores)))
        else:
            with open(matrix_file, 'r', encoding="utf8") as rrs:
                lines = rrs.readlines()
                for i in range(len(lines)):
                    scores = list(map(float, lines[i].strip().split('\t')))
                    self.values.append(scores)

    def convert(self, out_file):
        """!
        Convert similarity matrix to distance matrix or vice versa. The
        first value in the matrix will determine the type of conversion
        (0.0 means distance to similarity, 1.0 means similarity to distance).

        @param out_file str: File path to which write the converted matrix.
        @return Returns None
        """
        assert self.values[0][0] in (0.0, 1.0), 'The first value is not 0.0 or 1.0; '\
                  'please ensure the matrix is generated properly'
        if self.values[0][0] == 0.0:
            metric = 'd'
        elif self.values[0][0] == 1.0:
            metric = 's'
        #else:
        #   metric = None
        #   print('The first value is not 0.0 or 1.0; '
        #         'please ensure the matrix is generated properly')
        #   quit()

        def to_dist(s):
            return 1 - s

        def to_sim(d):
            return 1 / (1 + d)

        of = open(out_file, 'w', encoding="utf8")
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

    def normalize(self, outfile, dimension='drugs', method='avg'):
        """!
        Normalize the interaction scores across drugs (default) or proteins (not implemented yet).

        @param outfile str: File path to which is written the converted matrix.
        @param dimension str: which vector to normalize - either 'drugs' to normalize all
        scores within the proteomic vector or 'proteins' to normalize for a protein against
        all drug scores.
        @param method str: normalize by the average or max within the vectors
        @return Returns None
        """
        # dimensions include drugs or features (e.g. "proteins")
        # methods are average ('avg') or max ('max')
        dvs = {}  # drug vectors
        cc = 0
        if dimension == 'drugs':
            for vec in self.values:
                for vi in range(len(vec)):
                    if cc == 0:
                        dvs[vi] = []
                    dvs[vi].append(vec[vi])
                cc += 1

        new_dvecs = []
        for i in range(len(dvs)):
            vec = dvs[i]
            assert method in ('avg', 'max'), 'Please enter a proper normalization method: "max" or "avg"'
            if method == 'avg':
                norm_val = np.average(vec)
            elif method == 'max':
                norm_val = max(vec)
            #else:
            #   print('Please enter a proper normalization method: "max" or "avg"')
            #   quit()

            def norm(x):
                if norm_val == 0:
                    return 0.0
                else:
                    return x/norm_val

            new_dvecs.append(list(map(norm, vec)))

        pvs = {}
        for dvi in range(len(new_dvecs)):
            for p in range(len(self.proteins)):
                try:
                    pvs[p].append(new_dvecs[dvi][p])
                except KeyError:
                    pvs[p] = [new_dvecs[dvi][p]]

        with open(outfile, 'w', encoding="utf8") as fo:
            for p in range(len(self.proteins)):
                fo.write('{}\t{}\n'.format(self.proteins[p], '\t'.join(list(map(str, pvs[p])))))

def single_interaction(c_id, p_id, v="v2.2", fp="rd_ecfp4", vect="int", 
        dist="dice", org="nrpdb", bs="coach", 
        c_cutoff=0.0, p_cutoff=0.0, percentile_cutoff=0.0, 
        i_score="P", nr_ligs=True, approved_only=False, lig_name=False, 
        lib_path='',prot_path=''):

    def print_time(s):
        if s >= 60:
            m = s / 60.0
            s -= m * 60.0
            if m >= 60.0:
                h = m / 60.0
                m -= h * 60.0
                print("Interaciton calculation took {:.0f} hr {:.0f} min {:.0f} s to finish.".format(h, m, s))
            else:
                print("Interaciton calculation took {:.0f} min {:.0f} s to finish.".format(m, s))
        else:
            print("Interaciton calculation took {:.0f} s to finish.".format(s))

    print("Calculating BANDOCK interaction...")
    start = time.time()

    c_id = int(c_id)

    if org=='test':
        pre = "."
    else:
        pre = os.path.dirname(__file__) + "/data/v2.2+/"
    lig_path = "{}/ligs/fps".format(pre)
    if not lib_path:
        cmpd_path = "{}/cmpds/fps-{}".format(pre,v)
        map_path = "{}/mappings".format(pre)
    else:
        cmpd_path = "{0}/{1}/cmpds/fps-{1}".format(lib_path,v)
        map_path = "{0}/{1}/mappings".format(lib_path,v)

    # Remove redundant ligands from full list
    # Especially important for percentile calculations
    if nr_ligs:
        if not os.path.exists("{}/mappings/nr_ligs.csv".format(pre)):
            url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/mappings/nr_ligs.csv'
            dl_file(url, '{}/mappings/nr_ligs.csv'.format(pre))
        nr_ligs = pd.read_csv("{}/mappings/nr_ligs.csv".format(pre),header=None)
    nr_ligs = nr_ligs[0].values.flatten()

    # Download protein matrix if it does not exist
    if not prot_path:
        if not os.path.exists("{}/prots/{}-{}.tsv".format(pre,org,bs)):
            url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/prots/{}-{}.tsv'.format(org,bs)
            dl_file(url, '{}/prots/{}-{}.tsv'.format(pre,org,bs))
        p_matrix = pd.read_csv("{}/prots/{}-{}.tsv".format(pre,org,bs),sep='\t',header=None,index_col=0)
    else:
        p_matrix = pd.read_csv("{}/{}-{}.tsv".format(prot_path,org,bs),sep='\t',header=None,index_col=0)

    
    # Create dictionary of lists
    # Keys == proteins
    # Values == list of predicted bs + bs scores
    p_dict = {}
    for p in p_matrix.itertuples():
        p_dict[p[0]] = list(zip(p[1].split(','),p[2].split(',')))
    
    try:
        p_dict = {p_id: p_dict[p_id]}
    except:
        print("{} does not exist in protein library".format(p_id))
        sys.exit()

    if i_score not in ['C','dC','P','CxP','dCxP','avgC','medC','avgP','medP']:
        print("{} is not an applicable interaction score.".format(i_score))
        return

    if not os.path.exists("{}/{}-{}_vect.pickle".format(cmpd_path,fp,vect)):
        url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/cmpds/fps-{}/{}-{}_vect.pickle'.format(v,fp,vect)
        dl_file(url, '{}/{}-{}_vect.pickle'.format(cmpd_path,fp,vect))

    if not os.path.exists("{}/{}-{}_vect.pickle".format(lig_path,fp,vect)):
        url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/ligs/fps/{}-{}_vect.pickle'.format(fp,vect)
        dl_file(url, '{}/{}-{}_vect.pickle'.format(lig_path,fp,vect))

    # Load compound and ligand fingerprint pickles
    with open('{}/{}-{}_vect.pickle'.format(cmpd_path,fp,vect), 'rb') as f:
        c_fps = pd.read_pickle(f)
    with open('{}/{}-{}_vect.pickle'.format(lig_path,fp,vect), 'rb') as f:
        l_fps = pd.read_pickle(f)

    try:
        check = c_fps[c_id]
    except:
        print("{} does not exist in compound library".format(c_id))
        sys.exit()

    print("Interaction between {} and {}.".format(c_id,p_id))

    score = calc_scores(c_id,c_fps,l_fps,p_dict,dist,p_cutoff,c_cutoff,percentile_cutoff,i_score,nr_ligs,lig_name)
    print("Interaction score between {} and {} = {}".format(c_id,p_id,score[1][0]))

    end = time.time()
    print_time(end-start) 
    
    return(score[1][0])


def generate_matrix(v="v2.2", fp="rd_ecfp4", vect="int", dist="dice", org="nrpdb", bs="coach", c_cutoff=0.0,
                    p_cutoff=0.0, percentile_cutoff=0.0, i_score="P", out_file='', out_path=".", nr_ligs=True,
                    approved_only=False, lig_name=False, lib_path='', prot_path='', ncpus=1, pbar=True):
    """!
    Generate a matrix using our in-house protocol BANDOCK.

    @param v str: version to use (supports v2.2 - v2.5)
    @param fp str: the chemical fingerprint to use (rd_ecfp4, rd_ecfp10, etc)
    @param vect str: integer "int" or binary "bit" vector for fingerprint
    @param dist str: use Sorenson-Dice "dice" for vect="int" and Tanimoto "tani" for vect="bit"
    @param org str: protein library to use ('nrpdb' or 'homo_sapien')
    @param bs str: the method to use, just use "coach"
    @param c_cutoff float: minimum Cscore (Tanimoto/Dice similarity score) to consider for scoring
    @param p_cutoff float: minimum Pscore (binding site score from COACH) to consider for scoring
    @param percentile_cutoff float: %ile cutoff for fingerprint similarity scores in 'dC' scoring protocols
    @param i_score str: the scoring protocol to use ('P', 'C', 'dC', 'CxP', dCxP')
    @param out_file str: filename of the output matrix
    @param out_path str: path to the output matrix
    @param nr_ligs bool: use only the non-redundant set of ligands for 'dC' scoring protocols (recommended)
    @param approved_only bool: use only approved drugs to create the matrix
    @param lig_name bool: output the ligand chosen for the compound-protein interaction score instead of the score
    @param lib_path str: specify a local compound fingerprint set for custom analyses
    @param prot_path str: specify a local protein library for custom analyses
    @param ncpus int: number of cores to run on
    @return Returns None
    """

    def print_time(s):
        if s >= 60:
            m = s / 60.0
            s %= 60.0
            if m >= 60.0:
                h = m / 60.0
                m %= 60.0
                return f"{h:.0f} hr {m:.0f} min {s:.0f} s"
            else:
                return f"{m:.0f} min {s:.0f} s"
        else:
            return f"{s:.0f} s"

    print("Generating CANDO matrix...")
    start = time.time()

    if org=='test':
        pre = "."
    else:
        pre = os.path.dirname(__file__) + "/data/v2.2+/"

    lig_path = "{}/ligs/fps".format(pre)
    if not lib_path:
        cmpd_path = "{}/cmpds/fps-{}".format(pre,v)
        map_path = "{}/mappings".format(pre)
    else:
        cmpd_path = "{0}/{1}/cmpds/fps-{1}".format(lib_path,v)
        map_path = "{0}/{1}/mappings".format(lib_path,v)

    if out_file == '':
        if percentile_cutoff != 0.0:
            if approved_only:
                out_file = "{}-{}-{}-{}-{}-percentile{}-p{}-{}-approved.tsv".format(fp,vect,dist,org,bs,percentile_cutoff,p_cutoff,i_score)
            else:
                out_file = "{}-{}-{}-{}-{}-percentile{}-p{}-{}.tsv".format(fp,vect,dist,org,bs,percentile_cutoff,p_cutoff,i_score)
        else:
            if approved_only:
                out_file = "{}-{}-{}-{}-{}-c{}-p{}-{}-approved.tsv".format(fp,vect,dist,org,bs,c_cutoff,p_cutoff,i_score)
            else:
                out_file = "{}-{}-{}-{}-{}-c{}-p{}-{}.tsv".format(fp,vect,dist,org,bs,c_cutoff,p_cutoff,i_score)

    if not out_path and not lib_path:
        out_path = '{}/matrices/{}'.format(pre,v)
    elif not out_path and lib_path:
        out_path = '{}/{}/matrices'.format(lib_path,v)
    os.makedirs(out_path, exist_ok=True)

    # Remove redundant ligands from full list
    # Especially important for percentile calculations
    if nr_ligs:
        if not os.path.exists("{}/mappings/nr_ligs.csv".format(pre)):
            url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/mappings/nr_ligs.csv'
            dl_file(url, '{}/mappings/nr_ligs.csv'.format(pre))
        nr_ligs = pd.read_csv("{}/mappings/nr_ligs.csv".format(pre),header=None)
    nr_ligs = nr_ligs[0].values.flatten()

    # Download protein matrix if it does not exist
    if not prot_path:
        if not os.path.exists("{}/prots/{}-{}.tsv".format(pre,org,bs)):
            url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/prots/{}-{}.tsv'.format(org,bs)
            dl_file(url, '{}/prots/{}-{}.tsv'.format(pre,org,bs))
        p_matrix = pd.read_csv("{}/prots/{}-{}.tsv".format(pre,org,bs),sep='\t',header=None,index_col=0)
    else:
        p_matrix = pd.read_csv("{}/{}-{}.tsv".format(prot_path,org,bs),sep='\t',header=None,index_col=0)

    
    # Create dictionary of lists
    # Keys == proteins
    # Values == list of predicted bs + bs scores
    p_dict = {}
    for p in p_matrix.itertuples():
        p_dict[p[0]] = list(zip(p[1].split(','),p[2].split(',')))

    if i_score not in ['C','dC','P','CxP','dCxP','avgC','medC','avgP','medP']:
        print("{} is not an applicable interaction score.".format(i_score))
        return

    if not os.path.exists("{}/{}-{}_vect.pickle".format(cmpd_path,fp,vect)):
        url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/cmpds/fps-{}/{}-{}_vect.pickle'.format(v,fp,vect)
        dl_file(url, '{}/{}-{}_vect.pickle'.format(cmpd_path,fp,vect))

    if not os.path.exists("{}/{}-{}_vect.pickle".format(lig_path,fp,vect)):
        url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/ligs/fps/{}-{}_vect.pickle'.format(fp,vect)
        dl_file(url, '{}/{}-{}_vect.pickle'.format(lig_path,fp,vect))

    # Load compound and ligand fingerprint pickles
    with open('{}/{}-{}_vect.pickle'.format(cmpd_path,fp,vect), 'rb') as f:
        c_fps = pd.read_pickle(f)
    with open('{}/{}-{}_vect.pickle'.format(lig_path,fp,vect), 'rb') as f:
        l_fps = pd.read_pickle(f)

    if approved_only:
        if not os.path.exists("{}/drugbank-{}-approved.tsv".format(map_path,v)):
            url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/mappings/drugbank-{}-approved.tsv'.format(v)
            dl_file(url, '{}/drugbank-{}-approved.tsv'.format(map_path,v))
        approved_df = pd.read_csv('{}/drugbank-{}-approved.tsv'.format(map_path,v),sep='\t',index_col=0)
        c_list = approved_df.index
    else:
        c_list = list(c_fps.keys())

    if ncpus > 1:
        pool = mp.Pool(ncpus)
        scores = pool.starmap_async(calc_scores, [(c,c_fps,l_fps,p_dict,dist,p_cutoff,c_cutoff,percentile_cutoff,i_score,nr_ligs,lig_name) for c in c_list], chunksize=20).get()
        pool.close
        pool.join
    else:
        bar = tqdm(c_list) if pbar else c_list
        scores = [calc_scores(c,c_fps,l_fps,p_dict,dist,p_cutoff,c_cutoff,percentile_cutoff,i_score,nr_ligs,lig_name) for c in bar]
    scores = {d[0]:d[1] for d in scores}

    mat = pd.DataFrame.from_dict(scores)
    mat.sort_index(axis=1,inplace=True)
    mat.rename(index=dict(zip(range(len(p_matrix.index)), p_matrix.index)), inplace=True)
   
    mat.to_csv("{}/{}".format(out_path,out_file), sep='\t', index=True, header=False, float_format='%.3f')
    print(f"  Matrix written to {out_path}/{out_file}.")

    tot_time = print_time(time.time()-start)
    print(f"Matrix generation completed in {tot_time}.\n")


def calc_scores(c,c_fps,l_fps,p_dict,dist,pscore_cutoff=0.0,cscore_cutoff=0.0,percentile_cutoff=0.0,i_score='P',nr_ligs=[],lig_name=False):
    if i_score in ['dC','dCxP'] or percentile_cutoff != 0.0:
        if dist == 'dice':
            all_scores = DataStructs.BulkDiceSimilarity(c_fps[c],l_fps.loc[nr_ligs,0].values.tolist())
        elif dist == 'tani':
            all_scores = DataStructs.BulkTanimotoSimilarity(c_fps[c],l_fps.loc[nr_ligs,0].values.tolist())
        elif dist == 'cos':
            all_scores = DataStructs.BulkCosineSimilarity(c_fps[c],l_fps.loc[nr_ligs,0].values.tolist())
    if percentile_cutoff != 0.0:
        cscore_cutoff = np.percentile(all_scores,percentile_cutoff)
    scores = []
    for p in p_dict.keys():
        li = [i[0:2] for i in p_dict[p] if i[0] in l_fps.index and float(i[1]) >= pscore_cutoff]
        if li:
            li_bs, li_score = zip(*li)
            li_bs = list(li_bs)
            li_score = list(li_score)
        else:
            li_bs = li_score = []
        x = l_fps.loc[li_bs,0].values.tolist()
        y = l_fps.loc[li_bs].index.tolist()
        z = [float(li_score[li_bs.index(i)]) for i in y]

        try:
            if dist == 'dice':
                temp_scores = list(zip(y,DataStructs.BulkDiceSimilarity(c_fps[c],x),z))
            elif dist == 'tani':
                temp_scores = list(zip(y,DataStructs.BulkTanimotoSimilarity(c_fps[c],x),z))
            elif dist == 'cos':
                temp_scores = list(zip(y,DataStructs.BulkCosineSimilarity(c_fps[c],x),z))

            #Cscore cutoff
            temp_scores = [i for i in temp_scores if float(i[1]) >= cscore_cutoff]
    
            if i_score == 'dCxP':
                temp = sorted(temp_scores, key = lambda i:(i[1],i[2]),reverse=True)[0]
                if not lig_name:
                    c_score = stats.percentileofscore(all_scores,temp[1])/100.0
                    scores.append(float(c_score) * float(temp[2]))
                else:
                    scores.append(temp[0])
            elif i_score == 'CxP':
                temp = sorted(temp_scores, key = lambda i:(i[1],i[2]),reverse=True)[0]
                if not lig_name:
                    c_score = temp[1]
                    p_score = temp[2]
                    scores.append(float(temp[1]) * float(temp[2]))
                else:
                    scores.append(temp[0])
            elif i_score == 'P':
                temp = sorted(temp_scores, key = lambda i:(i[1],i[2]),reverse=True)[0]
                if not lig_name:
                    scores.append(float(temp[2]))
                else:
                    scores.append(temp[0])
            elif i_score == 'avgP':
                # Will produce a warning when li_score is empty
                # temp_p will then == nan, so we check for that
                # append 0.00 if True.
                temp_p = np.mean(li_score)
                if not np.isnan(temp_p):
                    scores.append(temp_p)
                else:
                    scores.append(0.000)
            elif i_score == 'medP':
                temp_p = np.median(li_score)
                if not np.isnan(temp_p):
                    scores.append(temp_p)
                else:
                    scores.append(0.000)
            elif i_score == 'dC':
                temp = sorted(temp_scores, key = lambda i:(i[1],i[2]),reverse=True)[0]
                if not lig_name:
                    scores.append(stats.percentileofscore(all_scores, temp[1]) / 100.0)
                else:
                    scores.append(temp[0])
            elif i_score == 'C':
                temp = sorted(temp_scores, key = lambda i:(i[1],i[2]),reverse=True)[0]
                if not lig_name:
                    scores.append(temp[1])
                else:
                    scores.append(temp[0])
            elif i_score == 'avgC':
                temp_c = np.mean(list(zip(*temp_scores))[1])
                if not np.isnan(temp_c):
                    scores.append(temp_c)
                else:
                    scores.append(0.000)
            elif i_score == 'medC':
                temp_c = np.median(list(zip(*temp_scores))[1])
                if not np.isnan(temp_c):
                    scores.append(temp_c)
                else:
                    scores.append(0.000)
        except:
            if not lig_name:
                scores.append(0.000)
            else:
                scores.append("None")
    return (c, scores)


def generate_signature(cmpd_file, fp="rd_ecfp4", vect="int", dist="dice", org="nrpdb", bs="coach", c_cutoff=0.0,
                       p_cutoff=0.0, percentile_cutoff=0.0, i_score="P", out_file='', out_path=".", nr_ligs=True,
                       prot_path=''):
    """!
       Generate an interaction signature for a query compound using our in-house protocol BANDOCK. Note: the parameters
       for this function MUST MATCH the parameters used to generate the matrix in use. Otherwise, the scores will be
       incompatible.

       @param cmpd_file str: filepath to an input mol file
       @param fp str: the chemical fingerprint to use (rd_ecfp4, rd_ecfp10, etc)
       @param vect str: integer "int" or binary "bit" vector for fingerprint
       @param dist str: use Sorenson-Dice "dice" for vect="int" and Tanimoto "tani" for vect="bit"
       @param org str: protein library to use ('nrpdb' or 'homo_sapien')
       @param bs str: the method to use, just use "coach"
       @param c_cutoff float: minimum Cscore (Tanimoto/Dice similarity score) to consider for scoring
       @param p_cutoff float: minimum Pscore (binding site score from COACH) to consider for scoring
       @param percentile_cutoff float: %ile cutoff for fingerprint similarity scores in 'dC' scoring protocols
       @param i_score str: the scoring protocol to use ('P', 'C', 'dC', 'CxP', dCxP')
       @param out_file str: filename of the output signature
       @param out_path str: path to the output signature
       @param nr_ligs bool: use only the non-redundant set of ligands for 'dC' scoring protocols (recommended)
       @param prot_path str: specify a local protein library for custom analyses
       @return Returns None
       """
    def print_time(s):
        if s >= 60:
            m = s / 60.0
            s %= 60.0
            if m >= 60.0:
                h = m / 60.0
                m %= 60.0
                print("Signature generation took {:.0f} hr {:.0f} min {:.0f} s to finish.".format(h, m, s))
            else:
                print("Signature generation took {:.0f} min {:.0f} s to finish.".format(m, s))
        else:
            print("signature generation took {:.0f} s to finish.".format(s))

    print("Generating CANDO signature...")
    start = time.time()
    if org=='test':
        pre = "."
    else:
        pre = os.path.dirname(__file__) + "/data/v2.2+/"
    lig_path = "{}/ligs/fps/".format(pre)
    if out_file == '':
        if percentile_cutoff != 0.0:
            out_file = "{}/cmpd_0-{}-{}-{}-{}-{}-percentile{}-p{}-{}.tsv".format(out_path,fp,vect,dist,org,bs,percentile_cutoff,p_cutoff,i_score)
        else:
            out_file = "{}/cmpd_0-{}-{}-{}-{}-{}-c{}-p{}-{}.tsv".format(out_path,fp,vect,dist,org,bs,c_cutoff,p_cutoff,i_score)
    os.makedirs(out_path, exist_ok=True)

    # Remove redundant ligands from full list
    # Especially important for percentile calculations
    if nr_ligs:
        if not os.path.exists("{}/mappings/nr_ligs.csv".format(pre)):
            url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/mappings/nr_ligs.csv'
            dl_file(url, '{}/mappings/nr_ligs.csv'.format(pre))
        nr_ligs = pd.read_csv("{}/mappings/nr_ligs.csv".format(pre),header=None)
    nr_ligs = nr_ligs[0].values.flatten()

    # Download protein matrix if it does not exist
    if not prot_path:
        if not os.path.exists("{}/prots/{}-{}.tsv".format(pre,org,bs)):
            url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/prots/{}-{}.tsv'.format(org,bs)
            dl_file(url, '{}/prots/{}-{}.tsv'.format(pre,org,bs))
        p_matrix = pd.read_csv("{}/prots/{}-{}.tsv".format(pre,org,bs),sep='\t',header=None,index_col=0)
    else:
        p_matrix = pd.read_csv("{}/{}-{}.tsv".format(prot_path,org,bs),sep='\t',header=None,index_col=0)
    
    # Create dictionary of lists
    # Keys == proteins
    # Values == list of predicted bs + bs scores
    p_dict = {}
    for p in p_matrix.itertuples():
        p_dict[p[0]] = list(zip(p[1].split(','),p[2].split(',')))

    if i_score not in ['C','dC','P','CxP','dCxP','avgC','medC','avgP','medP']:
        print("{} is not an applicable interaction score.".format(i_score))
        return

    nc = Chem.MolFromMolFile(cmpd_file)
    nc = Chem.RemoveHs(nc)
    name = nc.GetProp("_Name")

    c_fps = {}
    rad = int(int(fp[7:])/2)
    if fp[3]=='f':
        features = True
    else:
        features = False

    if vect=='int':
        c_fps[0] = AllChem.GetMorganFingerprint(nc,rad,useFeatures=features)
    else:
        bits = int(vect[:4])
        c_fps[0] = AllChem.GetMorganFingerprintAsBitVect(nc,rad,useFeatures=features,nBits=bits)

    if not os.path.exists("{}/{}-{}_vect.pickle".format(lig_path,fp,vect)):
        url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/ligs/fps/{}-{}_vect.pickle'.format(fp,vect)
        dl_file(url, '{}/{}-{}_vect.pickle'.format(lig_path,fp,vect))

    # Load ligand fingerprint pickles
    with open('{}/{}-{}_vect.pickle'.format(lig_path,fp,vect), 'rb') as f:
        l_fps = pd.read_pickle(f)

    scores = calc_scores(0,c_fps,l_fps,p_dict,dist,p_cutoff,c_cutoff,percentile_cutoff,i_score,nr_ligs)
    #scores = pool.starmap_async(calc_scores, [(c,c_fps,l_fps,p_dict,dist,p_cutoff,c_cutoff,percentile_cutoff,i_score,nr_ligs) for c in c_list]).get()
    scores = {scores[0]:scores[1]}

    mat = pd.DataFrame.from_dict(scores)
    mat.sort_index(axis=1,inplace=True)
    mat.rename(index=dict(zip(range(len(p_matrix.index)), p_matrix.index)), inplace=True)
   
    mat.to_csv("{}/{}".format(out_path,out_file), sep='\t', index=True, header=False, float_format='%.3f')
    
    end = time.time()
    print("Signature written to {}/{}.".format(out_path,out_file))
    print_time(end-start) 
    return(mat.iloc[:,0].values)

def generate_signature_smi(smi, fp="rd_ecfp4", vect="int", dist="dice", org="nrpdb", bs="coach", c_cutoff=0.0,
                       p_cutoff=0.0, percentile_cutoff=0.0, i_score="P", save_sig=False, out_file='', out_path=".", nr_ligs=True,
                       prot_path='', lig_name=False):
    """!
       Generate an interaction signature for a query compound using our in-house protocol BANDOCK. Note: the parameters
       for this function MUST MATCH the parameters used to generate the matrix in use. Otherwise, the scores will be
       incompatible.

       @param smi str: SMILES string of compound for which you want to generate a BANDOCK CANDO signature
       @param fp str: the chemical fingerprint to use (rd_ecfp4, rd_ecfp10, etc)
       @param vect str: integer "int" or binary "bit" vector for fingerprint
       @param dist str: use Sorenson-Dice "dice" for vect="int" and Tanimoto "tani" for vect="bit"
       @param org str: protein library to use ('nrpdb' or 'homo_sapien')
       @param bs str: the method to use, just use "coach"
       @param c_cutoff float: minimum Cscore (Tanimoto/Dice similarity score) to consider for scoring
       @param p_cutoff float: minimum Pscore (binding site score from COACH) to consider for scoring
       @param percentile_cutoff float: %ile cutoff for fingerprint similarity scores in 'dC' scoring protocols
       @param i_score str: the scoring protocol to use ('P', 'C', 'dC', 'CxP', dCxP')
       @param save_sig bool: Save signature to file or not. If False then out_file and out_path are not used. (default: False)
       @param out_file str: filename of the output signature
       @param out_path str: path to the output signature
       @param nr_ligs bool: use only the non-redundant set of ligands for 'dC' scoring protocols (recommended)
       @param prot_path str: specify a local protein library for custom analyses
       @param lig_name bool: output the ligand chosen for the compound-protein interaction score instead of the score
       @return Returns None
       """
    def print_time(s):
        if s >= 60:
            m = s / 60.0
            s -= m * 60.0
            if m >= 60.0:
                h = m / 60.0
                m -= h * 60.0
                print("Signature generation took {:.0f} hr {:.0f} min {:.0f} s to finish.".format(h, m, s))
            else:
                print("Signature generation took {:.0f} min {:.0f} s to finish.".format(m, s))
        else:
            print("signature generation took {:.0f} s to finish.".format(s))

    print("Generating CANDO signature...")
    start = time.time()
    if org=='test':
        pre = "."
    else:
        pre = os.path.dirname(__file__) + "/data/v2.2+/"
    lig_path = "{}/ligs/fps/".format(pre)
    if out_file == '':
        if percentile_cutoff != 0.0:
            out_file = "{}/cmpd_0-{}-{}-{}-{}-{}-percentile{}-p{}-{}.tsv".format(out_path,fp,vect,dist,org,bs,percentile_cutoff,p_cutoff,i_score)
        else:
            out_file = "{}/cmpd_0-{}-{}-{}-{}-{}-c{}-p{}-{}.tsv".format(out_path,fp,vect,dist,org,bs,c_cutoff,p_cutoff,i_score)
    os.makedirs(out_path, exist_ok=True)

    # Remove redundant ligands from full list
    # Especially important for percentile calculations
    if nr_ligs:
        if not os.path.exists("{}/mappings/nr_ligs.csv".format(pre)):
            url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/mappings/nr_ligs.csv'
            dl_file(url, '{}/mappings/nr_ligs.csv'.format(pre))
        nr_ligs = pd.read_csv("{}/mappings/nr_ligs.csv".format(pre),header=None)
    nr_ligs = nr_ligs[0].values.flatten()

    # Download protein matrix if it does not exist
    if not prot_path:
        if not os.path.exists("{}/prots/{}-{}.tsv".format(pre,org,bs)):
            url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/prots/{}-{}.tsv'.format(org,bs)
            dl_file(url, '{}/prots/{}-{}.tsv'.format(pre,org,bs))
        p_matrix = pd.read_csv("{}/prots/{}-{}.tsv".format(pre,org,bs),sep='\t',header=None,index_col=0)
    else:
        p_matrix = pd.read_csv("{}/{}-{}.tsv".format(prot_path,org,bs),sep='\t',header=None,index_col=0)
    
    # Create dictionary of lists
    # Keys == proteins
    # Values == list of predicted bs + bs scores
    p_dict = {}
    for p in p_matrix.itertuples():
        p_dict[p[0]] = list(zip(p[1].split(','),p[2].split(',')))

    if i_score not in ['C','dC','P','CxP','dCxP','avgC','medC','avgP','medP']:
        print("{} is not an applicable interaction score.".format(i_score))
        return

    try:
        nc = Chem.MolFromSmiles(smi)
        nc = Chem.RemoveHs(nc)
        #name = nc.GetProp("_Name")
    except:
        print("SMILES string cannot be processed.")
        sys.exit()

    if nc is None:
        print("SMILES string cannot be processed.")
        sys.exit()

    c_fps = {}
    rad = int(int(fp[7:])/2)
    if fp[3]=='f':
        features = True
    else:
        features = False

    if vect=='int':
        c_fps[0] = AllChem.GetMorganFingerprint(nc,rad,useFeatures=features)
    else:
        bits = int(vect[:4])
        c_fps[0] = AllChem.GetMorganFingerprintAsBitVect(nc,rad,useFeatures=features,nBits=bits)

    if not os.path.exists("{}/{}-{}_vect.pickle".format(lig_path,fp,vect)):
        url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/ligs/fps/{}-{}_vect.pickle'.format(fp,vect)
        dl_file(url, '{}/{}-{}_vect.pickle'.format(lig_path,fp,vect))

    # Load ligand fingerprint pickles
    with open('{}/{}-{}_vect.pickle'.format(lig_path,fp,vect), 'rb') as f:
        l_fps = pd.read_pickle(f)

    scores = calc_scores(0,c_fps,l_fps,p_dict,dist,p_cutoff,c_cutoff,percentile_cutoff,i_score,nr_ligs,lig_name)
    #scores = pool.starmap_async(calc_scores, [(c,c_fps,l_fps,p_dict,dist,p_cutoff,c_cutoff,percentile_cutoff,i_score,nr_ligs) for c in c_list]).get()
    scores = {scores[0]:scores[1]}

    mat = pd.DataFrame.from_dict(scores)
    mat.sort_index(axis=1,inplace=True)
    mat.rename(index=dict(zip(range(len(p_matrix.index)), p_matrix.index)), inplace=True)
    print("Signature generation complete.")
    
    if save_sig:
        mat.to_csv("{}/{}".format(out_path,out_file), sep='\t', index=True, header=False, float_format='%.3f')
        print("Signature written to {}/{}.".format(out_path,out_file))
    
    end = time.time()
    print_time(end-start)
    return(mat)
    #return(mat.iloc[:,0].values)



def add_cmpds(cmpd_list, file_type='smi', fp="rd_ecfp4", vect="int", cmpd_dir=".", v=None, map_indications='v2.3'):
    """!
   Add new compounds to an existing CANDO Compound library, or create a new Compound library using our in-house protocol
   BANDOCK.

   @param cmpd_list str: filepath to all input compounds
   @param fp str: the chemical fingerprint to use (rd_ecfp4, rd_ecfp10, etc)
   @param vect str: integer "int" or binary "bit" vector for fingerprint
   @param cmpd_dir str: ??
   @param v str: ??
   @param map_indications str: CANDO version number to string match exact names from compound file to existing ind_map
   @return Returns None
   """
    start = time.time()
    if v=='test.0':
        pre = "."
    else:
        pre = os.path.dirname(__file__) + "/data/v2.2+/"
    # List of new compounds loaded into df
    ncs = pd.read_csv(cmpd_list, sep='\t', header=None)
  
    vs = ['v2.2', 'v2.3', 'v2.4', 'v2.5', 'test.0']
    if v in vs:
        # Redundant with future lines. 
        # Remove future lines and implement them into get_data()
        #get_data(v=v, org=None)
        curr_v = v
        print("Adding new compounds to compound library {}...".format(curr_v))
        t = curr_v.split('.')
        t[-1] = str(int(t[-1])+1)
        new_v = '.'.join(t)
        print("New compound library is {}.".format(new_v))
        
        curr_cmpd_path = "{}/cmpds/fps-{}/".format(pre, curr_v)
        if not os.path.exists("{}/cmpds/fps-{}/{}-{}_vect.pickle".format(pre, curr_v, fp, vect)):
            url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/cmpds/fps-{}/{}-{}_vect.pickle'.format(curr_v,
                                                                                                               fp, vect)
            dl_file(url, '{}/cmpds/fps-{}/{}-{}_vect.pickle'.format(pre, curr_v, fp, vect))
        cmpd_path = "{}/cmpds/fps-{}/".format(pre, new_v)
        os.makedirs(cmpd_path, exist_ok=True)

        shutil.copy('{0}/{1}-{2}_vect.pickle'.format(curr_cmpd_path, fp, vect),\
                    '{0}/{1}-{2}_vect.pickle'.format(cmpd_path, fp, vect))
        #os.system("cp {0}/{2}-{3}_vect.pickle {1}/{2}-{3}_vect.pickle".format(curr_cmpd_path, cmpd_path, fp, vect))
        
        if not os.path.exists("{}/mappings/drugbank-{}.tsv".format(pre, curr_v)):
            url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/mappings/drugbank-{}.tsv'.format(curr_v)
            dl_file(url, '{}/mappings/drugbank-{}.tsv'.format(pre, curr_v))
        
        d_map = pd.read_csv("{}/mappings/drugbank-{}.tsv".format(pre, curr_v), sep='\t')
        
        if not os.path.exists("{}/mappings/drugbank2ctd-{}.tsv".format(pre, curr_v)):
            url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/mappings/drugbank2ctd-{}.tsv'.format(curr_v)
            dl_file(url, '{}/mappings/drugbank2ctd-{}.tsv'.format(pre, curr_v))

        shutil.copy('{0}/mappings/drugbank2ctd-{1}.tsv'.format(pre, curr_v),\
                    '{0}/mappings/drugbank2ctd-{1}.tsv'.format(pre, new_v))
        #os.system("cp {0}/mappings/drugbank2ctd-{1}.tsv {0}/mappings/drugbank2ctd-{2}.tsv".format(pre, curr_v, new_v))

        if not os.path.exists("{}/cmpds/fps-{}/inchi_keys.pickle".format(pre, curr_v)):
            url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/cmpds/fps-{}/inchi_keys.pickle'.format(curr_v)
            dl_file(url, '{}/cmpds/fps-{}/inchi_keys.pickle'.format(pre, curr_v))
  
        with open('{}/inchi_keys.pickle'.format(curr_cmpd_path), 'rb') as f:
            inchi_dict = pickle.load(f)
        cmpd_num = len(inchi_dict)

        for c in ncs.itertuples(index=False):
            try:
                if file_type == 'mol':
                    nc = Chem.MolFromMolFile("{}/{}.mol".format(cmpd_dir, c[0]))
                elif file_type == 'smi':
                    nc = Chem.MolFromSmiles("{}".format(c[0]))
                    nc.SetProp("_Name", c[1])
                nc = Chem.RemoveHs(nc)
            except:
                print("{} cannot load this molecule.".format(c[0]))
                continue
            name = nc.GetProp("_Name")
            inchi_key = Chem.MolToInchiKey(nc)
            try:
                match = str(inchi_dict[inchi_key])
            except:
                match = None
            if match:
                print("    {} is the same as {} - {} in the library".format(name, int(match),
                                                                            d_map.loc[(d_map['CANDO_ID'] == int(match)),
                                                                                      'GENERIC_NAME'].values[0], match))
                continue
            else:
                print("    Adding compound {} - {}".format(cmpd_num,name))
            
            with open('{}/inchi_keys.pickle'.format(cmpd_path), 'wb') as f:
                inchi_dict[inchi_key] = cmpd_num
                pickle.dump(inchi_dict, f)
           
            d_map = pd.concat([d_map, pd.DataFrame([[cmpd_num, 'NA', name, 'other']],
                                              columns=['CANDO_ID', 'DRUGBANK_ID', 'GENERIC_NAME', 'DRUG_GROUPS'])], axis=1)
            rad = int(int(fp[7:])/2)
            if fp[3] == 'f':
                features = True
            else:
                features = False

            if vect == 'int':
                with open('{}/{}-{}_vect.pickle'.format(cmpd_path, fp, vect), 'rb') as f:
                    c_fps = pickle.load(f)
                c_fps[cmpd_num] = AllChem.GetMorganFingerprint(nc, rad, useFeatures=features)
                with open('{}/{}-{}_vect.pickle'.format(cmpd_path, fp, vect), 'wb') as f:
                    pickle.dump(c_fps, f)
            else:
                bits = int(vect[:4])
                with open('{}/{}-{}_vect.pickle'.format(cmpd_path, fp, vect), 'rb') as f:
                    c_fps = pickle.load(f)
                c_fps[cmpd_num] = AllChem.GetMorganFingerprintAsBitVect(nc, rad, useFeatures=features, nBits=bits)
                with open('{}/{}-{}_vect.pickle'.format(cmpd_path, fp, vect), 'wb') as f:
                    pickle.dump(c_fps, f)
            cmpd_num += 1
    elif v and v not in vs:
        new_v = v
        print("Creating new compound library {}...".format(new_v))
        print("The library will be built at {}/{}.".format(os.getcwd(), new_v))
        os.makedirs(new_v, exist_ok=True)
        os.makedirs("{}/cmpds".format(new_v), exist_ok=True)
        os.makedirs("{}/mappings".format(new_v), exist_ok=True)
        cmpd_path = "{0}/cmpds/fps-{0}/".format(new_v)
        os.makedirs(cmpd_path, exist_ok=True)
        d_map = pd.DataFrame(columns=['CANDO_ID', 'DRUGBANK_ID', 'GENERIC_NAME', 'DRUG_GROUPS'])

        cid2name = {}
        cname2inds = {}
        if map_indications:
            if not os.path.exists("{}/mappings/drugbank-{}.tsv".format(pre, map_indications)):
                url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/mappings/' \
                      'drugbank-{}.tsv'.format(map_indications)
                dl_file(url, '{}/mappings/drugbank-{}.tsv'.format(pre, map_indications))
            if not os.path.exists("{}/mappings/drugbank2ctd-{}.tsv".format(pre, map_indications)):
                url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/mappings/' \
                      'drugbank2ctd-{}.tsv'.format(map_indications)
                dl_file(url, '{}/mappings/drugbank2ctd-{}.tsv'.format(pre, map_indications))

            fcm = open('{}/mappings/drugbank-{}.tsv'.format(pre, map_indications), 'r', encoding="utf8")
            cmls = fcm.readlines()
            fcm.close()
            for cml in cmls[1:]:
                cls = cml.split('\t')
                cid = cls[0]
                cname = cls[2]
                cid2name[cid] = cname

            fim = open('{}/mappings/drugbank2ctd-{}.tsv'.format(pre, map_indications), 'r', encoding="utf8")
            imls = fim.readlines()
            fim.close()
            for iml in imls[1:]:
                ils = iml.split('\t')
                cid = ils[0]
                indname = ils[1]
                indid = ils[2]
                cname = cid2name[cid]
                if cname in cname2inds:
                    if (indname, indid) not in cname2inds[cname]:
                        cname2inds[cname].append((indname, indid))
                else:
                    cname2inds[cname] = [(indname, indid)]

        cmpd_num = 0
        # Create new fingerprint dict and save it to pickle for future use
        c_fps = {}
        if vect == 'int':
            with open('{}/{}-{}_vect.pickle'.format(cmpd_path, fp, vect), 'wb') as f:
                pickle.dump(c_fps, f)
        else:
            bits = int(vect[:4])
            with open('{}/{}-{}_vect.pickle'.format(cmpd_path, fp, vect), 'wb') as f:
                pickle.dump(c_fps, f)
        # Create new inchi dict
        inchi_dict = {}

        if map_indications:
            foind = open("{0}/mappings/inds-{0}.tsv".format(new_v), 'w', encoding="utf8")
            foind.write('CANDO_ID\tINDICATION_NAME\tMESH_ID\tINDICATION_ID\n')
            ind2id = {}
            curr_ind_id = 0

        for c in ncs.itertuples(index=False):
            try:
                if file_type == 'mol':
                    nc = Chem.MolFromMolFile(f"{cmpd_dir}/{c[0]}.mol")
                    name = nc.GetProp("_Name")
                elif file_type == 'smi':
                    nc = Chem.MolFromSmiles(c[0])
                    nc.SetProp("_Name", c[1])
                nc = Chem.RemoveHs(nc)
            except:
                print("{} cannot load this molecule.".format(c[0]))
                continue
            name = nc.GetProp("_Name")
            inchi_key = Chem.MolToInchiKey(nc)
            try:
                match = str(inchi_dict[inchi_key])
            except:
                match = None
            if match:
                print("    {} is the same as {} - {} in the library".format(name, int(match),
                                                                            d_map.loc[(d_map['CANDO_ID'] == int(match)),
                                                                                      'GENERIC_NAME'].values[0], match))
                continue
            else:
                print("    Adding compound {} - {}".format(cmpd_num, name))
            
            with open('{}/inchi_keys.pickle'.format(cmpd_path), 'wb') as f:
                inchi_dict[inchi_key] = cmpd_num
                pickle.dump(inchi_dict, f)
           
            d_map = d_map.append(pd.DataFrame([[cmpd_num, 'NA', name, 'other']],
                                              columns=['CANDO_ID', 'DRUGBANK_ID', 'GENERIC_NAME', 'DRUG_GROUPS']),
                                 ignore_index=True)

            if map_indications:
                if name in cname2inds:
                    inds = cname2inds[name]
                    for ind in inds:
                        if ind in ind2id:
                            indid = ind2id[ind]
                        else:
                            indid = curr_ind_id
                            ind2id[ind] = curr_ind_id
                            curr_ind_id += 1
                        foind.write('{}\t{}\t{}\t{}\n'.format(cmpd_num, ind[0], ind[1], indid))
            
            rad = int(int(fp[7:])/2)
            if fp[3] == 'f':
                features = True
            else:
                features = False

            if vect == 'int':
                with open('{}/{}-{}_vect.pickle'.format(cmpd_path, fp, vect), 'rb') as f:
                    c_fps = pickle.load(f)
                c_fps[cmpd_num] = AllChem.GetMorganFingerprint(nc, rad, useFeatures=features)
                with open('{}/{}-{}_vect.pickle'.format(cmpd_path, fp, vect), 'wb') as f:
                    pickle.dump(c_fps, f)
            else:
                with open('{}/{}-{}_vect.pickle'.format(cmpd_path, fp, vect), 'rb') as f:
                    c_fps = pickle.load(f)
                c_fps[cmpd_num] = AllChem.GetMorganFingerprintAsBitVect(nc, rad, useFeatures=features, nBits=bits)
                with open('{}/{}-{}_vect.pickle'.format(cmpd_path, fp, vect), 'wb') as f:
                    pickle.dump(c_fps, f)
            cmpd_num += 1
 
    elif not v:
        new_v = "v0.0"
        print("Creating new compound library {}...".format(new_v))
        cmpd_path = "{0}/cmpds/fps-{0}/".format(new_v)
        os.makedirs(cmpd_path, exist_ok=True)
        d_map = pd.DataFrame(columns=['CANDO_ID', 'DRUGBANK_ID', 'GENERIC_NAME', 'DRUG_GROUPS'])
        cmpd_num = 0
        # Create new fingerprint dict and save it to pickle for future use
        c_fps = {}
        if vect=='int':
            with open('{}/{}-{}_vect.pickle'.format(cmpd_path,fp,vect), 'wb') as f:
                pickle.dump(c_fps, f)
        else:
            bits = int(vect[:4])
            with open('{}/{}-{}_vect.pickle'.format(cmpd_path,fp,vect), 'wb') as f:
                pickle.dump(c_fps, f)
        # Create new inchi dict
        inchi_dict = {}

        for c in ncs.itertuples(index=False):
            try:
                if file_type == 'mol':
                    nc = Chem.MolFromMolFile(f"{cmpd_dir}/{c[0]}.mol")
                elif file_type == 'smi':
                    nc = Chem.MolFromSmiles(c[0])
                    nc.SetProp("_Name", c[1])
                nc = Chem.RemoveHs(nc)
            except:
                print("{} cannot load this molecule.".format(c[0]))
                continue
            name = nc.GetProp("_Name")
            inchi_key = Chem.MolToInchiKey(nc)
            try:
                match = str(inchi_dict[inchi_key])
            except:
                match = None
            if match:
                print("    {} is the same as {} - {} in the library".format(name, int(match),
                                                                            d_map.loc[(d_map['CANDO_ID'] == int(match)),
                                                                                      'GENERIC_NAME'].values[0], match))
                continue
            else:
                print("    Adding compound {} - {}".format(cmpd_num, name))
            
            with open('{}/inchi_keys.pickle'.format(cmpd_path), 'wb') as f:
                inchi_dict[inchi_key] = cmpd_num
                pickle.dump(inchi_dict, f)
           
            d_map = pd.concat([d_map, pd.DataFrame([[cmpd_num, 'NA', name, 'other']],
                            columns=['CANDO_ID', 'DRUGBANK_ID', 'GENERIC_NAME', 'DRUG_GROUPS'])],
                            ignore_index=True)
            
            rad = int(int(fp[7:])/2)
            if fp[3] == 'f':
                features = True
            else:
                features = False

            if vect == 'int':
                with open('{}/{}-{}_vect.pickle'.format(cmpd_path, fp, vect), 'rb') as f:
                    c_fps = pickle.load(f)
                c_fps[cmpd_num] = AllChem.GetMorganFingerprint(nc, rad, useFeatures=features)
                with open('{}/{}-{}_vect.pickle'.format(cmpd_path, fp, vect), 'wb') as f:
                    pickle.dump(c_fps, f)
            else:
                with open('{}/{}-{}_vect.pickle'.format(cmpd_path, fp, vect), 'rb') as f:
                    c_fps = pickle.load(f)
                c_fps[cmpd_num] = AllChem.GetMorganFingerprintAsBitVect(nc, rad, useFeatures=features, nBits=bits)
                with open('{}/{}-{}_vect.pickle'.format(cmpd_path, fp, vect), 'wb') as f:
                    pickle.dump(c_fps, f)
            cmpd_num += 1
    os.makedirs("{}/mappings".format(new_v), exist_ok=True)
    d_map.to_csv("{0}/mappings/cmpds-{0}.tsv".format(new_v), sep='\t', index=False, na_rep='NA')
    print("Added compounds to compound library {}.\n".format(new_v))
    # Need to add functionality to handle loading a new version created by user.


def tanimoto_sparse(str1, str2):
    """!
    Calculate the tanimoto coefficient for a pair of sparse vectors

    @param str1 str: String of 1s and 0s representing the first compound fingerprint
    @param str2 str: String of 1s and 0s representing the second compound fingerprint
    @return Returns float
    """
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
    """!
    Calculate the tanimoto coefficient for a pair of dense vectors

    @param list1 list: List of positions that have a 1 in first compound fingerprint
    @param list2 list: List of positions that have a 1 in second compound fingerprint
    @return Returns float
    """
    c = [common_item for common_item in list1 if common_item in list2]
    return float(len(c))/(len(list1) + len(list2) - len(c))


def get_prot_info(p, org='homo_sapien'):
    pre = os.path.dirname(__file__) + "/data/v2.2+"
    prots_df = pd.read_csv('{}/prots/{}-metadata.tsv'.format(pre,org),index_col=0,sep='\t')
    return prots_df.loc[p].to_dict()

def get_fp_lig(fp):
    """!
    Download precompiled binding site ligand fingerprints using the given fingerprint method.

    @param fp str: Fingerprinting method used to compile each binding site ligand fingerprint
    @return Returns None
    """
    pre = os.path.dirname(__file__)
    out_file = '{}/v2.2+/ligs/{}.pickle'.format(pre, fp)
    if not os.path.exists(out_file):
        print('Downloading ligand fingerprints for {}...'.format(fp))
        url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/ligs/{}.pickle'.format(fp)
        dl_file(url, out_file)
        print("{} ligand fingerprints downloaded.".format(fp))
    else:
        print("{} ligand fingerprints have already been downloaded.".format(fp))
        print("This file can be found at {}".format(out_file))


def get_data(v="v2.2", cmpd_set='all', org='nrpdb', fp='rd_ecfp4', vect='int', dist='dice', i_score='CxP', out_path=os.path.dirname(__file__)):
    """!
    Download CANDO v2.2+ data.

    @param v str: Compound library version to use. 'v2.2' - 'v2.5' are currently supported. Default: 'v2.2'
    @param cmpd_set: Sublibrary of compounds to use. 'all' will use all compounds in library, 'approved' will only use approved drugs. Default: 'all'
    @param org str: Protein library (i.e., proteome) to use. 'nrpdb' and 'homo_sapien' are currently supported. Default: 'nrpdb'
    @param fp str: Chemical fingerprint method to use. 'rd_ecfp4' is the only currently supported method. Default: 'rd_ecfp4'
    @param vect str: Vector type for the chenical fingerprints. 'int' for an integer vector. Default: 'int'
    @param dist: Distance metric used to compare chemical fingerprints with the BANDOCK method. 'dice' is currentl supproted. Default: 'dice'
    @param i_score: Scoring protocol for the BANDOCK method. 'C', 'CxP', and 'dCxP' are currently supported. Default: 'CxP'
    @param out_path: Path to which the files will be downloaded. All data will be stored in ./data/v2.2+ at the path defined here along with subdirs for each data type, i.e., prots, cmpds, mappings, and matrices. Default: path to the installed cando.py file

    @returns Returns None
    """
    # Check v and org before moving on
    vs = ['v2.2','v2.3','v2.4','v2.5','test.0']
    #vs = ['v2.2','v2.3','v2.4','v2.5','v2.6','v2.7','v2.8','test.0']
    cmpd_sets = ['all','approved']
    orgs = ['nrpdb','homo_sapien','test','tutorial']
    #orgs = ['all','nrpdb','homo_sapien','cryptococcus','aspire','test','tutorial']
    fps = ['rd_ecfp4']
    vects = ['int']
    dists = ['dice']
    i_scores = ['C','CxP','dCxP']
    if v not in vs:
        print(f"{v} is not a supported version.")
        sys.exit()
    if cmpd_set not in cmpd_sets:
        print(f"{cmpd_set} is not a supported version.")
        sys.exit()        
    if org not in orgs:
        print(f"{org} is not a supported organism.")
        sys.exit()
    if fp not in fps:
        print(f"{fp} is not a supported version.")
        sys.exit()
    if vect not in vects:
        print(f"{vect} is not a supported version.")
        sys.exit()
    if dist not in dists:
        print(f"{dist} is not a supported version.")
        sys.exit()
    if i_score not in i_scores:
        print(f"{i_score} is not a supported version.")
        sys.exit()
    if org=='test':
        pre = os.getcwd()
    elif out_path == os.path.dirname(__file__):
        pre = os.path.dirname(__file__) + "/data/v2.2+"
    elif out_path == '.':
        pre = os.getcwd() + "/data/v2.2+"
    else:
        pre = out_path + "/data/v2.2+"
    # Dirs
    print(f'Creating subdirs in {pre}...')
    os.makedirs(pre, exist_ok=True)
    os.makedirs(f'{pre}/mappings', exist_ok=True)
    os.makedirs(f'{pre}/cmpds', exist_ok=True)
    os.makedirs(f'{pre}/prots', exist_ok=True)
    os.makedirs(f'{pre}/matrices', exist_ok=True)
    print() 

    # Mappings
    print("Downloading library and mapping files...")
    print(f"Files will be downloaded to: {pre + '/mappings/'}")
    if cmpd_set == 'all':
        url = f'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/mappings/drugbank-{v}.tsv'
        dl_file(url, f'{pre}/mappings/drugbank-{v}.tsv')
    elif cmpd_set == 'approved':
        url = f'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/mappings/drugbank-{v}-approved.tsv'
        dl_file(url, f'{pre}/mappings/drugbank-{v}-approved.tsv')
    url = f'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/mappings/drugbank2ctd-{v}.tsv'
    dl_file(url, f'{pre}/mappings/drugbank2ctd-{v}.tsv')
    print() 

    # Compounds
    print(f"Downloading {v} compound library data...")
    print(f"Files will be downloaded to: {pre + '/cmpds/'}")
    if not os.path.exists(f"{pre}/cmpds/fps-{v}/{fp}-{vect}_vect.pickle"):
        print(f"fps-{v}/{fp}-{vect}_vect.pickle")
        url = f'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/cmpds/fps-{v}/{fp}-{vect}_vect.pickle'
        dl_file(url, f'{pre}/cmpds/fps-{v}/{fp}-{vect}_vect.pickle')
    if not os.path.exists(f"{pre}/ligs/fps/{fp}-{vect}_vect.pickle"):
        print(f"ligs/{fp}-{vect}_vect.pickle")
        url = f'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/ligs/fps/{fp}-{vect}_vect.pickle'
        dl_file(url, f'{pre}/ligs/fps/{fp}-{vect}_vect.pickle')
    print() 

    # Proteins
    print(f"Downloading {org} protein library data...")
    print(f"Files will be downloaded to: {pre + '/prots/'}")
    print(f"{org}-coach.tsv")
    url = f'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/prots/{org}-coach.tsv'
    dl_file(url, f'{pre}/prots/{org}-coach.tsv')
    print() 

    # Matrices
    print(f"Downloading matrix for {v} compound library, {cmpd_set} sublibrary, {org} protein library, {fp} fingerprint method, {vect} vector type, {dist} distance, and {i_score} interaction score...")
    print(f"Files will be downloaded to: {pre + '/matrices/'}")
    matrix = f'{fp}-{org}-{v}-{cmpd_set}-{vect}_vect-{dist}-{i_score}.tsv'
    print(f"{matrix}")
    url = f'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/matrices/{matrix}'
    dl_file(url, f'{pre}/matrices/{matrix}')
    print() 
    print('All data was downloaded.\n')


def clear_cache():
    """!
    Clear files in "data/" directory.
    @returns Returns None
    """
    pre = os.path.dirname(__file__) + "/data/"
    shutil.rmtree(pre) 
    #os.system("rm -r {}".format(pre))
    print("{} directory has been removed.".format(pre))


def get_tutorial():
    """!
    Download data for tutorial.
    @returns Returns None
    """
    print('Downloading data for tutorial...')
    pre = os.path.dirname(__file__) + "/data/v2.2+"
    if not os.path.exists('tutorial'):
        os.mkdir('tutorial')
    # Example matrix (rd_ecfp4 w/ 64 prots x 2,449 drugs)
    if not os.path.exists('./tutorial/tutorial_matrix-all.tsv'):
        url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/tutorial/tutorial_matrix-all.tsv'
        dl_file(url, './tutorial/tutorial_matrix-all.tsv')
    if not os.path.exists('./tutorial/tutorial_matrix-approved.tsv'):
        url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/tutorial/tutorial_matrix-approved.tsv'
        dl_file(url, './tutorial/tutorial_matrix-approved.tsv')
    # Mappings
    if not os.path.exists('./tutorial/cmpds-v2.2.tsv'):
        url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/mappings/cmpds-v2.2.tsv'
        dl_file(url, './tutorial/cmpds-v2.2.tsv')
    #if not os.path.exists('./tutorial/cmpds-v2.2-approved.tsv'):
    #    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/mappings/cmpds-v2.2-approved.tsv'
    #    dl_file(url, './tutorial/cmpds-v2.2-approved.tsv')
    if not os.path.exists('./tutorial/cmpds2inds-v2.2.tsv'):
        url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/mappings/cmpds2inds-v2.2.tsv'
        dl_file(url, './tutorial/cmpds2inds-v2.2.tsv')
    # Protein scores
    if not os.path.exists('{}/prots/tutorial-coach.tsv'.format(pre)):
        url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/prots/tutorial-coach.tsv'
        dl_file(url, '{}/prots/tutorial-coach.tsv'.format(pre))
    # New compound set
    if not os.path.exists('./tutorial/tki_set-test.smi'):
        url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/tutorial/tki_set-test.smi'
        dl_file(url, './tutorial/tki_set-test.smi')
    # New compound
    if not os.path.exists('./tutorial/lm235.mol'):
        url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/tutorial/lmk235.mol'
        dl_file(url, './tutorial/lmk235.mol')
    # Protein subset
    if not os.path.exists('./tutorial/tutorial-bac-prots.txt'):
        url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/tutorial/tutorial-bac-prots.txt'
        dl_file(url, './tutorial/tutorial-bac-prots.txt')
    print('All data for tutorial downloaded.\n')


def get_test():
    """!
    Download data for test script.
    @returns Returns None
    """
    print('Downloading data for test...')
    pre = "cando/data/v2.2+/test"
    #pre = os.path.dirname(__file__) + "/data/v2.2+/test"
    os.makedirs(pre,exist_ok=True)
    #url = 'http://protinfo.compbio.buffalo.edu/cando/data/test/test-cmpd_scores.tsv'
    #dl_file(url, '{}/test-cmpd_scores.tsv'.format(pre))
    #url = 'http://protinfo.compbio.buffalo.edu/cando/data/test/test-prot_scores.tsv'
    #dl_file(url, '{}/test-prot_scores.tsv'.format(pre))
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/test/test-cmpds.tsv'
    dl_file(url, '{}/test-cmpds.tsv'.format(pre))
    with open('{}/test-cmpds.tsv'.format(pre), 'r', encoding="utf8") as f:
        l = []
        f.readline()
        for i in f:
            i = i.split('\t')[0]
            i = "{}.mol".format(i)
            l.append(i)
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/test/test-cmpds_mol'
    out = '{}/test-cmpds_mol'.format(pre)
    dl_dir(url, out, l)
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/test/test-inds.tsv'
    dl_file(url, '{}/test-inds.tsv'.format(pre))
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/test/test-cmpds_mol/8100.mol'
    dl_file(url, '{}/test-cmpds_mol/8100.mol'.format(pre))
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/test/test_set.smi'
    dl_file(url, '{}/tki_set-test.smi'.format(pre))
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/test/test-uniprot_set.tsv'
    dl_file(url, '{}/test-uniprot_set.tsv'.format(pre))
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/test/vina64x.fpt'
    dl_file(url, '{}/vina64x.fpt'.format(pre))
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/test/toy64x.fpt'
    dl_file(url, '{}/toy64x.fpt'.format(pre))
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/test/test-pathway-prot.tsv'
    dl_file(url, '{}/test-pathway-prot.tsv'.format(pre))
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/test/test-pathway-mesh.tsv'
    dl_file(url, '{}/test-pathway-mesh.tsv'.format(pre))
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/test/test-new_cmpds.tsv'
    dl_file(url, '{}/test-new_cmpds.tsv'.format(pre))
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/test/test-uniprot_set.tsv'
    print('All test data downloaded.\n')


def dl_dir(url, out, l, pbar=True):
    """!
    Function to recursively download a directory.

    Prints the name of the directory and a progress bar.

    @param url str: URL of the dir to be downloaded
    @param out str: Path to where the dir will be downloaded
    @param l list: List of files in dir to be downloaded
    @returns Returns None
    """
    if not os.path.exists(out):
        os.makedirs(out)
    else:
        for n in l:
            if not os.path.exists("{}/{}".format(out, n)):
                break
        return
    bar = tqdm(l) if pbar else l
    for n in bar:
        url2 = "{}/{}".format(url, n)
        r = requests.get(url2)
        out_file = "{}/{}".format(out, n)
        with open(out_file, 'wb') as f:
            f.write(r.content)


def dl_file(url, out_file, pbar=True):
    """!
    Function to download a file.

    Prints the name of the file and a progress bar.

    @param url str: URL of the file to be downloaded
    @param out_file str: File path to where the file will be downloaded
    @returns Returns None
    """
    if os.path.exists(out_file):
        print("{} exists.".format(out_file))
        return
    elif not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
    time.sleep(1)
    headers = {'Accept-Encoding': 'gzip'}
    r = requests.get(url, stream=True, headers=headers)
    with open(out_file, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        if total_length >= 512000000:
            chunk_size = 512000000
            num_bars = int(total_length / chunk_size)
        else:
            chunk_size = total_length
            num_bars = int(total_length / chunk_size)
        bar = tqdm(r.iter_content(chunk_size=chunk_size), total=num_bars) if pbar else\
               r.iter_content(chunk_size=chunk_size)
        for chunk in bar:
            f.write(chunk)
            f.flush()
            os.fsync(f.fileno())


def load_version(v='v2.3', protlib='nrpdb', i_score='CxP', approved_only=False, compute_distance=False,
                 dist_metric='cosine', protein_set='', ncpus=1):
    """!
    Directly load a pre-compiled version of CANDO.

    @param v str: version to use (supports v2.2 - v2.5)
    @param protlib str: protein library to use ('nrpdb' or 'homo_sapien')
    @param i_score str: the scoring protocol to use ('P', 'C', 'dC', 'CxP', dCxP')
    @param approved_only bool: use only approved drugs to create the matrix
    @param compute_distance bool: compute distance between compounds for specified matrix
    @param dist_metric str: the distance metric to use if compute_distance=True ('cosine', 'rmsd', etc)
    @param protein_set str: path to a file containing a subset of proteins of interest
    @param ncpus int: number of cores to run on
    @return Returns CANDO object
    """

    # download data for version
    get_data(v=v, org=protlib)

    # separate matrix file download (for now)
    app = 'approved' if approved_only else 'all'
    mat_name = 'rd_ecfp4-{}-{}-{}-int_vect-dice-{}.tsv'.format(protlib, v, app, i_score)
    url = 'http://protinfo.compbio.buffalo.edu/cando/data/v2.2+/matrices/{}'.format(mat_name)
    dl_file(url, './data/v2.2+/matrices/{}'.format(mat_name))

    # create CANDO object
    if approved_only:
        cmpd_map_path = 'data/v2.2+/mappings/drugbank-{}-approved.tsv'.format(v)
        matrix_path = 'data/v2.2+/matrices/rd_ecfp4-{}-{}-approved-int_vect-dice-{}.tsv'.format(protlib, v, i_score)
    else:
        cmpd_map_path = 'data/v2.2+/mappings/drugbank-{}.tsv'.format(v)
        matrix_path = 'data/v2.2+/matrices/rd_ecfp4-{}-{}-all-int_vect-dice-{}.tsv'.format(protlib, v, i_score)

    ind_map_path = 'data/v2.2+/mappings/drugbank2ctd-{}.tsv'.format(v)

    cando = CANDO(cmpd_map_path, ind_map_path, matrix=matrix_path, compound_set=app,
                  compute_distance=compute_distance, dist_metric=dist_metric, protein_set=protein_set, ncpus=ncpus)

    return cando

def ind_accuracies(effect_id, effect_cmpds, cmpd_lib, d_name, metrics, approved, n, cando_db, dist_metric, exclude_indic, tierank):
    db_name = f'{d_name}/{effect_id}.db'
    if os.path.exists(db_name):
        db = create_engine(f'sqlite:///{db_name}')
        df_ia_results = pd.read_sql("SELECT cmpd_id FROM ia_results", db)
        df_pa_results = pd.read_sql("SELECT `cmpd_id-1` FROM pa_results", db)
        if len(df_ia_results) == len(effect_cmpds) and len(df_pa_results) == (math.factorial(len(effect_cmpds))/(math.factorial(len(effect_cmpds)-2))):
            return
        else:
            try:
                # Connecting to sqlite
                conn = sqlite3.connect(db_name)
                # Creating a cursor object using the cursor() method
                cursor = conn.cursor()
                # Droping dists table if already exists
                cursor.execute("DROP TABLE ia_results")
                cursor.execute("DROP TABLE pa_results")
                # Commit your changes in the database
                conn.commit()
                # Closing the connection
                conn.close()
            except:
                # print("dists table does not exist.")
                pass
    conn = f'sqlite://{cando_db}'
    effect_cmpds = [str(c) for c in effect_cmpds]
    ss = []
    pa_ss = []
    df_dists = pl.read_database(query=f"SELECT * FROM {dist_metric} WHERE id IN {tuple(effect_cmpds)}", connection=conn)
    dist_df = pl.read_database(query=f"SELECT id FROM {dist_metric}", connection=conn).sort('id')
    if approved:
        dist_df = dist_df.filter(pl.col('id').is_in(cmpd_lib))
    score_df = dist_df.clone()
    rank_df = dist_df.clone()
    nrank_df = dist_df.clone()
    
    for c in effect_cmpds:
        c_sorted = json.loads(df_dists.filter(pl.col('id') == c).select(['dists']).item().replace("'", '"'))
        c_sorted = pl.DataFrame(c_sorted)
        if approved:
            cmpd_lib_temp = [x for x in cmpd_lib if x!=c]
            c_sorted = c_sorted.select(cmpd_lib_temp)
        df_temp = c_sorted.transpose().with_columns(pl.Series(name='id', values=c_sorted.columns)).sort('column_0')
        df_temp = df_temp.with_columns(rank=df_temp['column_0'].rank(method='min'))
        df_temp = df_temp.with_columns(df_temp['rank'].apply(lambda x: 1 if x <= n else 0).alias(c))

        score_df = score_df.join(df_temp.select('id',c), on='id', how='left')
        rank_df = rank_df.join(df_temp.select('id','rank'), on='id', how='left')
        rank_df = rank_df.rename({'rank': c})
        dist_df = dist_df.join(df_temp.select('id','column_0'), on='id', how='left')
        dist_df = dist_df.rename({'column_0': c})
        nrank_df = nrank_df.join(df_temp.select('id', 'rank').apply(lambda t: (t[0], t[1] if (t[1] <= n) else 0)).rename({'column_0':'id','column_1':c}),\
                                 on='id', how='left')
        # Pairwise accuracy
        for c2 in effect_cmpds:
            if c == c2:
                continue
            s = [c, c2, effect_id]
            rank = rank_df.filter(pl.col('id') == c2)[0, c]
            dist = dist_df.filter(pl.col('id') == c2)[0, c]
            for x in metrics:
                if rank <= x[1]:
                    s.append('1')
                else:
                    s.append('0')
            s.append(str(int(rank)))
            s.append(str(float(dist)))
            pa_ss.append(s)
    del df_dists, c_sorted
    for c_loo in effect_cmpds:
        dist_df_loo = dist_df.drop(c_loo)
        score_df_loo = score_df.drop(c_loo)
        rank_df_loo = rank_df.drop(c_loo)
        nrank_df_loo = nrank_df.drop(c_loo)

        score_df_loo = score_df_loo.with_columns(score=pl.sum_horizontal(score_df_loo.columns[1:]))
        dist_df_loo = dist_df_loo.with_columns(summed_dist=pl.sum_horizontal(dist_df_loo.columns[1:]))
        rank_df_loo = rank_df_loo.with_columns(summed_rank=pl.sum_horizontal(rank_df_loo.columns[1:]))
        nrank_df_loo = nrank_df_loo.with_columns(summed_nrank=pl.sum_horizontal(nrank_df_loo.columns[1:]))

        c_df_loo = dist_df_loo.select('id','summed_dist').join(score_df_loo.select('id','score'), on='id', how='left')
        c_df_loo = c_df_loo.join(rank_df_loo.select('id','summed_rank'), on='id', how='left')
        c_df_loo = c_df_loo.with_columns(c_df_loo['score'].apply(lambda x: len(effect_cmpds)-1 - x).alias('neg_score'))
        c_df_loo = c_df_loo.with_columns(c_df_loo['summed_rank'].apply(lambda x: x / (len(effect_cmpds)-1)).alias('avg_rank'))
        c_df_loo = c_df_loo.with_columns(c_df_loo['summed_dist'].apply(lambda x: x / (len(effect_cmpds)-1)).alias('avg_dist'))

        df_temp = nrank_df_loo.select('id','summed_nrank').join(score_df_loo.select('id', 'score'), on='id', how='left')
        df_temp = df_temp.select('id', 'summed_nrank', 'score').apply(lambda t: (t[0], t[1]/t[2] if t[2] > 0 else float(len(cmpd_lib)))).rename({'column_0':'id', 'column_1':'avg_nrank'})
        c_df_loo = c_df_loo.join(df_temp.select('id', 'avg_nrank'), on='id', how='left')

        c_df_loo = c_df_loo.sort('score','avg_nrank','avg_rank','avg_dist', descending=[True,False,False,False])
        # This is competitive ranking
        # Add other ranking methods using polars.Series.rank
        if exclude_indic:
            other_indic = effect_cmpds[:]
            other_indic.remove(c_loo)
            c_df_loo = c_df_loo.filter(~pl.col('id').is_in(other_indic))

        c_df_loo = c_df_loo.with_columns(rank=pl.struct('neg_score','avg_nrank','avg_rank', 'summed_dist').rank(method=tierank))

        rank = c_df_loo.filter(pl.col('id') == c_loo)[0, 'rank']
        score = c_df_loo.filter(pl.col('id') == c_loo)[0, 'score']
        avg_dist = c_df_loo.filter(pl.col('id') == c_loo)[0, 'avg_dist']
        avg_rank = c_df_loo.filter(pl.col('id') == c_loo)[0, 'avg_rank']
        conf = 0.5*(score/(len(effect_cmpds)-1)) + 0.3*(1-avg_dist) + 0.2*(1/avg_rank)

        s = [c_loo, effect_id]
        for x in metrics:
            if rank <= x[1]:
                s.append('1')
            else:
                s.append('0')
        s.append(str(int(rank)))
        s.append(str(int(score)))
        s.append(str(float(avg_rank)))
        s.append(str(float(avg_dist)))
        s.append(str(float(conf)))
        ss.append(s)
    db_benchmark = create_engine(f'sqlite:///{db_name}')
    # Indication accuracies
    benchmark_cols = ['cmpd_id', 'effect_id',
                      'top10', 'top25', 'top50', 'top100', f'top{len(cmpd_lib)}', 'top1%', 'top5%',
                      'top10%', 'top50%', 'top100%', 'rank', 'score', 'avg_rank', 'avg_dist', 'conf']
    df_temp = pd.DataFrame(ss, columns=benchmark_cols)
    df_temp.to_sql('ia_results', db_benchmark, if_exists='append', index=False)
    # Pairwise accuracies
    benchmark_cols = ['cmpd_id-1', 'cmpd_id-2', 'effect_id',
                      'top10', 'top25', 'top50', 'top100', f'top{len(cmpd_lib)}', 'top1%', 'top5%',
                      'top10%', 'top50%', 'top100%', 'rank', 'dist']
    df_temp = pd.DataFrame(pa_ss, columns=benchmark_cols)
    df_temp.to_sql('pa_results', db_benchmark, if_exists='append', index=False)
    return

def ind_accuracies_cmpd_pair(effect_id, effect_cps, cp_lib, d_name, metrics, approved, n, cando_db):
    db_name = f'{d_name}/{effect_id}.db'
    if os.path.exists(db_name):
        db = create_engine(f'sqlite:///{db_name}')
        df_ia_results = pd.read_sql("SELECT cmpd_pair_id FROM ia_results", db)
        df_pa_results = pd.read_sql("SELECT cmpd_pair_id_1 FROM pa_results", db)
        if len(df_ia_results) == len(effect_cps) and len(df_pa_results) == (math.factorial(len(effect_cps))/(math.factorial(len(effect_cps)-2))):
            return
        else:
            try:
                # Connecting to sqlite
                conn = sqlite3.connect(db_name)
                # Creating a cursor object using the cursor() method
                cursor = conn.cursor()
                # Droping dists table if already exists
                cursor.execute("DROP TABLE ia_results")
                cursor.execute("DROP TABLE pa_results")
                # Commit your changes in the database
                conn.commit()
                # Closing the connection
                conn.close()
            except:
                # print("dists table does not exist.")
                pass
    conn = f'sqlite://{cando_db}'
    effect_cps = [str(cp) for cp in effect_cps]
    ss = []
    pa_ss = []
    df_dists = pl.read_database(query=f"SELECT * FROM dists WHERE id IN {tuple(effect_cps)}", connection=conn)
    dist_df = pl.read_database(query=f"SELECT id FROM dists", connection=conn).sort('id')
    if approved:
        dist_df = dist_df.filter(pl.col('id').is_in(cp_lib))
    score_df = dist_df.clone()
    rank_df = dist_df.clone()
    for c in effect_cps:
        c_sorted = json.loads(
            df_dists.filter(pl.col('id') == c).select(['dists']).item().replace("'", '"'))
        c_sorted = pl.DataFrame(c_sorted)
        if approved:
            cp_lib_temp = [x for x in cp_lib if x!=c]
            c_sorted = c_sorted.select(cp_lib_temp)
        df_temp = c_sorted.transpose().with_columns(pl.Series(name='id', values=c_sorted.columns)).sort('column_0')
        df_temp = df_temp.with_columns(rank=df_temp['column_0'].rank(method='min'))
        df_temp = df_temp.with_columns(df_temp['rank'].apply(lambda x: 1 if x <= n else 0).alias(c))
        score_df = score_df.join(df_temp.select('id',c), on='id', how='left')
        rank_df = rank_df.join(df_temp.select('id','rank'), on='id', how='left')
        rank_df = rank_df.rename({'rank': c})
        dist_df = dist_df.join(df_temp.select('id','column_0'), on='id', how='left')
        dist_df = dist_df.rename({'column_0': c})
        # Pairwise accuracy
        for c2 in effect_cps:
            if c == c2:
                continue
            s = [c, c2, effect_id]
            rank = rank_df.filter(pl.col('id') == c2)[0, c]
            dist = dist_df.filter(pl.col('id') == c2)[0, c]
            for x in metrics:
                if rank <= x[1]:
                    s.append('1')
                else:
                    s.append('0')
            s.append(str(int(rank)))
            s.append(str(float(dist)))
            pa_ss.append(s)
    del df_dists, c_sorted

    for c_loo in effect_cps:
        dist_df_loo = dist_df.drop(c_loo)
        score_df_loo = score_df.drop(c_loo)
        rank_df_loo = rank_df.drop(c_loo)

        score_df_loo = score_df_loo.with_columns(score=pl.sum_horizontal(score_df_loo.columns[1:]))
        dist_df_loo = dist_df_loo.with_columns(summed_dist=pl.sum_horizontal(dist_df_loo.columns[1:]))
        rank_df_loo = rank_df_loo.with_columns(summed_rank=pl.sum_horizontal(rank_df_loo.columns[1:]))

        c_df_loo = dist_df_loo.select('id','summed_dist').join(score_df_loo.select('id','score'), on='id', how='left')
        c_df_loo = c_df_loo.join(rank_df_loo.select('id','summed_rank'), on='id', how='left')
        c_df_loo = c_df_loo.with_columns(c_df_loo['score'].apply(lambda x: len(effect_cps)-1 - x).alias('neg_score'))
        c_df_loo = c_df_loo.with_columns(c_df_loo['summed_rank'].apply(lambda x: x / (len(effect_cps)-1)).alias('avg_rank'))
        c_df_loo = c_df_loo.with_columns(c_df_loo['summed_dist'].apply(lambda x: x / (len(effect_cps)-1)).alias('avg_dist'))

        c_df_loo = c_df_loo.sort('score','avg_rank', 'avg_dist', descending=[True,False,False])
        # This is competitive ranking
        # Add other ranking methods using polars.Series.rank
        c_df_loo = c_df_loo.with_columns(rank=pl.struct('neg_score', 'avg_rank', 'summed_dist').rank(method='min'))

        rank = c_df_loo.filter(pl.col('id') == c_loo)[0, 'rank']
        score = c_df_loo.filter(pl.col('id') == c_loo)[0, 'score']
        avg_dist = c_df_loo.filter(pl.col('id') == c_loo)[0, 'avg_dist']
        avg_rank = c_df_loo.filter(pl.col('id') == c_loo)[0, 'avg_rank']
        conf = 0.5*(score/(len(effect_cps)-1)) + 0.3*(1-avg_dist) + 0.2*(1/avg_rank)

        s = [c_loo, effect_id]
        for x in metrics:
            if rank <= x[1]:
                s.append('1')
            else:
                s.append('0')
        s.append(str(int(rank)))
        s.append(str(int(score)))
        s.append(str(float(avg_rank)))
        s.append(str(float(avg_dist)))
        s.append(str(float(conf)))
        ss.append(s)
    db_benchmark = create_engine(f'sqlite:///{db_name}')
    # Indication accuracies
    benchmark_cols = ['cmpd_pair_id', 'effect_id',
                      'top10', 'top25', 'top50', 'top100', f'top{len(cp_lib)}', 'top1%', 'top5%',
                      'top10%', 'top50%', 'top100%', 'rank', 'score', 'avg_rank', 'avg_dist', 'conf']
    df_temp = pd.DataFrame(ss, columns=benchmark_cols)
    df_temp.to_sql('ia_results', db_benchmark, if_exists='append', index=False)
    # Pairwise accuracies
    benchmark_cols = ['cmpd_pair_id_1', 'cmpd_pair_id_2', 'effect_id',
                      'top10', 'top25', 'top50', 'top100', f'top{len(cp_lib)}', 'top1%', 'top5%',
                      'top10%', 'top50%', 'top100%', 'rank', 'dist']
    df_temp = pd.DataFrame(pa_ss, columns=benchmark_cols)
    df_temp.to_sql('pa_results', db_benchmark, if_exists='append', index=False)
    return



#---Additional Metric Functions-------------------------------------------------
def recall(ranks, thresholds, out_of):
    '''
    ranks - list of int, ranks at which true positives recalled
    thresholds - list of int, ranks above which metrics should be calculated
    out_of - int, highest possible rank

    Returns recall; equivalent to nAIA when one left out cmpd per trial
    '''
    scores = [0]*len(thresholds)
    for rank in ranks:
        for i in range(len(thresholds)):
            if rank <= thresholds[i][1]:
                scores[i] += 1
    scores = [100*x/len(ranks) for x in scores]
    return scores

def precision1(ranks, thresholds, out_of):
    '''
    ranks - list of int, ranks at which true positives recalled
    thresholds - list of int, ranks above which metrics should be calculated
    out_of - int, highest possible rank

    Returns averaged precision across all ranks
        max = 100/threshold for leave-one-out
    '''
    scores = [0]*len(thresholds)
    for rank in ranks:
        for i in range(len(thresholds)):
            if rank <= thresholds[i][1] and thresholds[i][1] > 0:
                scores[i] += 1/thresholds[i][1]
    scores = [100*x/len(ranks) for x in scores]
    return scores

def adjust_ranks(ranks):
    '''
    ranks - list of int, ranks at which true positives are recalled

    Returns a sorted list of ranks where later ranks are increased as if
        indicated compounds were all ranked together
        e.g. [1, 3, 3, 3, 5] becomes [1, 4, 5, 6, 9]
    Mimics pooling multiple leave-one-out runs into a single trial
    Does not alter original ranks list
    '''
    ranks = sorted(ranks)
    for i in range(len(ranks)):
        ranks[i] = ranks[i] + i
    return ranks

def precision2(ranks, thresholds, out_of):
    '''
    ranks - list of int, ranks at which true positives recalled
    thresholds - list of int, ranks above which metrics should be calculated
    out_of - int, highest possible rank

    Returns precision if all attempts ranks are pooled
        max = min(100, (100*(# of cmpds) / threshold))
    '''
    scores = [0]*len(thresholds)
    for rank in adjust_ranks(ranks):
        for i in range(len(thresholds)):
            if rank <= thresholds[i][1]:
                scores[i] += 1
    scores = [100*scores[i]/thresholds[i][1] if thresholds[i][1] > 0 else 0 for i in range(len(scores))]
    return scores

def auroc(ranks, thresholds, out_of):
    '''
    ranks - list of int, ranks at which true positives recalled
    thresholds - list of int, ranks above which metrics should be calculated
    out_of - int, highest possible rank

    Returns AUROC through given thresholds
        Area calculated using trapezoid method
    '''
    scores = [0]*len(thresholds)
    ranks = sorted(ranks)
    tprs = [0]
    fprs = [0]
    i = 0
    while i < len(ranks):
        if i+1 < len(ranks) and ranks[i + 1] == ranks[i]:
            i += 1
        tprs.append(i + 1)
        fprs.append((ranks[i]*len(ranks)) - (i + 1))
        i += 1

    tprs = [x/len(ranks) for x in tprs] + [1]
    fprs = [x/(len(ranks)*out_of) for x in fprs] + [1]
    thresholds = [x[1]/out_of for x in thresholds]

    for j in range(len(thresholds)):
        thresh = thresholds[j]
        total_area = 0
        for i in range(1, len(fprs)):
            if fprs[i] > thresh:
                run = thresh - fprs[i-1]
                rise = ((tprs[i] - tprs[i-1])/(fprs[i] - fprs[i -1]))*run
                total_area += (tprs[i-1]*run) + ((rise*run)/2)
                break
            else:
                run = fprs[i] - fprs[i-1]
                rise = tprs[i] - tprs[i-1]
                total_area += (tprs[i-1]*run) + ((rise*run)/2)
        scores[j] = total_area
    return scores

def control_auroc(ranks, thresholds, out_of):
    '''
    ranks - list of int, ranks at which true positives recalled
    thresholds - list of int, ranks above which metrics should be calculated
    out_of - int, highest possible rank

    Returns AUROC for control curve (slope of 1) at all thresholds
    '''
    thresholds = [min(1,x[1]/out_of) for x in thresholds]
    scores = [0 for x in thresholds]
    for i in range(len(scores)):
        scores[i] = (thresholds[i]**2)/2
    return scores

def ndcg(ranks, thresholds, out_of):
    '''
    ranks - list of int, ranks at which true positives recalled
    thresholds - list of int, ranks above which metrics should be calculated
    out_of - int, highest possible rank

    Returns NDCG, equivalent to that calculated in canbenchmark
        Only works for leave-one-out method currently
    '''
    dcgs = [0]*len(thresholds)
    for rank in ranks:
        for i in range(len(thresholds)):
            if rank <= thresholds[i][1]:
                dcgs[i] += 1/math.log(rank + 1, 2)
    return [x/len(ranks) for x in dcgs]
