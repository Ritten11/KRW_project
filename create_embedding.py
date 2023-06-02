#!/usr/bin/env python3

import pandas as pd
import rdflib
import os
import copy
import pickle
import sys
import gc  # garbage collector

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

from pprint import pprint
import progressbar
from multiprocessing import Pool

# Note: this is needed for reproducibility. Makes the 'random' processes within this notebook deterministic
# SEED = 42 ## NOTE: Does not appear to work properly due to Word2Vec implementation not allowing for setting the random state


import argparse

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-id", "--run_id", help="Identifier indicating the current run")
parser.add_argument("-e", "--epoch_list",
                    help="Specify at which intervals the embedding should be saved. The embedding " \
                         "is trained until the max specified epoch count. Format: [1,2,3,4,...]")
parser.add_argument("-w", "--number_of_workers", help="specify the number of workers/threats your wish to use")
parser.add_argument("-kg", "--knowledge_graph", help="location to the knowledge graph that is to be embedded")

parser.add_argument("-m", "--modified", help="The supplies graphs are modified from the original graph",
                    action="store_true")
parser.add_argument("-r", "--change_ratio", help="The ratio of changes made to the original graph")

# Read arguments from command line
args = parser.parse_args()


def create_walks(transformer, knowledge_graph, nodes):
    """
    Function for creating walks and determining the corpus from those walks
    :param transformer: The RDF2VecTransformer used for making the walks.
    :param knowledge_graph: Instance of RDF2Vec.Graph to which the random walker should be applied
    :param nodes: Instances from which the walker should start walking. Should be a list of strings.
    :return: Both the returned walks and the corpus of the walks (which is the same thing but unnested)
    """
    walks = transformer.get_walks(knowledge_graph, nodes)
    corpus = [walk for entity_walks in walks for walk in entity_walks]
    transformer.embedder._model.build_vocab(corpus, update=False)
    return walks, corpus


def fit_embedding(transformer, knowledge_graph, nodes, epochs_list, rep, sub_dir):
    """

    :param transformer: The RDF2VecTransformer used for making the embeddings
    :param knowledge_graph: Instance of RDF2Vec.Graph that is to be embedded
    :param nodes: Instances from which an embedding should be made. Should be a list of strings.
    :param epochs_list: List of epochs at which the embedding should be saved
    :param rep: The current repetition of the embedding. Sometimes multiple embeddings of the save graph are made, and
    this is needed for saving the embedding to the right directory
    :param sub_dir: subdirectory to which the embedding should be saved.
    :return:
    """
    # loss_df = pd.DataFrame(columns=['epoch', 'loss'])
    walks = transformer.get_walks(knowledge_graph, nodes)

    print('Starting fitting of word2vec embedding:')

    bar = progressbar.ProgressBar(maxval=max(epochs_list),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for e in range(max(epochs_list)):
        transformer.embedder.fit(walks, False)
        if (e + 1) in epochs_list:
            embeddings, literals = transformer.transform(knowledge_graph, nodes)
            save_embeddings(embeddings, literals, e + 1, rep, sub_dir)
    bar.finish()
    return


def init_transformer(seed):
    # Create our transformer, setting the embedding & walking strategy.
    transformer = RDF2VecTransformer(
        Word2Vec(epochs=1, workers=int(args.number_of_workers)),
        walkers=[RandomWalker(1, 2, with_reverse=True, n_jobs=int(args.number_of_workers), random_state=seed, md5_bytes=None)],
        # walkers=[RandomWalker(4, 10, with_reverse=True, n_jobs=10, random_state=seed, md5_bytes=None)],
        verbose=1
    )
    # transformer.embedder._model.compute_loss = True # Needed to keep track of the loss of the embedding. Used to verify that the embedding is actually learning something
    return transformer


def save_embeddings(embeddings, literals, n_epochs, rep, rank):
    """
    Function for parsing the directory path and saving the embedding
    :param embeddings: The embeddings of the entities. Should be a 2d matrix
    :param n_epochs: The number of epochs used to train the emedding
    :param rep: The current repetition of the embedding
    :param rank: If the embedding is of a modified graph, this provides an indication of where the modifications have been made
    :return: None
    """
    if args.modified:
        path = f'./data/embeddings/modified_graphs/' + args.knowledge_graph + '/' + rank + '/'
    else:
        path = f'./data/embeddings/' + args.knowledge_graph + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    df = pd.DataFrame(embeddings)
    print(df)
    df = pd.concat([pd.DataFrame(literals, columns=['label']), df], axis=1)
    print(df)
    df.to_pickle(path + f'{n_epochs}_{rep}.pkl', protocol=4)
    df.to_csv(path + f'{n_epochs}_{rep}.csv', index=False)


def save_loss_data(loss_df, rep, rank):
    """
    Function for saving the loss data. Could be usefull for validating training progress of the embedding
    :param loss_df: Pandas DataFrame containing the loss at every epoch
    :param rep: The current repetition of the embedding
    :param rank: If the embedding is of a modified graph, this provides an indication of where the modifications have been made
    :return: None
    """
    if args.modified:
        path = f'./data/loss_data/modified_graphs/' + args.knowledge_graph + '/' + rank + '/'
    else:
        path = f'./data/loss_data/' + args.knowledge_graph + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    loss_df.to_pickle(path + f'{rep}.pkl', protocol=4)


def run_experiment(knowledge_graph, nodes, epoch_list, rep, sub_dir):
    """
    Simple function combining the initialisation of the transformer, the creation of the walks, and the fitting of the embedding
    :param knowledge_graph: Instance of RDF2Vec.Graph that is to be embedded
    :param nodes: List of instances for which an embedding should be made.
    :param epoch_list: List of epochs at which the embedding should be saved
    :param rep: The current repetition of the embedding
    :param sub_dir: subdirectory to which the embedding should be saved.
    :return:
    """
    transformer = init_transformer(rep)
    # walks, corpus = create_walks(transformer, knowledge_graph, nodes)
    fit_embedding(transformer, knowledge_graph, nodes, epoch_list, rep, sub_dir)


def get_entities(KG_loc):
    """
    In hindsight a rather inefficient implementation to extract all entities from the graph
    :param KG_loc: Location from which the graph should be loaded.
    :return: A list of al unique entities within the graph.
    """
    print("loading entities from graph")
    tax_and_sub = rdflib.Graph()
    tax_and_sub.parse(KG_loc)
    print("finished loading KG - started entity query")
    nodes_result = list(tax_and_sub.query(
        'SELECT DISTINCT ?s ?p ?o WHERE { ?s ?p ?o. FILTER(isURI(?o) && isURI(?s))}'
    ))
    print("finished entity query - combining objects and subjects and filtering out duplicates")
    objects = {str(n[0]) for n in nodes_result}
    subjects = {str(n[2]) for n in nodes_result}
    entities = objects.union(subjects)
    print("finished constructing entity list")
    return list(entities)


def main(run_id, epoch_list):
    """
    The main function of the script. It determines the locations of the various graphs that are to be loaded and starts
    the embedding procedure with n_reps repetitions
    :param run_id: Unique number identifying the run
    :param epoch_list: List of epochs of when the embedding should be saved
    :return:
    """
    if args.modified:
        sub_dirs = [x[0] for x in os.walk(f'./data/modified_graphs/{args.knowledge_graph}')][1:]
        path_to_KG = [str(sub_dir) + f'/{args.change_ratio}.ttl' for sub_dir in sub_dirs]
        path_to_KG = zip(path_to_KG, [x[1] for x in os.walk(f'./data/modified_graphs/{args.knowledge_graph}')][0])
    else:
        path_to_KG = [(f'./data/{args.knowledge_graph}.ttl', None)]
    for path, rank in path_to_KG:
        print("Loading RDF2Vec KG object")
        knowledge_graph = KG(
            path,
            literals=[['http://www.w3.org/2000/01/rdf-schema#label']]
        )
        nodes = get_entities(path)

        print("Started with creation of embeddings")
        run_experiment(knowledge_graph, nodes, epoch_list, run_id, rank)
        gc.collect()  # call garbage collector to free unused memory


if __name__ == '__main__':
    """
    Initial function for parsing the CLI arguments.
    """
    if args.modified and not args.change_ratio:
        print("please specify the change ratio. Use -h for additional help")
        sys.exit(1)
    epoch_list = args.epoch_list[1:-1].split(',')
    epoch_list = [int(e) for e in epoch_list]
    print(f'saving the embedding at model intervals: {epoch_list}')
    print(f'Current run: {args.run_id}')
    main(int(args.run_id), epoch_list)
