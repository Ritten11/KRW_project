# KRW_project
### _do you catch my drift?_
Code reporitory of the Knowledge Representation on the Web project by Ritten Roothaert and Stella verkijk

### Running instructions:
Install the correct dependencies using the command: ```pip install -r requirements.txt```

NOTE: The embeddings within this repository are pickled to save on storage space. Make sure the correct pickle version 
is installed, otherwise you might experience issues de-pickling the objects within this repository. Your version should 
be compatible with pickle format version 4.0.


## Cleaning the code and creating different ontology versions
All code concerning the prepartion of the knowledge graphs is located in the `data_processing.ipynb` notebook. Here, 
the knoledge graph is cleaned and several different versions of the graph are saved in the `./data` directory.

To run the notebook, run: `jupyter-notebook data_processing.ipynb`

If the notebook is completed, modified graphs should be saved as 

`./data/modified_graphs/[KG_NAME]/[CHANGES_AT_RANK]/[CHANGE_RATIO].ttl`,

where `[KG_NAME]` is the name of the original graph, `[CHANGES_AT_RANK]` is the ranks at which the knowledge graph has 
been modified, and `[CHANGE_RATIO]` is the ratio of all triples between the ranks of `[CHANGES_AT_RANK]` that have been 
modified.

All changes made to the graph are saved in the `./data/change_log/` directory.

## Initial exploratory data analysis
Before having making embeddings of the graphs, it is important to understand what we are working with. The 
`data_analysis.ipynb` notebook contains a few functions providing some additional insights into the `tax_NCIT` 
and `tac_and_subset_NCIT` graphs created in the `data_processing.ipynb` notebook

To run the notebook, run: `jupyter-notebook data_processing.ipynb`

## Creation of the embeddings
This step takes quite some time, depending on the number of embeddings made and training epochs. Therefore, it was 
decided not to use a Jupyter notebook, but a regular python script. This way, the script can be started through the 
terminal, making it easier to run the script on a remote server. 

This script comes with an additional set of running instructions. First, make sure the script is executable using the 
line:

`chmox +x create_embedding.py`

Once this is done, run `./create_embedding.py -h` for a list of options. The output should look like this:

```
usage: create_embedding.py [-h] [-n N_ITER] [-e EPOCH_LIST]
                           [-kg KNOWLEDGE_GRAPH] [-m] [-r CHANGE_RATIO]

optional arguments:
  -h, --help            show this help message and exit
  -n N_ITER, --n_iter N_ITER
                        Number of iterations
  -e EPOCH_LIST, --epoch_list EPOCH_LIST
                        Specify at which intervals the embedding should be
                        saved. The embedding is trained until the max
                        specified epoch count. Format: [1,2,3,4,...]
  -kg KNOWLEDGE_GRAPH, --knowledge_graph KNOWLEDGE_GRAPH
                        location to the knowledge graph that is to be embedded
  -m, --modified        The supplies graphs are modified from the original
                        graph
  -r CHANGE_RATIO, --change_ratio CHANGE_RATIO
                        The ratio of changes made to the original graph
```
For example, the line 

`./create_embedding.py -n 5 -kg tax_NCIT -e [10,20,50,100]` 

creates 5 embeddings
(`-n 5`) of the knowledge graph `tax_NCIT` (`-kg tax_NCIT`), and saves the embedding at epochs 10, 20, 50, and 100 
(`-e [10,20,50,100]`). These embeddings will be saved in the `./data/embeddings/[KNOWLEDGE_GRAPH]/` directory with the 
file name `[epoch]_[iter].pkl`, which represents at which epoch the graph was saved and the current embedding iteration.

Making embeddings of a modified graph is similar, but has been implemented in a slightly 'hacky' fashion. To do so, the
`-m` flag should be used, along with the ration of the changes made to the graph using for example `-r 0.1` if at 10% of 
the subclass relations between two ranks have been changed. The script will then iterate over all directories within the 
`./data/modified_graphs/[KNOWLEDGE_GRAPH]/` directory to retrieve all modified graphs, which all should have the 
file name `0.1.pkl`. 

These embeddings will be stored in the directory `./data/embeddings/modifed_graphs/KNOWLEDGE_GRAPH/`.

## Analysing the embeddings

All the analysis of the embeddings is done in `embedding_comparison.ipynb` notebook. 

To run the notebook, run: `jupyter-notebook embedding_comparison.ipynb`

