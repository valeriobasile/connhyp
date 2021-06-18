# connhyp
Connotative Hyperplane for computing connotative shift in dia* corpora

The script takes five arguments:

 * first corpus
 * second corpus
 * first label
 * second label
 * a file containing the target words

The corpora are text files, with one sentence per line and words separated by whitespace.
The labels are the identifiers of the two extremes of the connotation axis to be considered, and they have to be reflected in the presence of two text files called *seed_{LABEL}.txt*. These files contain two sets of seed words to train the hyperplane.

For example, with the provided example data:

    ./connhyp.py ccoha_old.txt ccoha_new.txt neg pos targetwords.txt

Will prodice a table with the connotative shifts along the polarity axis (negative/positive) of the words contained in the file *targetwords.txt* computed between two diachronic halves of the  Clean Corpus of Historical American English (CCOHA) corpus (Davies, 2012; Alatrashet al., 2020). Example data adapted from SemEval-2020 Task 1:Unsupervised Lexical Semantic Change Detection.
