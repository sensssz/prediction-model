#!/usr/local/bin/python
# -*- coding: utf-8 -*-
""" SQL query encoder.

Extract the dictionary from the training set and encode all queries using the dictionary.

Example:
    $ python query_encoder.py [training_set_file]
    or
    $ ./query_encoder.py [training_set_file]

Generates file ${training_set_file}_encoded containing all encoded queries.
Generates file ${training_set_file}_clean containing all raw queries.

"""

import re
import sys


def filename_append(filename, suffix):
    """
    Add a suffix to a file name (separated by '_'). If it has an extension,
    add the suffix before the extension.

    Args:
        filename: Original file name.
        suffix: Suffix to be added to the file name.

    Returns:
        string: New file name.
    """
    ext_index = filename.rfind('.')
    if ext_index != -1:
        ext = filename[ext_index:]
        name = filename[:ext_index]
        return name + '_' + suffix + ext
    else:
        return filename + '_' + suffix


def clean_data_and_extract_queries(filename):
    """
    Generates file ${training_set_file}_clean containing all raw queries.

    Args:
        filename: path of the training set file.

    Returns:
        list: All raw queries.

    """
    file_in = open(filename, 'r')
    clean_file = open(filename_append(filename, 'clean'), 'w')
    queries = []
    for line in file_in:
        i = line.find('SQL: ')
        if i != -1:
            query = line[i + len('SQL: '):-1]
            queries.append(query)
            clean_file.write(query + '\n')
    clean_file.close()
    return queries


def is_word(word):
    """
    Check if a string is a non-value word.

    Args:
        word: The word to be checked

    Returns:
        bool: True if the given string is a non-value word.
    """
    for char in word:
        if not (char.isalpha() or char == '.'
                or char == '*' or char == '_'
                or char == '+' or char == '-'
                or char == '/' or char == '%'
                or char == ':'):
            return False

    return True


def slice_query(query):
    """
    Slice a query into works, excluding all values.

    Args:
        query: A raw query.

    Returns:
        list: List of words.
    """
    back_quote_rm = query.replace('`', '')
    # Handle keystores.key specially.
    back_quote_rm = re.sub(r"keystores.key = '(.+?)'", r'keystores.key = \1', back_quote_rm)
    back_quote_rm = re.sub(r"keystores.key = user.+:([^\s]+)", r'keystores.key = \1', back_quote_rm)
    str_val_rm = re.sub(r"'.+?'", r'', back_quote_rm)
    symbol_rm = str_val_rm.replace('(', ' ').replace(')', ' ').replace(',', ' ')
    words_and_values = symbol_rm.split()
    words = []
    for word in words_and_values:
        if is_word(word):
            words.append(word)
    return words


def encode_query(dictionary, query_words):
    """
    Encode a query using a dictionary and the words in the query.
    Example:
        dictionary: ['SELECT', 'UPDATE', 'IN']
        query_words: ['SELECT', 'IN', 'SELECT']
        encoded query: [2, 0, 1]

    Args:
        dictionary: The dictionary.
        query_words: Words in a query.

    Returns:
        list: A vector representing the query.
    """
    encoded = [0] * len(dictionary)
    for word in query_words:
        encoded[dictionary.index(word)] += 1
    return encoded


def cluster_queries(cluster_filename):
    """
    Cluster the queries.

    Args:
        cluster_filename: File containing the cluster label of each query.

    Returns:
        dict: A dict of cluster label to query indices.
    """
    cluster_file = open(cluster_filename, 'r')
    cluster_indices = [int(line) for line in cluster_file]
    clusters = {}
    i = 0
    for cluster in cluster_indices:
        if cluster not in clusters:
            clusters[cluster] = [i]
        else:
            clusters[cluster].append(i)
        i += 1
    return clusters


def encode_trx(queries, cluster_filename):
    """
    Encode transactions using query clusters as dictionary.

    Args:
        queries: All raw queries.
        cluster_filename: File containing the cluster label of each query.

    Returns:
        list: List of vectors representing the transactions.
    """
    cluster_file = open(cluster_filename, 'r')
    cluster_indices = [int(line) for line in cluster_file]
    num_clusters = max(cluster_indices)
    encoded_trxes = []
    encoded_trx = [0] * num_clusters
    new_trx = True
    for i, query in enumerate(queries):
        if new_trx and query != 'BEGIN':
            # Expect a BEGIN but see a normal query.
            encoded_trx[cluster_indices[i] - 1] = 1
            encoded_trxes.append(encoded_trx)
            encoded_trx = [0] * num_clusters
        elif query == 'BEGIN':
            new_trx = False
        elif query == 'COMMIT':
            new_trx = True
            encoded_trxes.append(encoded_trx)
            encoded_trx = [0] * num_clusters
        else:
            encoded_trx[cluster_indices[i] - 1] += 1
    return encoded_trxes


def extract_dict(queries, filename):
    """
    Extract a dictionary from the raw queries.
    Output to a file named ${filename}_dict.

    Args:
        queries: All raw queries.
        filename: Name of the training set file.

    Returns:
        dict: A dict of cluster label to query indices.
    """
    queries_words = []
    words = set()
    for query in queries:
        query_words = slice_query(query)
        words = words.union(set(query_words))
        queries_words.append(query_words)
    dictionary = list(words)

    dictionary_file = open(filename_append(filename, 'dict'), 'w')
    for word in dictionary:
        dictionary_file.write(word + '\n')
    dictionary_file.close()

    return dictionary, queries_words


def cluster_trx(queries, cluster_filename):
    """
    Encode transactions using query clusters as dictionary.

    Args:
        queries: All raw queries.
        cluster_filename: File containing the cluster label of each transaction.

    Returns:
        dict: Cluster of transactions.
    """
    transactions = []
    transaction = []
    new_trx = True
    for i, query in enumerate(queries):
        if new_trx and query != 'BEGIN':
            transactions.append([query])
        elif query == 'BEGIN':
            new_trx = False
        elif query == 'COMMIT':
            new_trx = True
            transactions.append(transaction)
            transaction = []
        else:
            transaction.append(query)

    cluster_file = open(cluster_filename, 'r')
    cluster_indices = [int(line) for line in cluster_file]
    trx_clusters = {}
    i = 0
    for cluster in cluster_indices:
        if cluster not in trx_clusters:
            trx_clusters[cluster] = [transactions[i]]
        else:
            trx_clusters[cluster].append(transactions[i])
        i += 1
    return trx_clusters


def main():
    """
    Main function.
    """
    if len(sys.argv) < 2:
        print 'Usage: query_encoder.py [training_set_file] '\
              '[query_cluster_file](Optional) [trx_cluster_file](Optional)'
        sys.exit(1)
    filename = sys.argv[1]
    queries = clean_data_and_extract_queries(filename)

    if len(sys.argv) == 2:
        dictionary, queries_words = extract_dict(queries, filename)
        encoded_queries = []
        for query_words in queries_words:
            encoded_queries.append(encode_query(dictionary, query_words))

        encoded_query_file = open(filename_append(filename, 'query_encoded'), 'w')
        for encoded_query in encoded_queries:
            encoded_query_file.write(str(encoded_query)[1:-1] + '\n')
        encoded_query_file.close()
    elif len(sys.argv) == 3:
        query_cluster_filename = sys.argv[2]
        query_cluster_file = open(filename_append(filename, 'query_cluster'), 'w')
        clusters = cluster_queries(query_cluster_filename)
        for cluster_label, query_indices in clusters.iteritems():
            query_cluster_file.write('Cluster ' + str(cluster_label) + '\n')
            for query_index in query_indices:
                query_cluster_file.write(queries[query_index] + '\n')
        query_cluster_file.close()

        encoded_trxes = encode_trx(queries, query_cluster_filename)
        encoded_trx_file = open(filename_append(filename, 'trx_encoded'), 'w')
        for encoded_trx in encoded_trxes:
            encoded_trx_file.write(str(encoded_trx)[1:-1] + '\n')
        encoded_trx_file.close()
    elif len(sys.argv) == 4:
        trx_cluster_filename = sys.argv[3]
        trx_cluster_file = open(filename_append(filename, 'trx_cluster'), 'w')
        trx_clusters = cluster_trx(queries, trx_cluster_filename)
        for cluster_label, trxes in trx_clusters.iteritems():
            trx_cluster_file.write('Cluster ' + str(cluster_label) + '\n')
            for trx in trxes:
                trx_cluster_file.write('BEGIN\n')
                for query in trx:
                    trx_cluster_file.write(query + '\n')
                trx_cluster_file.write('COMMIT\n')
        trx_cluster_file.close()


if __name__ == '__main__':
    main()
