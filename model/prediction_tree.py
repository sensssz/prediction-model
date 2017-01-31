# --*-- coding: utf-8 --*--
""" A tree-based prediction model.
"""

import itertools
import json
import operator
import re
import sys

import numpy as np

from spinner import Spinner

class Console(object):
    """ Provides printing utilities.
    """
    console_width = 100
    spinner = Spinner.create_spinner(18, 100)

    @classmethod
    def print_str(cls, str_to_print, newline=False, override=True, spinner=False):
        """ Print without new line.
        """
        if override:
            cls.eat_line(cls.console_width)
        if spinner:
            str_to_print = cls.spinner.next() + ' ' + str_to_print
        if newline:
            str_to_print += '\n'
        sys.stdout.write(str_to_print)
        sys.stdout.flush()

    @classmethod
    def eat_line(cls, num_to_eat):
        """ Print \b to eat words.
        """
        sys.stdout.write('\b' * num_to_eat)
        sys.stdout.write(' ' * cls.console_width)
        sys.stdout.write('\b' * num_to_eat)

class Query(object):
    """ Represents a query, including its template, its arguments and
    its result set.
    """
    def __init__(self, query_id, result_set, arguments, is_select, ordered):
        self._query_id = query_id
        self._result_set = result_set
        self._arguments = arguments
        self._is_select = is_select
        self._ordered = ordered

    @property
    def query_id(self):
        """ Get the query id.
        """
        return self._query_id

    @property
    def result_set(self):
        """ Get the result set of the query.
        Valid only when the query is a SELECT query.
        """
        return self._result_set

    @property
    def arguments(self):
        """ Get the arguments of the current query.
        """
        return self._arguments

    @property
    def is_select(self):
        """ Return if the current query is a SELECT query.
        """
        return self._is_select

    @property
    def ordered(self):
        """ Return if the result of the current query is ordered.
        """
        return self.is_select and self._ordered

    def get_sql(self, query_set):
        """ Ge the SQL statement of the query.
        """
        query = query_set.query_template(self._query_id)
        for argument in self._arguments:
            if isinstance(argument, str):
                query = query.replace('?s', argument)
            elif isinstance(argument, int) or isinstance(argument, float):
                query = query.replace('?d', str(argument))
            elif isinstance(next(iter(argument)), str):
                query = query.replace('?ls', str(argument)[1:-1])
            else:
                query = query.replace('?ld', str(argument)[1:-1])
        return query

    def __str__(self):
        return str(self.query_id)

    def __repr__(self):
        return str(self.query_id)


class QuerySet(object):
    """ Stores the set of queries and their query IDs.
    """
    def __init__(self):
        self._query_ids = {}
        self._id_queries = {}

    def query_id(self, query_template):
        """ Return the query ID given a query template.
        """
        if query_template in self._query_ids:
            return self._query_ids[query_template]
        else:
            query_id = len(self._query_ids)
            self._query_ids[query_template] = query_id
            self._id_queries[query_id] = query_template
            return query_id

    def query_template(self, query_id):
        """ Given a query ID, return the query template.
        """
        return self._id_queries[query_id]


class QueryParser(object):
    """ Parse SQL queries and create Query objects.
    Maintain a set of all query templates.
    """
    def __init__(self, query_set):
        self._query_set = query_set
        self._argument_pattern = re.compile(r"('[^']*'|\d+|\d?\.\d+| IN \([^)]+\))")
        self._num_str_pattern = re.compile(r"('[^']*'|\d+|\d?\.\d+)")
        self._string_pattern = re.compile(r"('[^']*')")
        self._number_pattern = re.compile(r"(\d+|\d?\.\d+)")
        self._num_list_pattern = re.compile(r" IN \(([^)']+)\)")
        self._str_list_pattern = re.compile(r" IN \(([^)0-9]+)\)")

    def convert_arguments(self, arguments):
        """ Convert the arguments of a query to correct type.
        """
        for i, arg in enumerate(arguments):
            if ' IN ' in arg:
                list_args = self._num_str_pattern.findall(arg)
                self.convert_arguments(list_args)
                arguments[i] = set(list_args)
            elif "'" in arg:
                arguments[i] = arguments[i][1:-1]
            elif '.' in arg:
                arguments[i] = float(arguments[i])
            else:
                arguments[i] = int(arguments[i])

    def json_loads_byteified(self, json_text):
        """ Byteified json.loads.
        """
        return self._byteify(
            json.loads(json_text, object_hook=self._byteify),
            ignore_dicts=True
        )

    def _byteify(self, data, ignore_dicts=False):
        """ json_hook to byteify strings.
        """
        # if this is a unicode string, return its string representation
        if isinstance(data, unicode):
            return data.encode('utf-8')
        # if this is a list of values, return list of byteified values
        if isinstance(data, list):
            return [self._byteify(item, ignore_dicts=True) for item in data]
        # if this is a dictionary, return dictionary of byteified keys and values
        # but only if we haven't already byteified it
        if isinstance(data, dict) and not ignore_dicts:
            return {
                self._byteify(key, ignore_dicts=True): self._byteify(value, ignore_dicts=True)
                for key, value in data.iteritems()
            }
        # if it's anything else, return it in its original form
        return data

    def parse_query(self, query_text):
        """ Parse a SQL query in JSON format and create a Query object.
        """
        query_json = None
        try:
            query_json = self.json_loads_byteified(query_text)
        except ValueError as _:
            print query_text
        sql = query_json['sql']
        template = self._string_pattern.sub("'?s'", sql)
        template = self._number_pattern.sub('?d', template)
        template = self._num_list_pattern.sub(' IN (?ld)', template)
        template = self._str_list_pattern.sub(' IN (?ls)', template)
        query_id = self._query_set.query_id(template)
        result_set = query_json['results']
        arguments = self._argument_pattern.findall(sql)
        self.convert_arguments(arguments)
        is_select = sql.lstrip().lower().startswith('select')
        ordered = 'order by' in sql
        return Query(query_id, result_set, arguments, is_select, ordered)


class QueryResultOperand(object):
    """ Represents a value in the result set of a previous query.
    """
    def __init__(self, query_index, row_index, column_index, vtype):
        self._query_index = query_index
        self._row_index = row_index
        self._column_index = column_index
        self._type = vtype

    def get_value(self, trx):
        """ Find the corresponding value from the previous queries.
        """
        query_result = trx[self._query_index].result_set
        value = query_result[self._row_index][self._column_index]
        return value

    def is_string(self):
        """ Check if the operand has a string value.
        """
        return self._type == str

    def __str__(self):
        return 'RES(' + str(self._query_index) + ',' + \
               str(self._row_index) + ',' + str(self._column_index) + ')'

    def __repr__(self):
        return 'RES(' + str(self._query_index) + ',' + \
               str(self._row_index) + ',' + str(self._column_index) + ')'


class AggregationOperand(object):
    """ Represents the result of an aggregation on the result set of a query.
    """
    def __init__(self, query_index, aggregation, column_index):
        self._query_index = query_index
        self._aggregation = aggregation
        self._column_index = column_index

    def get_value(self, trx):
        """ Get the value of the aggregation operation.
        """
        query_result = trx[self._query_index].result_set
        j = self._column_index
        column = [row[j] for row in query_result if row[j] is not None]
        if len(column) > 0:
            return self._aggregation(column)
        else:
            return 0

    def is_string(self):
        """ Check if the operand has a string value.
        """
        return False

    def __str__(self):
        return 'AGG(' + str(self._query_index) + ',' + \
               str(self._aggregation.__name__) + ',' + str(self._column_index) + ')'

    def __repr__(self):
        return 'AGG(' + str(self._query_index) + ',' + \
               str(self._aggregation.__name__) + ',' + str(self._column_index) + ')'


class QueryArgumentOperand(object):
    """ Represents one of the arguments of a previous query.
    """
    def __init__(self, query_index, argument_index, vtype):
        self._query_index = query_index
        self._argument_index = argument_index
        self._type = vtype

    def get_value(self, trx):
        """ Find the corresponding value from the previous queries.
        """
        value = trx[self._query_index].arguments[self._argument_index]
        return value

    def is_string(self):
        """ Check if the operand has a string value.
        """
        return self._type == str

    def __str__(self):
        return 'ARG(' + str(self._query_index) + ',' + str(self._argument_index) + ')'

    def __repr__(self):
        return 'ARG(' + str(self._query_index) + ',' + str(self._argument_index) + ')'


class ColumnListOperand(object):
    """ Represents a whole column as a list operand.
    """
    def __init__(self, query_index, column_index):
        self._query_index = query_index
        self._column_index = column_index

    def get_value(self, trx):
        """ Return a set of values of the corresponding column.
        """
        query_result = trx[self._query_index].result_set
        column = set([row[self._column_index] for row in query_result])
        return column

    def is_string(self, _):
        """ Check if the operand has a string value.
        """
        return False


class BinaryOperation(object):
    """ Represents an operation on some values as the argument of a query.
    """
    def __init__(self, operation, left_operand, right_operand, trx):
        self._operation = operation
        self._left_operand = left_operand
        self._right_operand = right_operand
        self._value = self._operation(self._left_operand.get_value(trx),
                                      self._right_operand.get_value(trx))

    @property
    def value(self):
        """ Get the value of the argument.
        """
        return self._value

    def matches_value(self, value):
        """ Return if the operation matches the given value.
        """
        return self._value == value

    def __str__(self):
        return str(self._left_operand) + ' ' + str(self._operation) + ' ' + str(self._right_operand)

    def __repr__(self):
        return str(self._left_operand) + ' ' + str(self._operation) + ' ' + str(self._right_operand)


class UnaryOperation(object):
    """ Represents a previously appearing value as the argument
    of a query.
    """
    def __init__(self, operand, trx):
        self._operand = operand
        self._value = operand.get_value(trx)

    @property
    def value(self):
        """ Get the value of the argument.
        """
        return self._value

    def matches_value(self, value):
        """ Return if the operation matches the given value.
        """
        return self._value == value

    def __str__(self):
        return str(self._value)

    def __repr__(self):
        return str(self._value)


class RandomValue(object):
    """ Represents a random value.
    """
    def matches_value(self, _):
        """ Return true since all value could possibly be a random value.
        """
        return True

    def __str__(self):
        return "RAMD"

    def __repr__(self):
        return "RAMD"


class Prediction(object):
    """ A node in the prediction tree. Represents prediction of how a query's
    arguments are computed.
    """
    def __init__(self, query_id, arguments):
        self._query_id = query_id
        self._args_predictions = arguments
        self._hit_count = 0

    @property
    def query_id(self):
        """ Return the query ID.
        """
        return self._query_id

    def is_random(self):
        """ Return true if any of the argument prediciton is random.
        """
        for arg_prediction in self._args_predictions:
            if isinstance(arg_prediction, RandomValue):
                return True
        return False

    def matches_query(self, query):
        """ Check if the prediction matches the query.
        """
        if self._query_id != query.query_id:
            return False
        for arg_predic, arg in zip(self._args_predictions, query.arguments):
            if not arg_predic.matches_value(arg):
                return False
        return True

    def hit(self):
        """ Increase the hit count of the current node.
        """
        self._hit_count += 1

    @property
    def hit_count(self):
        """ Return the hit count of the current node.
        """
        return self._hit_count

    def __str__(self):
        return ' (' + str(self._query_id) + ':' + str(self._args_predictions) + ') '

    def __repr__(self):
        return ' (' + str(self._query_id) + ':' + str(self._args_predictions) + ') '


class RandomPrediction(Prediction):
    """ Represents a random value that will always match the target query.
    """
    def __init__(self, query_id):
        super(RandomPrediction, self).__init__(query_id, None)

    def matches_query(self, *_):
        """ Check if the prediction matches the query.
        """
        return True

    def __str__(self):
        return ' (RANDOM: ' + str(self.query_id) + ') '

    def __repr__(self):
        return ' (RANDOM: ' + str(self.query_id)+ ') '


class Node(object):
    """ A tree node.
    """
    def __init__(self, data):
        self._data = data
        self._children = []

    @property
    def data(self):
        """ Return the data stored in the current node.
        """
        return self._data

    @property
    def children(self):
        """ Return the children of the current node.
        """
        return self._children

    @children.setter
    def children(self, children):
        """ Set the value of childre.
        """
        self._children = children

    def add_child(self, child):
        """ Add a child to the current node.
        """
        self._children.append(Node(child))

    def add_children(self, children):
        """ Add children to the current node.
        """
        for child in children:
            self.add_child(child)

    def print_tree(self, depth=0):
        """ Print the whole subtree.
        """
        sys.stdout.write('  ' * depth)
        print str(self.data)
        for child in self.children:
            child.print_tree(depth + 1)

    def __str__(self):
        if len(self.children) > 0:
            return str(self.data) + '=>' + str(self.children) + '\n'
        else:
            return str(self.data) + '\n'

    def __repr__(self):
        if len(self.children) > 0:
            return str(self.data) + '=>' + str(self.children) + '\n'
        else:
            return str(self.data) + '\n'


class BinaryOp(object):
    """ Represent a binary operation.
    """
    def __init__(self, operation, symmetric):
        self._operation = operation
        self._symmetric = symmetric

    def __call__(self, left_operand, right_operand):
        return self._operation(left_operand, right_operand)

    @property
    def symmetric(self):
        """ Returns whether the bianry operation is symmetric.
        """
        return self._symmetric

    def __str__(self):
        return self._operation.__name__


class PredictionTreeBuilder(object):
    """ Builds a prediction tree using the training set.
    """
    def __init__(self, file_path):
        self._query_set, self._transactions = self.parse_queries(file_path)
        self._trees = {}

    def parse_queries(self, file_path):
        """ Parse all queries from the training set.
        """
        query_file = open(file_path, 'r')
        query_set = QuerySet()
        query_parser = QueryParser(query_set)
        queries = []
        num_read = 0
        Console.print_str('Parsing queries ' + str(num_read), spinner=True)
        for line in query_file:
            if len(line) <= 1:
                continue
            queries.append(query_parser.parse_query(line))
            num_read += 1
            Console.print_str('Parsing queries ' + str(num_read), spinner=True)
        query_file.close()
        Console.print_str('Done parsing.', newline=True)
        transactions = self.split_trx(queries, query_set, True)
        return query_set, transactions

    def query_is(self, query, query_set, sql):
        """ Check if the SQL statement of the query is |sql|.
        """
        return query.get_sql(query_set) == sql

    def split_trx(self, queries, query_set, include_stray):
        """ Split the queries into transactions.
        """
        transactions = []
        current_trx = []
        query_index = 0
        starts_with_begin = self.query_is(queries[0], query_set, 'BEGIN')
        if starts_with_begin:
            query_index += 1
        while query_index < len(queries):
            query = queries[query_index]
            if (starts_with_begin and self.query_is(query, query_set, 'COMMIT') or
                    not starts_with_begin and self.query_is(query, query_set, 'BEGIN')):
                if len(current_trx) > 0 and (starts_with_begin or include_stray):
                    transactions.append(current_trx)
                current_trx = []
                if self.query_is(query, query_set, 'COMMIT'):
                    query_index += 1
                    if query_index >= len(queries):
                        break
                    starts_with_begin = self.query_is(queries[query_index], query_set, 'BEGIN')
                    if starts_with_begin:
                        query_index += 1
                else:
                    starts_with_begin = True
                    query_index += 1
            else:
                current_trx.append(query)
                query_index += 1
        for trx in transactions:
            if len(trx) > 50:
                print trx
        return transactions

    def cluster_trx(self, transactions):
        """ Put transactions with exactly the same sequence of query
            templates into the same cluster.
        """
        templates = []
        for trx in transactions:
            templates.append([query.query_id for query in trx])
        clusters = []
        while len(templates) > 0:
            i = 0
            current_cluster = []
            current_template = templates[0]
            while i < len(templates):
                trx_template = templates[i]
                if trx_template == current_template:
                    current_cluster.append(i)
                    del templates[i]
            clusters.append(current_cluster)
        return clusters

    @property
    def transactions(self):
        """ Return the list of transactions.
        """
        return self._transactions

    def __str__(self):
        return str(self._trees)

    def __repr__(self):
        return str(self._trees)

    def enumerate_result_operand(self, query_index, query):
        """ Enumerate all QueryResultOperand from the given query.
        """
        string_operands = []
        number_operands = []
        length = len(query.result_set)
        if (not query.ordered and length > 1) or length == 0:
            return [], []
        res_len = len(query.result_set)
        len_limit = 40
        if res_len <= len_limit:
            indices = range(res_len)
        else:
            indices = range(len_limit / 2)
            indices += range(-len_limit / 2, 0)
        for i in indices:
            row = query.result_set[i]
            for j, value in enumerate(row):
                operand = QueryResultOperand(query_index, i, j, type(value))
                if isinstance(value, str):
                    string_operands.append(operand)
                elif isinstance(value, int) or isinstance(value, float):
                    number_operands.append(operand)
        return string_operands, number_operands

    def enumerate_aggregation_operand(self, query_index, query, ops):
        """ Enumerate all AggregationOperand from the given query.
        """
        operands = []
        if len(query.result_set) == 0:
            return []
        for aggregation in ops:
            first_row = query.result_set[0]
            for i, value in enumerate(first_row):
                if isinstance(value, int) or isinstance(value, float):
                    operand = AggregationOperand(query_index, aggregation, i)
                    operands.append(operand)
        return operands

    def enumerate_argument_operand(self, query_index, query):
        """ Enumerate all QueryArgumentOperand from the given query.
        """
        string_operands = []
        number_operands = []
        for i, value in enumerate(query.arguments):
            operand = QueryArgumentOperand(query_index, i, type(value))
            if isinstance(value, str):
                string_operands.append(operand)
            elif not isinstance(value, set):
                number_operands.append(operand)
        return string_operands, number_operands

    def enumerate_list_operand(self, query_index, query):
        """ Enumerate all QueryColumnOperand.
        """
        string_lists = []
        number_lists = []
        if len(query.result_set) == 0:
            return [], []
        first_row = query.result_set[0]
        for i, value in enumerate(first_row):
            operand = ColumnListOperand(query_index, i)
            if isinstance(value, str):
                string_lists.append(operand)
            else:
                number_lists.append(operand)
        return string_lists, number_lists

    def enumerate_all_operands(self, query_index, query, aggregation_ops):
        """ Enumerate all possible operands.
        """
        string_operands = []
        number_operands = []
        str_ops, num_ops = self.enumerate_result_operand(query_index, query)
        string_operands += str_ops
        number_operands += num_ops
        str_ops, num_ops = self.enumerate_argument_operand(query_index, query)
        string_operands += str_ops
        number_operands += num_ops
        num_ops = self.enumerate_aggregation_operand(query_index, query, aggregation_ops)
        number_operands += num_ops
        string_lists, number_lists = self.enumerate_list_operand(query_index, query)
        return string_operands, number_operands, string_lists, number_lists

    def type_compatible(self, value, target_type):
        """ Check if the type of the value is compatilbe with the target type.
        """
        if target_type == float:
            return isinstance(value, float) or isinstance(value, int)
        else:
            return isinstance(value, target_type)

    def enumerate_unary_for_type(self, trx, operands, value_type):
        """ Wrap all previously appearing value as UnaryOperations
        for the given type.
        """
        unary_ops = []
        for operand in operands:
            if self.type_compatible(operand.get_value(trx), value_type):
                unary_ops.append(UnaryOperation(operand, trx))
                # if len(unary_ops) >= 100:
                #     return unary_ops
        return unary_ops

    def get_binary_ops(self):
        """ Return all binary operations.
        """
        return [BinaryOp(operator.add, True), BinaryOp(operator.sub, False),
                BinaryOp(operator.mul, True), BinaryOp(operator.div, False),
                BinaryOp(operator.mod, False)]

    def get_aggregation_ops(self):
        """ Return all aggregation operations.
        """
        return np.mean, np.sum, len, min, max

    def enumerate_binary(self, trx, operands, ops):
        """ Enumerate all combinations of binary operations and operands.
        """
        binary_ops = []
        combis = list(itertools.combinations(operands, 2))
        permus = list(itertools.permutations(operands, 2))
        for binary_op in ops:
            if binary_op.symmetric:
                op_set = combis
            else:
                op_set = permus
            for op_pair in op_set:
                if not ((str(binary_op) == 'div' or str(binary_op) == 'mod')
                        and op_pair[1].get_value(trx) == 0):
                    operand = BinaryOperation(binary_op, op_pair[0],
                                              op_pair[1], trx)
                    binary_ops.append(operand)
                    # if len(binary_ops) >= 100:
                    #     return binary_ops
        return binary_ops

    def enumerate_all_ops_for_value(self, trx, str_operands, num_operands,
                                    str_lists, num_lists, value):
        """ Enumerate all possible operations on all possible values
        to compute the target value.
        """
        if isinstance(value, set):
            if isinstance(next(iter(value)), str):
                ops = self.enumerate_unary_for_type(trx, str_lists, type(value))
                ops = [op for op in ops if op.value == value]
            else:
                ops = self.enumerate_unary_for_type(trx, num_lists, type(value))
                ops = [op for op in ops if op.value == value]
        elif isinstance(value, str):
            ops = self.enumerate_unary_for_type(trx, str_operands, type(value))
            ops = [op for op in ops if op.value == value]
        else:
            unaries = self.enumerate_unary_for_type(trx, num_operands, type(value))
            binaries = self.enumerate_binary(trx, num_operands, self.get_binary_ops())
            ops = unaries + binaries
            ops = [op for op in ops if op.value == value]
        ops.insert(0, RandomValue())
        return ops

    def combinations_recur(self, op_lists, current_op_list_index, current_combi, all_combis):
        """ Recursively enumerate all possible combinations of operations of different
        operation lists.
        """
        if len(all_combis) >= 4000:
            return
        if current_op_list_index == len(op_lists):
            all_combis.append(list(current_combi))
            return
        for operation in op_lists[current_op_list_index]:
            current_combi[current_op_list_index] = operation
            self.combinations_recur(op_lists, current_op_list_index + 1, current_combi, all_combis)

    def combinations(self, op_lists):
        """ Enumerate all possible combinations of operations of different
        operation lists.
        """
        all_combis = []
        num_lists = len(op_lists)
        current_combi = [None] * num_lists
        self.combinations_recur(op_lists, 0, current_combi, all_combis)
        return all_combis

    def enumerate_prediction(self, trx, str_operands, num_operands,
                             str_lists, num_lists, query):
        """ Enumerate all possible predictions for the given query.
        """
        op_for_args = []
        i = 0
        for argument in query.arguments:
            print 'Enumerating operands for argument ' +\
                  str(i) + ', which is a ' + str(type(argument))
            op_for_arg = self.enumerate_all_ops_for_value(
                trx, str_operands, num_operands, str_lists, num_lists, argument)
            op_for_args.append(op_for_arg)
            i += 1
        print 'Operand enumeration done. Number of operands for each arg:'
        # if len(op_for_args) > 1:
        #     i = 0
        #     for i, op_for_arg in enumerate(op_for_args):
        #         op_for_args[i] = op_for_arg[:200]
        for op_for_arg in op_for_args:
            sys.stdout.write(str(len(op_for_arg)) + ',')
        print '\nGenerating all combinations'
        op_combis = self.combinations(op_for_args)
        print str(len(op_combis)) + ' combinations generated.'
        nodes = []
        for combi in op_combis:
            prediction = Prediction(query.query_id, combi)
            prediction.hit()
            nodes.append(Node(prediction))
        print str(len(nodes)) + ' generated.'
        return nodes

    def get_prediction_tree(self, query):
        """ Return the corresponding prediction tree given the first query.
        """
        if query.query_id not in self._trees:
            root = Node(RandomPrediction(query.query_id))
            self._trees[query.query_id] = root
        else:
            root = self._trees[query.query_id]
        return root


    def print_trees(self):
        """ Print all trees.
        """
        for _, tree in self._trees.items():
            tree.print_tree()


    def build_model_for_trx(self, trx):
        """ Build a model for a given transaction type.
        """
        current_query = trx[0]
        root = self.get_prediction_tree(current_query)
        hit_nodes = [root]
        str_ops = []
        num_ops = []
        str_lists = []
        num_lists = []
        query_index = 0
        while query_index < len(trx):
            # Console.print_str('Processing query ' + str(query_index), spinner=True)
            print 'Enumerating operands for query ' + str(query_index)
            str_operands, num_operands, string_lists, number_lists = self.enumerate_all_operands(
                query_index, current_query,
                self.get_aggregation_ops())
            str_ops += str_operands
            num_ops += num_operands
            str_lists += string_lists
            num_lists += number_lists
            query_index += 1
            if query_index >= len(trx):
                break
            current_query = trx[query_index]
            next_hit_nodes = set()
            predictions = None
            i = 0
            num_nodes = len(hit_nodes)
            for node in hit_nodes:
                Console.print_str('Processing node ' + str(i) + '/' + str(num_nodes), spinner=True)
                i += 1
                query_not_found = True
                children = node.children
                for child in children:
                    if child.data.query_id == current_query.query_id:
                        query_not_found = False
                        if child.data.matches_query(current_query):
                            child.data.hit()
                            next_hit_nodes.add(child)
                if query_not_found:
                    if predictions is None:
                        print '\nEnumerating predioction for query ' + str(query_index) + ' with '\
                            + str(len(current_query.arguments)) + ' arguments and ('\
                            + str(len(str_ops)) + ',' + str(len(num_ops)) + ','\
                            + str(len(str_lists)) + ',' + str(len(num_lists)) + ') ops.'
                        predictions = self.enumerate_prediction(trx, str_ops, num_ops,
                                                                str_lists, num_lists, current_query)
                        next_hit_nodes.union(set(predictions))
                    node.children = predictions
            hit_nodes = next_hit_nodes
            print ''
            print str(len(hit_nodes)) + ' hit nodes'

    def pick_highest_hit_node(self, nodes):
        """ Pick the node with the highest hit count.
        """
        highest_hit_node = nodes[0]
        for node in nodes:
            if node.data.hit_count > highest_hit_node.data.hit_count:
                highest_hit_node = node
        return highest_hit_node

    def hit_matches(self, nodes, query):
        """ Increase the hit count of all hit predictions.
        """
        highes_hit_node = nodes[0]
        for node in nodes:
            if node.data.matches_query(query):
                node.data.hit()
                if node.data.hit_count > highes_hit_node.data.hit_count:
                    highes_hit_node = node
        return highes_hit_node

    def predict_next_query(self, trx):
        """ Predict the next query using the prediction tree.
        """
        hits = 0
        first_query = trx[0]
        if first_query.query_id not in self._trees:
            return 0
        root = self._trees[first_query.query_id]
        next_nodes = root.children
        for index in range(1, len(trx) - 1):
            query = trx[index]
            if len(next_nodes) == 0:
                break
            highest_hit = self.pick_highest_hit_node(next_nodes)
            if highest_hit.data.matches_query(query):
                highest_hit.data.hit()
                # if not highest_hit.data.is_random():
                #     hits += 1
                hits += 1
                next_nodes = highest_hit.children
            else:
                highest_hit = self.hit_matches(next_nodes, query)
                next_nodes = highest_hit.children
        return hits

def main(args):
    """ Main function.
    """
    builder = PredictionTreeBuilder(args[1])
    transactions = builder.transactions
    thirty_percent = int(len(transactions) * 0.3)
    for trx in transactions[:thirty_percent]:
        if len(trx) <= 50:
            builder.build_model_for_trx(trx)
    Console.print_str('Done processing.', newline=True)
    num_queries = 0
    num_hits = 0
    i = 0
    for trx in transactions[thirty_percent:]:
        Console.print_str("Processing transaction " + str(i), spinner=True)
        num_queries += len(trx)
        num_hits += builder.predict_next_query(trx)
        i += 1
    lens = [len(x) for x in transactions]
    avg_len = np.mean(lens)
    Console.print_str(str(num_hits) + ',' + str(num_queries) + ','\
                      + str(len(transactions)) + ',' + str(avg_len), newline=True)
    # builder.print_trees()
    # print builder.queries

if __name__ == '__main__':
    main(sys.argv)
