import csv, math, random
from graphviz import Digraph

RENDER = True

def main():
    with open ('mush_train.data', 'r') as csvfile:
        reader = csv.reader(csvfile)
        train_cases = []
        for row in reader:
            case = (row[1:], row[0])
            train_cases += [case]
    with open ('mush_test.data', 'r') as csvfile:
        reader = csv.reader(csvfile)
        test_cases = []
        for row in reader:
            case = (row[1:], row[0])
            test_cases += [case]
    attrs = []
    attr_vals = []
    for i in range(0, len(train_cases[0][0])):
        attrs += [i]
        attr_vals += [set()]

    # Determine possible attribute values from training data
    for i in range(0, len(train_cases)):
        for j in range(0, len(train_cases[i][0])):
            attr_vals[j] = attr_vals[j] | {train_cases[i][0][j]}

    dt = DTNode(None, attrs, attr_vals, train_cases)

    if (RENDER == True):
        dot = Digraph(comment='Decision Tree')
        dot.node('0', str(dt.attr + 1))
        dt.make_graph(dot, '0')
        dot.render('decision_tree.gv', view=True)

    correct = 0
    for i in range(0, len(train_cases)):
        if (edible(train_cases[i]) == dt.classify(train_cases[i][0])):
            correct += 1
    print('Training accuracy: ' + str(correct) + ' / ' + str(len(train_cases)) + ' = ' + str(correct / len(train_cases)))
    correct = 0
    for i in range(0, len(test_cases)):
        if (edible(test_cases[i]) == dt.classify(test_cases[i][0])):
            correct += 1
    print('Testing accuracy: ' + str(correct) + ' / ' + str(len(test_cases)) + ' = ' + str(correct / len(test_cases)))

def edible(case):
    if (case[1] == 'e'):
        return True
    elif (case[1] == 'p'):
        return False
    else:
        raise Exception("Couldn't determine if case was poisonous.")

def split_cases(cases, attr, attr_vals):
    cases_split = []
    for i in range(0, len(attr_vals[attr])):
        cases_split += [[]]

    count = 0
    for x in attr_vals[attr]:
        for i in range(0, len(cases)):
            if (cases[i][0][attr] == x):
                cases_split[count] += [cases[i]]
        count += 1

    return cases_split

def entropy_c(cases, attr, attr_vals):
    sum = 0
    cases_split = split_cases(cases, attr, attr_vals)

    # When splitting the attribute doesn't actually do anything
    if (len(cases_split[0]) == len(cases)):
        return None

    for i in range(0, len(cases_split)):
        entr_y = entropy(cases_split[i])
        prob_x = len(cases_split[i]) / len(cases)
        sum += prob_x * entr_y

    return sum

def entropy(cases):
    if (len(cases) == 0):
        return 0
    edibles = 0
    for i in range (0, len(cases)):
        if (edible(cases[i])):
            edibles += 1
    pedible = (edibles / len(cases))
    ppoisonous = ((len(cases) - edibles) / len(cases))
    if (pedible == 0 or pedible == 1):
        return 0
    return - pedible * math.log(pedible, 2) - ppoisonous * math.log(ppoisonous, 2)

def cases_pure(cases):
    y = edible(cases[0])
    for i in range(1, len(cases)):
        if (edible(cases[i]) != y):
            return False
    return True

def get_votes(cases):
    votes = 0
    for i in range(0, len(cases)):
        if (edible(cases[i])):
            votes += 1
        else:
            votes -= 1
    return votes

class DTNode:
    """
    children: subtrees based on attribute
    attr: attribute that this tree splits on
    val: value of the previously split on attribute that this tree contains

    if attr is None, then edible describes the results of a majority vote. if the vote is tied, then when queried the
    node will return a random response
    """
    children = []
    attr = None
    val = None
    votes = None

    def any_ties(self):
        if (self.children == []):
            if (self.votes == 0):
                return True
            else:
                return False
        else:
            for c in self.children:
                if (c is None):
                    continue
                else:
                    if (c.any_ties() == True):
                        return True
            return False

    def height(self):
        if (self.children == []):
            return 0
        else:
            max = -1
            for c in self.children:
                if (c == None):
                    continue
                h = c.height()
                if (h > max):
                    max = h
            return max + 1

    def make_graph(self, dot, node_id):
        if (self.children == []):
            return
        else:
            count = 0
            for c in self.children:
                if (c is None):
                    continue
                if (c.attr is None):
                    if (c.votes > 0):
                        dot.node(node_id + str(count), 'E')
                    elif (c.votes < 0):
                        dot.node(node_id + str(count), 'P')
                    else:
                        dot.node(node_id + str(count), '-')
                    dot.edge(node_id, node_id + str(count), c.val)
                    count += 1
                else:
                    dot.node(node_id + str(count), str(c.attr + 1))
                    dot.edge(node_id, node_id + str(count), c.val)
                    c.make_graph(dot, node_id + str(count))
                    count += 1

    def get_votes_rec(self):
        if (self.children == []):
            return self.votes
        else:
            sum = 0
            for c in self.children:
                if (c is None):
                    continue
                else:
                    sum += c.get_votes_rec()
            return sum

    def classify(self, fv):
        if (self.children == []):
            return self.vote_edible()
        val = fv[self.attr]
        for c in self.children:
            if (c is None):
                continue
            if (c.val == val):
                return c.classify(fv)
        return self.vote_edible()

    def vote_edible(self):
        if (self.get_votes_rec() > 0):
            return True
        elif (self.get_votes_rec() < 0):
            return False
        else:
            return bool(random.getrandbits(1))

    """
    parent: parent node
    attrs: remaining features to split on. list of indexes into the cases which are valid to split on. from 0
    attr_vals: possible values for particular features. determines the number of children
    cases: cases to classify
    """
    def __init__(self, val, attrs, attr_vals, cases):
        self.val = val
        self.children = []
        self.attr = None
        self.votes = None

        if (cases_pure(cases) or attrs == []):
            self.votes = get_votes(cases)
        else:
            entr_y = entropy(cases)
            max_ig = -float("Inf")
            index = -1
            # for each possible split attribute find information gain
            for i in range(0, len(attrs)):
                entr_c = entropy_c(cases, attrs[i], attr_vals)
                if (entr_c is None):
                    continue
                ig = entr_y - entr_c
                if (ig > max_ig):
                    max_ig = ig
                    index = i

            # If splitting on no attribute produced any information gain
            if (max_ig == -float("Inf")):
                self.votes = get_votes(cases)
            else:
                self.attr = attrs[index]
                cases_split = split_cases(cases, self.attr, attr_vals)

                for i in range(0, len(cases_split)):
                    if (cases_split[i] == []):
                        self.children += [None]
                    else:
                        attrs_copy = attrs.copy()
                        attrs_copy.remove(self.attr)
                        self.children += [DTNode(cases_split[i][0][0][self.attr], attrs_copy, attr_vals,
                                                cases_split[i])]

if __name__ == "__main__":
    main()