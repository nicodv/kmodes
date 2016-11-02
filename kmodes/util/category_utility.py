import numpy as np
import math


def category_utility(x):
    labels = x[:,-1]
    x_shape = x.shape
    n_rows=  x_shape[0]
    n_cols = x_shape[1]

    unique_labels = np.unique(labels)
    m = len(unique_labels)
    print m
    probabilites =  []
    conditional_probability_terms = []
    print "Conditional probability "
    for label in unique_labels:
        x_size = len(x)
        x_sub = x[np.where(x[:,-1] == label)]
        x_sub_size = len(x_sub)
        probabilites.append(x_sub_size/float(x_size))
        print "Label : ", label
        conditional_probability_term = 0
        for column in range(0, n_cols-1):
            column_values = x_sub[:,column]
            unique_elements_in_column = np.unique(column_values)
            for unique_element in unique_elements_in_column:
                unique_values = x_sub[np.where(x_sub[:,column] == unique_element)]
                term = math.pow((len(unique_values)/float(x_sub_size)), 2)
                print "unique_element : ", unique_element
                print "Term : ", term
                conditional_probability_term += term
        conditional_probability_terms.append(conditional_probability_term)
    print "Conditional probability terms...."
    print conditional_probability_terms

    unconditional_probability_sum = 0
    unconditional_probability_terms = []
    print "Unconditional probability term..."
    for column in range(0, n_cols-1):
        column_values = x[:,column]
        unique_elements_incolumn = np.unique(column_values)
        term = 0
        for unique_element in unique_elements_incolumn:
            unique_values = x[np.where(x[:,column] == unique_element)]
            term = math.pow((len(unique_values)/float(n_rows)), 2)
            unconditional_probability_terms.append(term)
            unconditional_probability_sum += term
    print unconditional_probability_terms
    print unconditional_probability_sum
    sum = 0
    print "Probabailities : ", probabilites
    for i in range(0,m):
        sum += (probabilites[i] * (conditional_probability_terms[i] - unconditional_probability_sum))
    cu = (1/float(m)) * (sum)
    print "Categoru utility : ", cu