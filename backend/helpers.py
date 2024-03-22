#put merge_postings_and, merge_postings_or, and jaccardsim functions here
def merge_postings_and (ing1,ing2,inv_idx):
    '''
    ing1: str, the first ingredient
    ing2: str, the second ingredient
    inv_idx: inverted index, key as ingredient, value as list of recipes

    Returns the list of recipes that contains ing1 and ing2
    '''
    rep1 = set(inv_idx[ing1])
    rep2 = set(inv_idx[ing2])
    return list(rep1.intersection(rep2))

def merge_postings_or (ing1,ing2,inv_idx):
    '''
    ing1: str, the first ingredient
    ing2: str, the second ingredient
    inv_idx: inverted index, key as ingredient, value as list of recipes

    Returns the list of recipes that contains either ing1 or ing2
    '''
    rep1 = set(inv_idx[ing1])
    rep2 = set(inv_idx[ing2])
    return list(rep1.union(rep2))

def jaccard_sim ():
    pass