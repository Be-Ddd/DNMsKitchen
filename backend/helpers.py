#put merge_postings_and, merge_postings_or, and jaccardsim functions here
def merge_postings_and (lst1,lst2):
    merged_postings=[]
    index1 = 0
    index2 = 0 
    while index1 < len(lst1) and index2< len(lst2):
        if lst1[index1]==lst2[index2]:
            merged_postings.append(lst1[index1])
            index1 += 1
            index2 += 1
        else:
            if lst1[index1]<lst2[index2]:
                index1 += 1
            else:
                index2 += 1
    return merged_postings

def merge_postings_or (lst1,lst2):
    merged_postings=[]
    index1 = 0
    index2 = 0 
    while index1 < len(lst1) and index2< len(lst2):
        if lst1[index1]<lst2[index2]:
            merged_postings.append(lst1[index1])
            index1+=1
        else:
            merged_postings.append(lst2[index2])
            index2 += 1
    if index1 <len(lst1):
        merged_postings += lst1[index1:]
    if index2 <len(lst2):
        merged_postings += lst2[index2:]
    return merged_postings