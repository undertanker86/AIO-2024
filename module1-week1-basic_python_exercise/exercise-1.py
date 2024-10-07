def evaluate_clasification(tp, fp, fn):
    if isinstance(tp, int) == False:
        print("tp must be int")
        return False
    
    if isinstance(fp, int) == False:
        print("fp must be int")
        return False
    
    if isinstance(fn, int) == False:
        print("fn must be int")
        return False

    if tp <= 0 or fp <= 0 or fn <= 0:
        print("tp and fp and fn must begreater than zero")
        return False
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall)) 
    print("Precision is {0}".format(precision))
    print("Recall is {0}".format(recall))
    print("F1-Score is {0}".format(f1_score))
    
    return True


    
if __name__ == "__main__":
    a = evaluate_clasification(tp=2, fp=3, fn=4)