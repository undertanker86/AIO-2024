import math
alpha = 0.01
# Given
def is_number (n) :
    try :
        float (n) # Type - casting the string to ‘float ‘.
    # If string is not a valid ‘float ‘ ,
    # it ’ll raise ‘ValueError ‘ exception
    except ValueError :
        return False
    return True


def activation_fuctions(x, activation_name):
    if is_number(x) == False:
        print("x must be a number")
        exit()
    else:
        x = float(x) #Convert to float

    if activation_name != "sigmoid" and activation_name != "relu" and activation_name != "elu":
        print("{0} is not supported".format(activation_name))
        exit()
        
    if activation_name == "sigmoid":
        sigmoid = 1 / (1 + math.exp(-x) )
        print("sigmoid: f({0}) = {1}".format(x, sigmoid))
        
    elif activation_name == "relu":
        if x <= 0:
            relu = 0
            print("relu: f({0}) = {1}".format(x, relu))
        else:
            relu = x
            print("relu: f({0}) = {1}".format(x, relu))
            
    elif activation_name == "elu":
        if x <= 0:
            elu = alpha * (math.exp(x) - 1)
            print("elu: f({0}) = {1}".format(x, elu))
        else:
            elu = x
            print("elu: f({0}) = {1}".format(x, elu))
    
if __name__ == "__main__":
    x = input("Input x = ")
    activation_name = input("Input activation Function ( sigmoid | relu | elu ) : ")
    activation_fuctions(x, activation_name)