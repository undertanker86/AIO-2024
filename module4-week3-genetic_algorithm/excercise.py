import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(0)

def load_data_from_file(filename):
    data = np.genfromtxt(filename , dtype = None , delimiter = ',', skip_header =1)
    features_X = data[: , :3]
    N = len(features_X)
    sales_Y = data[: , 3]
    X_b = np.c_[np.ones((N, 1)), features_X] 
    return X_b, sales_Y

def create_individual(n=4, bound=10):
    individual = []
    for i in range(n):
        individual.append(random.uniform(-bound/2, bound/2))
    return individual

def compute_loss( individual ):
    theta = np.array( individual )
    y_hat = features_X.dot( theta )
    loss = np.multiply(( y_hat - sales_Y ) , ( y_hat - sales_Y ) ).mean()
    return loss

def compute_fitness( individual ):
    return 1 / ( 1 + compute_loss( individual ) )  

def crossover( individual1 , individual2 , crossover_rate = 0.9):
    individual1_new = individual1.copy()
    individual2_new = individual2.copy()
    for i in range(len( individual1 )):
        if random.random() < crossover_rate:
            individual1_new[i] = individual2[i]
            individual2_new[i] = individual1[i]
    return individual1_new, individual2_new

def mutate(individual , mutation_rate = 0.05, bound=10):
    individual_m = individual.copy()
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual_m[i] = random.uniform(-bound/2, bound/2)
    return individual_m

def initializePopulation(m) :
    population = [ create_individual() for _ in range ( m ) ]
    return population

def selection(sorted_old_population , m = 600) :
    index1 = random.randint(0 , m -1)
    while True:
        index2 = random.randint(0 , m -1)
        if index2 != index1:
            break
    individual_s = sorted_old_population[ index1 ]
    if index2 > index1:
        individual_s = sorted_old_population[ index2 ]
    return individual_s

def create_new_population( old_population , elitism =2 , gen =1) :
    m = len( old_population )
    # print(m)
    sorted_population = sorted( old_population , key = compute_fitness )

    # if gen % 1 == 0:
        # print (" Best loss :" , compute_loss( sorted_population [m -1]) , " with chromsome : " , sorted_population[m -1])

    new_population = []

    while len( new_population ) < m - elitism:
        # selection
        individual1 = selection( sorted_population )
        individual2 = selection( sorted_population )
        # crossover
        individual1, individual2 = crossover( individual1 , individual2 )
        # mutation
        individual1 = mutate( individual1 )
        individual2 = mutate( individual2 )

        new_population.append( individual1 )
        new_population.append( individual2 )

    for i in range( elitism ):
        new_population.append( sorted_population[ m - 1 - i ] )

    return new_population , compute_loss( sorted_population [m -1])

def run_GA():
    n_generations = 100
    m = 600
    # features_X, sales_Y = load_data_from_file('AIO-2024/module4-week3-genetic_algorithm/data/advertising.csv')
    population = initializePopulation( m )
    losses_list = []
    for i in range( n_generations ) :
        population , loss = create_new_population( population , gen = i )
        losses_list.append( loss )
    return population , losses_list

def visualize_loss ( losses_list ):
    plt.plot( losses_list )
    plt.xlabel('Generations')
    plt.ylabel('Loss')
    plt.show()

def visualize_predict_gt(population):
    sorted_population = sorted( population , key = compute_fitness )
    theta = np.array( sorted_population [ -1])

    estimated_prices = []
    for feature in features_X:
        estimated_prices.append( theta.dot( feature ))
    
    fig , ax = plt.subplots( figsize =(10 , 6) )
    plt.xlabel('Samples')
    plt.ylabel('Price')

    plt.plot( sales_Y , c = 'green' , label = 'Real Prices')

    plt.plot( estimated_prices , c = 'blue' , label ='Estimated Prices')

    plt.legend()

    plt.show()

if __name__ == '__main__':
    #Btap1: 
    features_X, sales_Y = load_data_from_file('AIO-2024/module4-week3-genetic_algorithm/data/advertising.csv')
    # print (features_X[:5 ,:])
    # print (sales_Y.shape)

    #Btap2:
    # individual = create_individual()
    # print( individual )

    #Btap3:
    # individual = [4.09 , 4.82 , 3.10 , 4.02]
    # fitness_score = compute_fitness( individual )
    # print( fitness_score )
    
    #Btap4:
    # individual1 = [4.09 , 4.82 , 3.10 , 4.02]
    # individual2 = [3.44 , 2.57 , -0.79 , -2.41]
    # individual1 , individual2 = crossover( individual1 , individual2 , 2.0)
    # print (" individual1 : " , individual1 )
    # print (" individual2 : " , individual2 )

    #Btap5:
    # before_individual = [4.09 , 4.82 , 3.10 , 4.02]
    # after_individual = mutate( before_individual , mutation_rate = 2.0)
    # print ( before_individual == after_individual )

    #Btap6:
    # population = initializePopulation( 3 )
    # print( population )

    #Btap7: selection function

    #Btap8: 
    # individual1 = [4.09 , 4.82 , 3.10 , 4.02]
    # individual2 = [3.44 , 2.57 , -0.79 , -2.41]
    # individual3 = [2.487 , 1.57 , -1.23 , -3.33]
    # old_population = [ individual1 , individual2, individual3 ]
    # new_population , _ = create_new_population( old_population , elitism =  1 , gen =1)
    # print( new_population )

    #Btap9 & 10 & 11:
    # population, losses_list =run_GA()
    # visualize_loss( losses_list )
    # visualize_predict_gt(population)

