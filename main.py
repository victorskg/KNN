import numpy as np
from KNN import KNN
import DataUtils as ut

def main():
    #run_iris()
    #run_column_3C()
    run_column_2C()

def run_iris():
    acs = []
    inputs = [0, 1, 2, 3]
    d = ut.get_iris_data()
    knn = KNN(9, inputs, d)

    print('======== KNN IRIS ========')
    for i in range(10):
        knn = KNN(3, inputs, d)
        acs.append(realization(knn))
        print('Realização: {0}, Acurácia: {1}%'.format(i, acs[i]))
    
    print('Acurácia após 20 realizações: {0}%, Desvio padrão: {1}'.format(calc_accuracy(acs), np.std(acs)))
    
    

def run_column_3C():
    acs = []
    inputs = [0, 1, 2, 3, 4, 5]
    d = ut.get_column_data_3C()

    print('======== KNN COLUMN 3C ========')
    for i in range(10):
        knn = KNN(12, inputs, d)
        acs.append(realization(knn))
        print('Realização: {0}, Acurácia: {1}%'.format(i, acs[i]))
    
    print('Acurácia após 20 realizações: {0}%, Desvio padrão: {1}'.format(calc_accuracy(acs), np.std(acs)))

def run_column_2C():
    acs = []
    inputs = [0, 1, 2, 3, 4, 5]
    d = ut.get_column_data_2C()

    print('======== KNN COLUMN 2C ========')
    for i in range(10):
        knn = KNN(15, inputs, d)
        acs.append(realization(knn))
        print('Realização: {0}, Acurácia: {1}%'.format(i, acs[i]))
    
    print('Acurácia após 20 realizações: {0}%, Desvio padrão: {1}'.format(calc_accuracy(acs), np.std(acs)))

def realization(knn):
    predictions=[]
    for x in range(len(knn.test_set)):
        result = knn.predict(knn.test_set[x])
        predictions.append(result)
	
    return knn.get_accuracy(predictions)

def calc_accuracy(array):
    return sum(array) / len(array) 

if __name__ == '__main__':
    main()