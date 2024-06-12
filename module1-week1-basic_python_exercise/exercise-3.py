import math
import random


def regression_loss_functions(num_samples, loss_name):
    if num_samples.isnumeric() == False:
        print("number of samples must be an integer number")
        exit()
    if loss_name == "MAE":
        sum_loss = 0
        for i in range(int(num_samples)):
            predict = random.uniform(0, 10)
            target = random.uniform(0, 10)
            loss_sample = abs(target - predict)
            sum_loss += loss_sample
            print(
                f"loss name : {loss_name} , sample : {i} , pred : {predict} , target : {target} , loss : {loss_name}")
        print(f"final {loss_name}: {sum_loss / int(num_samples)}")

    if loss_name == "MSE":
        sum_loss = 0
        for i in range(int(num_samples)):
            predict = random.uniform(0, 10)
            target = random.uniform(0, 10)
            loss_sample = math.pow((target - predict), 2)
            sum_loss += loss_sample
            print("loss name : {0} , sample : {1} , pred : {2} , target : {3} , loss : {4}".format(
                loss_name, i, predict, target, loss_sample))
        print("final {0}: {1}".format(loss_name, sum_loss / int(num_samples)))

    if loss_name == "RMSE":
        sum_loss = 0
        for i in range(int(num_samples)):
            predict = random.uniform(0, 10)
            target = random.uniform(0, 10)
            loss_sample = math.pow((target - predict), 2)
            sum_loss += loss_sample
            print("loss name : {0} , sample : {1} , pred : {2} , target : {3} , loss : {4}".format(
                loss_name, i, predict, target, loss_sample))
        print("final {0}: {1}".format(
            loss_name, math.sqrt(sum_loss / int(num_samples))))


if __name__ == "__main__":
    num_samples = input(
        "Input number of samples ( integer number ) which are generated : ")
    loss_name = input("Input loss name : ")
    regression_loss_functions(num_samples, loss_name)
