import numpy as np


def activation_fun(x):
    return 1 if x >= 0.5 else 0


def go(house, rock, attr):
    x = np.array([house, rock, attr])
    w11 = [0.3, 0.3, 0]
    w12 = [0.4, -0.5, 1]
    weigth1 = np.array([w11, w12])
    weigth2 = np.array([-1, 1])

    sum_hidden = np.dot(weigth1, x)
    print("Value sum hidden layer: " + str(sum_hidden))

    out_hidden = np.array([activation_fun(x) for x in sum_hidden])
    print("Value on exit hidden layer: " + str(out_hidden))

    sum_end = np.dot(weigth2, out_hidden)
    y = activation_fun(sum_end)
    print("exit value: " + str(y))

    return y


result = go(house=1, rock=0, attr=1)

if result == 1:
    print("you like me")

else:
    print("Call me")