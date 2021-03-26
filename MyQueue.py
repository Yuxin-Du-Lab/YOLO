import numpy as np
class myQueue:
    list = []
    MAXSIZE = 0

    def __init__(self, maxsize):
        self.MAXSIZE = maxsize

    def push(self, element):
        if len(self.list) < self.MAXSIZE:
            self.list.append(element)
        else:
            self.pop()
            self.list.append(element)

    def pop(self):
        if len(self.list) == 0:
            print("empty queue!")
        else:
            self.list.pop(0)

    def check(self):
        print("queue-average:")
        print(self.average())
        return self.list

    def average(self):
        sumArray = np.zeros((15, 15))
        for array in self.list:
            sumArray += array
        sumArray /= len(self.list)
        return sumArray

    def getAverageArray(self):
        return self.average()

