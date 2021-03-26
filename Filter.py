import math
class SCANBOX:
    THRESHOLD = 0.0
    CURRENT = 0.0
    CENTERX = 0.0
    CENTERY = 0.0
    WIDTH = 0.0
    RATIO = 0

    def __init__(self, width, centerX, centerY, current, threshold):
        self.WIDTH = width
        self.CENTERX = centerX
        self.CENTERY = centerY
        self.CURRENT = current
        self.THRESHOLD = threshold
        self.RATIO = self.CURRENT / self.THRESHOLD


class FILTER:
    centerX = 0.0
    centerY = 0.0
    # magic
    CONST = 0.5
    WIDTH = 52
    STRIDE = 26
    BOUNDARY = 416

    def scan(self, curruntPointList):
        if len(curruntPointList) > 0:
            notFinish = True
        else:
            notFinish = False
        scanBoxes = []
        print("INIT: size of scanList:" + str(len(scanBoxes)))
        while notFinish:
            # list which points are in the filter
            inPoints = self.listIn(curruntPointList, self.centerX, self.centerY, self.WIDTH)
            numIn = len(inPoints)
            if numIn > 0:
                average = averageSize(inPoints, numIn)
                maxNum = (self.WIDTH ** 2) / (average)
                maxNum = math.log(maxNum)
                if maxNum > 1:
                    threshold = self.CONST * maxNum
                else:
                    threshold = 1.5
                current = numIn
                scanbox = SCANBOX(self.WIDTH, self.centerX, self.centerY, current, threshold)
            else:
                scanbox = SCANBOX(self.WIDTH, self.centerX, self.centerY, 0, 1)
            scanBoxes.append(scanbox)
            notFinish = self.step()
        print("size of scanList:" + str(len(scanBoxes)))
        return scanBoxes

    def step(self):
        if self.centerY < self.BOUNDARY - 2 * self.STRIDE:
            if self.centerX < self.BOUNDARY - self.STRIDE:
                self.centerX += self.STRIDE
            else:
                # X touch the Boundary
                self.centerX = 0
                self.centerY += self.STRIDE
            return True
        else:
            # Y touch the Boundary
            return False

    def listIn(self, PointList, centerX, centerY, width):
        inPoints = []
        for point in PointList:
            if ((point.x > centerX) & (point.x < (centerX + width))
                    & (point.y < centerY) & (point.y > (centerY - width))):
                # the point in filter
                inPoints.append(point)
        return inPoints


def averageSize(PointList, num):
    average = 0.0
    if num > 0:
        for point in PointList:
            average += point.size
        average /= num
    return average
