# Node for 3 Choice
class Node3:
    def __init__(self, light=None, flag=0, g=None, r=None, y=None):
        self.light = light
        self.flag = 0
        self.g = g
        self.r = r
        self.y = y


# Node for 4 Choice
class Node4:
    def __init__(self, light=None, flag=1, a=None, b=None, c=None, d=None):
        self.light = light
        self.flag = 1
        self.a = a
        self.b = b
        self.c = c
        self.d = d


class Network:
    # combination no. = node(data,flag)
    com1 = Node3([0, 1], 0)
    com2 = Node3([1, 6], 0)
    com3 = Node4([2, 6], 1)
    com4 = Node3([6, 7], 0)
    com5 = Node3([4, 7], 0)
    com6 = Node4([0, 4], 1)
    com7 = Node3([4, 5], 0)
    com8 = Node3([2, 5], 0)

    com9 = Node3([2, 3], 0)
    com10 = Node3([0, 3], 0)

    # Conecttions
    com1.g = com6
    com1.r = com10
    com1.y = com2

    com2.g = com3
    com2.r = com4
    com2.y = com1

    com4.g = com3
    com4.r = com2
    com4.y = com5

    com5.g = com6
    com5.r = com7
    com5.y = com4

    com7.g = com6
    com7.r = com5
    com7.y = com8

    com8.g = com3
    com8.r = com9
    com8.y = com7

    com9.g = com3
    com9.r = com8
    com9.y = com10

    com10.g = com6
    com10.r = com1
    com10.y = com9

    com3.a = com9
    com3.b = com2
    com3.c = com9
    com3.d = com8

    com6.a = com10
    com6.b = com1
    com6.c = com7
    com6.d = com5

    currentSignal = com1
    lastSwitchTime = ([0, 0, 0, 0, 0, 0, 0, 0])
    constChangeInterval = 5
    time = 1

    def __init__(self):
        self.count = None
        self.score = None

    c = constChangeInterval

    def updateLST(self):
        for x in range(len(self.lastSwitchTime)):
            if x in self.currentSignal.light:
                self.lastSwitchTime[x] = 0
            if not (x in self.currentSignal.light):
                self.lastSwitchTime[x] = self.lastSwitchTime[x] + 1
        # print(self.lastSwitchTime)

    def NoVehicle(self):
        flag = 0
        for x in self.currentSignal.light:
            if self.count[x] == 0:
                flag += 1
        if flag == 2:
            return True
        else:
            return False

    def selectComb(self, op, max, assign, curA):
        temp = []
        for e in op:
            if e not in self.currentSignal.light:
                temp.append(e)
        for x in temp:
            if max < self.score[x]:
                max = self.score[x]
                curA = assign
        return [max, curA]

    def solve(self, count):
        self.count = count
        # print(self.count)
        # print(self.currentSignal.light)
        # print()
        self.updateLST()
        if (self.time % self.constChangeInterval) == 0 or self.NoVehicle():
            self.time += 1
            self.score = ([0, 0, 0, 0, 0, 0, 0, 0])
            for i in range(len(self.score)):
                # self.score[i] = self.lastSwitchTime[i]*self.count[i]
                self.score[i] = self.count[i]
            if self.currentSignal.flag == 0:
                cob = self.currentSignal.light

                # compare nex ele
                opr = self.currentSignal.r.light
                opg = self.currentSignal.g.light
                opy = self.currentSignal.y.light
                max = 0
                assign = 'n'
                max, assign = self.selectComb(opr, max, 'r', assign)
                max, assign = self.selectComb(opg, max, 'g', assign)
                max, assign = self.selectComb(opy, max, 'y', assign)
                # print(assign)

                # time on ele
                # maxt = 0
                # for x in cob:
                #     if (maxt < score[x]):
                #         maxt = score[x]
                # if assign != 'n':
                #     if maxt < 5:
                #         time += maxt
                #     else:
                #         time += 5

                # # reduce in liston ele
                # for x in cob:
                #     if score[x] < 5:
                #         score[x] = 0
                #     else:
                #         score[x] = score[x] - 5

                # error check on ele

                if assign == 'r': self.currentSignal = self.currentSignal.r
                if assign == 'g': self.currentSignal = self.currentSignal.g
                if assign == 'y': self.currentSignal = self.currentSignal.y

                flag_chck = 0
                for x in cob:
                    if self.count[x] == 0:
                        flag_chck += 1
                if (flag_chck == len(cob)) & (assign == 'n'):
                    self.currentSignal = self.com3

                # assign
                return

            if self.currentSignal.flag == 1:
                cob = self.currentSignal.light

                # compare
                opa = self.currentSignal.a.light
                opb = self.currentSignal.b.light
                opc = self.currentSignal.c.light
                opd = self.currentSignal.d.light

                max = 0
                assign = 'n'
                max, assign = self.selectComb(opa, max, 'a', assign)
                max, assign = self.selectComb(opb, max, 'b', assign)
                max, assign = self.selectComb(opc, max, 'c', assign)
                max, assign = self.selectComb(opd, max, 'd', assign)

                # time
                # maxt = 0
                # for x in cob:
                #     if (maxt < score[x]):
                #         maxt = score[x]
                # if assign != 'n':
                #     if maxt < 5:
                #         time += maxt
                #     else:
                #         time += 5
                #
                # # reduce in list
                # for x in cob:
                #     if score[x] < 5:
                #         score[x] = 0
                #     else:
                #         score[x] = score[x] - 5
                # error check
                flag_chck = 0
                for x in cob:
                    if self.count[x] == 0: flag_chck += 1
                if (flag_chck == len(cob)) & (assign == 'n'):
                    self.currentSignal = self.com6

                # assign

                if assign == 'a': self.currentSignal = self.currentSignal.a
                if assign == 'b': self.currentSignal = self.currentSignal.b
                if assign == 'c': self.currentSignal = self.currentSignal.c
                if assign == 'd': self.currentSignal = self.currentSignal.d

                return

        # return ([time, step])

# def check(inp):
#   flag = 0
#   for l in inp:
#     if(l==0) : flag = flag + 1
#   if(flag == 8):return False
#   else : return True
# smart = Network()
# inp = ([3, 2, 5, 5, 2, 3, 5, 1])
# while check(inp):
#     print(smart.currentSignal.light)
#     print(inp)
#     smart.solve(inp)
#     a = smart.currentSignal.light
#     for i in a :
#         if inp[i]!= 0:
#             inp[i] -= 1
# print(smart.currentSignal.light)
# print(inp)