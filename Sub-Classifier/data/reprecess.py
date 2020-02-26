from random import shuffle
f1 = open("newpos_train.txt", "r", encoding="utf-8")
lines1 = f1.readlines()
lines1 = ["1," + line1 for line1 in lines1]

f2 = open("pos_test.txt", "r", encoding="utf-8")
lines2 = f2.readlines()
lines2 = ["1," + line2 for line2 in lines2]

f3 = open("n_train.txt", "r", encoding="utf-8")
lines3 = f3.readlines()
lines3 = ["0," + line3 for line3 in lines3]

f4 = open("n_test.txt", "r", encoding="utf-8")
lines4 = f4.readlines()
lines4 = ["0," + line4 for line4 in lines4]


wlines1 = lines1 + lines3
wlines2 = lines2 + lines4

shuffle(wlines1)
shuffle(wlines2)

with open("train.txt", "w", encoding="utf-8") as w1:
    w1.writelines(wlines1)

with open("test.txt", "w", encoding="utf-8") as w2:
    w2.writelines(wlines2)