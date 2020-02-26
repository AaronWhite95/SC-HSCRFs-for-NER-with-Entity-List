import random
with open("newtest.txt", "r", encoding="utf-8") as rf:
	lines1 = rf.readlines()
with open("finetune_neg.txt", "r", encoding="utf-8") as rf:
	lines2 = rf.readlines()

lines = lines1 + lines2
random.shuffle(lines)
with open("new_train.txt", "w", encoding="utf-8") as wf:
	wf.writelines(lines)
