import random
import copy

with open("pos_train.txt", "r", encoding="utf-8") as r1:
	line1 = r1.readlines()

with open("add_classifier.txt", "r", encoding="utf-8") as r2:
	line2 = r2.readlines()
print(len(line2))
line2 = list(set(line2))
newlines = copy.deepcopy(line2 * 3)

print(len(line2))
print(len(newlines))

wl = []
for line in line1:
	wl.append(line)
	if len(newlines) > 0:
		tmp = newlines.pop(random.randint(0, len(newlines)-1))
		wl.append(tmp)

with open("newpos_train.txt", "w", encoding="utf-8") as wf:
	print(len(wl))
	wf.writelines(wl)
	
with open("newtest.txt", "w", encoding="utf-8") as wf1:
	wf1.writelines(["1," + each for each in line2])