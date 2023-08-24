import sys, math
from collections import Counter

def get_denominator(gold_counter):
    denom = 0
    total = sum(gold_counter.values())
    for key in gold_counter:
        denom += (gold_counter[key]/total) * math.log((gold_counter[key]/total), 10)

    return denom

def get_numerator(pred_counter, joint_counter):
    num = 0
    total = sum(pred_counter.values())
    for key in joint_counter:
        num += (joint_counter[key]/total) * math.log((joint_counter[key]/pred_counter[key[1]]), 10)

    return num

def get_denominator_2(gold_counter, gold_list):
    denom = 0
    total = sum(gold_counter.values())
    for label in gold_list:
        denom += math.log((gold_counter[label]/total), 10)

    return denom

def get_numerator_2(pred_counter, joint_counter, gold_list, pred_list):
    num = 0
    for gold, pred in zip(gold_list, pred_list):
        num += math.log((joint_counter[(gold, pred)]/pred_counter[pred]), 10)

    return num

all_gold = []
all_pred = []
all_gold_bysent = []
all_pred_bysent = []
gold_pred_joined = []

with open(sys.argv[1]) as f:
    lines = f.readlines()
    for line in lines:
        if line.strip() == "":
            all_gold_bysent.append([])
            all_pred_bysent.append([])
        else:
            # print(line)
            gold, pred = line.strip().split("\t")
            all_gold += gold.split(" ")
            all_pred += pred.split(" ")
            all_gold_bysent.append(gold.split(" "))
            all_pred_bysent.append(pred.split(" "))
            for g, p in zip(gold.split(" "), pred.split(" ")):
                gold_pred_joined.append((g, p))

gold_counter = Counter(all_gold)
pred_counter = Counter(all_pred)
gold_pred_counter = Counter(gold_pred_joined)

# print(get_numerator(pred_counter, gold_pred_counter))
# print(get_denominator(gold_counter))
# print(get_numerator_2(pred_counter, gold_pred_counter, all_gold, all_pred))
# print(get_denominator_2(gold_counter, all_gold))
#
# num1 = get_numerator(pred_counter, gold_pred_counter)
# denom1 = get_denominator(gold_counter)
# num2 = get_numerator_2(pred_counter, gold_pred_counter, all_gold, all_pred)
# denom2 = get_denominator_2(gold_counter, all_gold)
#
# print(num1/denom1)
# print(num2/denom2)

total = 0
denom = get_denominator_2(gold_counter, all_gold)
for g, p in zip(all_gold_bysent, all_pred_bysent):
    if g == p == []:
        print(0.0)
    else:
        sent_num = get_numerator_2(pred_counter, gold_pred_counter, g, p)
        print(sent_num/denom)
#         total += sent_num/denom
# print("total", 1 - total)
