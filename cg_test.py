from cg_type import CGNode, generate_categories_by_depth, \
    category_from_string, read_categories_from_file
import bidict, sys

p0 = CGNode("0")
p1 = CGNode("1")

x = CGNode('-a', p0, p0)
y = CGNode('-a', p0, p1)
z = CGNode('-a', p0, p1)

print(x)
print(y)
print(x==y)
print(y==z)

ix2cat = bidict.bidict()
ix2cat[0] = x
ix2cat[1] = y

print(ix2cat)

print(ix2cat.inverse[CGNode('-a', p0, p0)])

print(category_from_string("{{0-a1}-b2}"))
print(category_from_string("{{0-bfoo}-bbar}"))
print(category_from_string("{{0-cfoo}-bbar}"))
print(category_from_string("0"))
print(category_from_string("1"))
print(category_from_string("{{{0-a0}-a{0-a0}}-a{{1-b1}-b{1-b1}}}"))

all_cats, res_cats, arg_cats, ix2cat = read_categories_from_file(sys.argv[1])
print(all_cats)
print(res_cats)
print(arg_cats)
print(ix2cat)

# primitives, depth(, argdepth)
#cs_dleq, ix2cat = generate_categories(5, 3, 1)
#cs_dleq, ix2cat = generate_categories(2, 2)
#cats = cs_dleq[2]
#print("num cats: {}".format(len(cats)))
#print(cats)
#print("ix2cat: {}".format(ix2cat))
#
#cats_d1 = cs_dleq[1]
#print("num cats of depth 1: {}".format(len(cats_d1)))
#print(cats_d1)
#print("ix2cat: {}".format(list(ix2cat.items())[:len(cats_d1)]))
