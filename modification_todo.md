# Updates to implement modification
* preds that can be modifiers: Adj, VI, VT * preds that can be 
* allow Adj, VI, VT preds to be modifiers
* allow N preds only 


# Valid predicate-category pairs (predcats)
* valid generated predcats:
    * mod cat w/ Adj/VI/VT pred -
        * from modification
    * arg cat, depth 0 w/ any pred - 
        * from arg attachment
    * arg cat, depth 1 w/ Adj/VI/VT pred (overlaps with mod cats) -
        * from arg attachment
    * arg cat, depth 2 w/ VT pred -
        * from arg attachment

* valid parent predcats:
    * any cat, depth 0 w/ any pred
        * does modification
    * any cat, depth 1 w/ Adj/VI/VT pred
        * does modification
    * any cat, depth 2 w/ VT pred --
        * does modification
    * res cat, depth 0 w/ Adj/VI/VT pred (overlaps with depth-0 arg cats) -
        * does arg attachment
    * res cat, depth 1 w/ VT pred (overlaps with depth-1 arg cats) -
        * does arg attachment

#    * mod cat w/ Adj/VI/VT pred --
#        * does modification
#    * arg cat, depth 0 w/ any pred --
#        * does modification
#    * arg cat, depth 1 w/ Adj/VI/VT pred (overlaps with mod cats) --
#        * does modification
#    * arg cat, depth 2 w/ VT pred --
#        * does modification
#    * primitive cat w/ any pred (overlaps with depth-0 arg cats) --
#        * does modification

all_cats = valid preterminal predcats:
* argument
    * arg cat, depth 0 w/ any pred -
    * arg cat, depth 1 w/ Adj/VI/VT pred -
    * arg cat, depth 2 w/ VT pred -
* modifier
    * mod cat w/ Adj/VI/VT pred -
* functor
    * depth 1 cat w/ Adj/VI/VT pred
    * depth 2 cat w/ VT pred
* modificand
    *  any parent predcat from modification -
* lone word
    * primitive cat w/ any pred -


# Old notes -- obsolete because Noun and Adj predicates shouldn't be grouped together
valid parent predcats:
* u-av cat w/ NA pred - 
* primitve cat w/ NA pred -
* arg cat w/ NA pred -
* res cat, depth 0-1 w/ VT pred - 
* res cat, depth 0 w/ VI pred - 

valid generated predcats:
* u-av cat w/ NA pred - 
* arg cat w/ NA pred - 

all_cats = valid preterminal predcats:
* argument
    * arg cat w/ NA pred -
* modifier
    * u-av cat w/ NA pred -
* functor
    * depth 1-2 w/ VT pred -
    * depth 1 with VI pred -
* modificand
    * arg cat w/ NA pred -
    * u-av cat w/ NA pred -
    * primitve cat w/ NA pred -
* lone word
    * primitive cat w/ NA, VT, VI pred -

DEBUG: ix2predcat_all: bidict({0: (0, 0), 1: (1, 0), 2: (3, 0), 3: (2, 0), 4: (3, 1), 5: (2, 1), 6: (0, 1), 7: (1, 1), 8: (0, 2), 9: (1, 2), 10: (3, 2), 11: (2, 2), 12: (3, 3), 13: (2, 3), 14: (3, 4), 15: (3, 5), 16: (3, 6), 17: (3, 7)})