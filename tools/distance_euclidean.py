#!/usr/bin/env python

import os
import sys
import argparse
import math


def prepend_zeros(tab, aa_diff):
    return [0 for x in range(aa_diff*9)] + tab


def append_zeros(tab, aa_diff):
    return tab + [0 for x in range(aa_diff*9)]

def distance_euclid(tab1, tab2):
    return math.sqrt(sum((a - b)**2 for a, b in zip(tab1, tab2)))

def distance_tanimoto(tab1, tab2):
    tmp = zip(tab1, tab2)
    common = len([x for x in tmp if x[0] == x[1] and x[0] > 0])
    a = len([x for x in tmp if x[0] != x[1] and x[0] > 0])
    b = len([x for x in tmp if x[0] != x[1] and x[1] > 0])
    return float(common)/(a+b-common)

usage = "usage: python %prog [options] patternfile1 patternfile2 ..."
parser = argparse.ArgumentParser(usage=usage)
parser.add_argument('-t', dest='tanimoto', action='store_true', help="Calculate Tanimoto distance instead of Euclidean.")
parser.add_argument("-o", "--output", dest="outfile", default='out_pattern_match.dat',
                  help="Write output to a file")
parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                  help="Toggle to print the output to the console as well")
parser.add_argument("-p", "--reference", dest="ref_file",
                  type="string", help="File storing reference fingerprint")

(options, args) = parser.parse_args()
# ==============================================================================

if not os.path.exists(options.ref_file):
    print("Can't find %s, exiting..." % options.pattern_file)
    sys.exit(-1)

ref_pattern_fh = open(options.ref_file)
min_aa, ref_pattern_str = ref_pattern_fh.readline().split(':')
min_aa = int(min_aa)

outfile = open(options.outfile, 'w')

ref_pattern = [float(x) for x in ref_pattern_str.split(' ')]

for infile in args:
    if not os.path.exists(infile):
        print("Can't find %s, skipping..." % infile)
        continue
    with open(infile) as alt_fh:
        for fp in alt_fh:
            aa, alt_pattern_str = fp.split(':')
            alt_pattern = [float(x) for x in alt_pattern_str.split(' ')]
            if int(aa) < min_aa:
                ref_pattern = prepend_zeros(pattern, min_aa-int(aa))
                min_aa = int(aa)
            if int(aa) > min_aa:
                alt_pattern = prepend_zeros(list(alt_pattern), int(aa)-min_aa)

            if len(alt_pattern) < len(ref_pattern):
                alt_pattern = append_zeros(
                    list(alt_pattern),  len(ref_pattern)-len(alt_pattern))
            elif len(alt_pattern) > len(ref_pattern):
                pattern = append_zeros(
                    list(ref_pattern),  len(alt_pattern)-len(pattern))
            if options.tanimoto:
                dist = distance_tanimoto(ref_pattern, alt_pattern)
            else:
                dist = distance_euclid(ref_pattern, alt_pattern)
            if options.verbose:
                print("Distance from pattern %s is %0.2f" % (infile, dist))

            outfile.write("%s:%0.2f\n" % (infile, dist))

outfile.close()
