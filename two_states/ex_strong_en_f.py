#!/usr/bin/env python
"""
Get all excited states for target molecules with f > 0.01
"""
import os
import sys
import os.path

def get_states(line, strong = 0):
    #if 'eV' in line:
    list1 = line.split()
    list2 = list1[8].split('=')
    if strong != 0:
        if float(list2[1]) < 0.01:
            return None
    state = list1[4]+' '+ list2[1] + ' '
    return str(state)

def get_file_lines(filename):
    myfile = open (filename)
    lines = myfile.readlines()
    flen = len (lines)
    compound = os.path.split(filename)[1][:-7]+' '
    return lines,flen,compound
    
def main():
    total=''
    for dirpath, dirnames, filenames in os.walk('.'):
        for f in filenames:
            if os.path.splitext(f)[1]=='.log':
                lines,flen,compound = get_file_lines(f)
                result = compound 
                for i in range(flen):
                    if 'eV' in lines[i]:
                        state = get_states(lines[i],strong=1)
                        if state is None:
                            continue
                        result = result + state 
                total = total +result+'\n'
    print total
    output = open ('ex_properties','w')
    output.write(total)
    output.close()

if __name__ == '__main__':
    main()
 
