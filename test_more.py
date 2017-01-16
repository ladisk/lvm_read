"""
Unit test for lvm_read.py
"""

import numpy as np
from lvm_read import read
import os
import io

def test_1():
    lvm="""\
LabVIEW Measurement	
Writer_Version	0.92
Reader_Version	1
Separator	Tab
Multi_Headings	Yes
X_Columns	Multi
Time_Pref	Absolute
Operator	mt
Date	2016/12/12
Time	09:54:07,284627
***End_of_Header***	
	
Channels	3					
Samples	4		4		4	
Date	2016/12/12		2016/12/12		2016/12/12	
Time	09:54:07,483999		09:54:07,483999		09:54:07,483999	
Y_Unit_Label	g		g		g	
X_Dimension	Time		Time		Time	
X0	0.0000000000000000E+0		0.0000000000000000E+0		0.0000000000000000E+0	
Delta_X	0.000250		0.000250		0.000250	
***End_of_Header***						
X_Value	ax	X_Value	ay	X_Value	az	Comment
0.000000	-0.008807	0.000000	-0.028189	0.000000	0.021503
0.000250	-0.025979	0.000250	-0.031060	0.000250	-0.005606
0.000500	-0.011987	0.000500	-0.013517	0.000500	0.007789
0.000750	0.059248	0.000750	-0.021172	0.000750	-0.009433

"""
    f = open('test.lvm', 'w')
    f.write(lvm)
    f.close()
    data = read('test.lvm', read_from_pickle=False, dump_file=False)
    np.testing.assert_equal(data[0]['data'][0,1],-0.008807)

    f = io.open('test.lvm', 'w', newline='\r\n')
    f.write(unicode(lvm))
    f.close()
    data = read('test.lvm', read_from_pickle=False, dump_file=False)
    np.testing.assert_equal(data[0]['data'][0,1],-0.008807)

    os.remove('test.lvm')

if __name__ == '__main__':
    np.testing.run_module_suite()
