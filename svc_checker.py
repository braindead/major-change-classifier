import SVC
import sys
import numpy as np


checker = SVC.Checker()

if sys.argv[1] is not None:
    checker.predict(sys.argv[1])
else:
    checker.predict('./data.csv')
