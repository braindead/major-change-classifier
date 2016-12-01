import SVC
import sys


checker = SVC.Checker()

if sys.argv[1] is not None:
    checker.predict(sys.argv[1])
else:
    checker.predict('./data.csv')
