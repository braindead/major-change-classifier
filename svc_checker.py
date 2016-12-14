import sys
import SVC

checker = SVC.Checker()

if len(sys.argv) > 1:
    checker.predict(sys.argv[1])
else:
    checker.predict('./data.csv')
