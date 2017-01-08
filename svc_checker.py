import sys
import SVC

checker = SVC.Checker()

if len(sys.argv) > 1:
    debug = False

    try:
        if sys.argv[2] == "1":
            debug = True
    except IndexError:
        debug = False

    checker.predict_json(sys.argv[1], debug)
else:
    checker.predict('./data.csv')
