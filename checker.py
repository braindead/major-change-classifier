import classifier
import sys

try:
    checker = classifier.error_checker('./')

    if sys.argv[1] == 'train':
        checker.train()
    elif sys.argv[1] == 'predict':
        if len(sys.argv) == 2:
            csv_path = './data.csv'
        else:
            csv_path = sys.argv[2]
        checker.predict(csv_path)
    else:
        print "Invalid first argument. Expected format is <'train' OR 'predict'> <PATH TO CSV FILE (if 'predict'; defaults to current directory)>"

except IndexError:
    print "Required arguments are <'train' OR 'predict'> <PATH TO CSV FILE (if 'predict'; defaults to current directory)>"
