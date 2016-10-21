import classifiertest
import sys

try:
    checker = classifiertest.error_checker(sys.argv[1])

    if sys.argv[2] == 'train':
        checker.train()
    elif sys.argv[2] == 'predict':
        checker.predict(sys.argv[3])
    else:
        print('Invalid second argument. Expected format is <PATH TO DATASET AND MODEL> <'train' OR 'predict'> <PATH TO CSV FILE (if 'predict')>')

except IndexError:
    print("Required arguments are <PATH TO DATASET AND MODEL> <'train' OR 'predict'> <PATH TO CSV FILE (if 'predict')>")
