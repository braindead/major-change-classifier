import classifier
import sys

try:
    checker = classifiertest.error_checker(sys.argv[1])

    if sys.argv[2] == 'train':
        checker.train()
    elif sys.argv[2] == 'predict':
        checker.predict(sys.argv[3])
    elif sys.argv[2] == 'check':
        checker.check_results()
    else:
        print('Invalid second argument. Expected format is <PATH TO DATASET AND MODEL> <'train' OR 'predict' OR 'check'> <PATH TO CSV FILE (if 'predict')>')

except IndexError:
    print("Required arguments are <PATH TO DATASET AND MODEL> <'train' OR 'predict' OR 'check'> <PATH TO CSV FILE (if 'predict')>")
