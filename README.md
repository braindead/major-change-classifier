# Major change classifier

Download embeddings from
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
- unzip to the same folder as 'dataset.csv'
- can be deleted after training/predicting once, when 'embed.dat' and 'embed.vocab' have been created

### Training
From the command line, run:
```
python checker.py <MODEL AND DATA PATH> train
```

### Predicting
From the command line, run:
```
python checker.py <MODEL AND DATA PATH> predict <CSV PATH>
```
