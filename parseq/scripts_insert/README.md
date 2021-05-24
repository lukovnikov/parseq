This directory contains the code for the paper "Insertion-based Tree Decoding".

To run the tree insertion experiments on domain $DOMAIN on gpu $GPU, run this:
```shell script
python overnight_treeinsert_new.py -gpu $GPU -numbered -batsize 10 -userelpos -domain $DOMAIN -lr 0.00005 -oraclemix 1. -evaltrain -goldtemp 0.1 -cosinelr -epochs 201 -dropout 0.2 -numtraj 5 -usechildrels true
```

To run the sequence insertion experiments on domain $DOMAIN, with either binary or uniform, run this:
```shell script
python overnight_seqinsert.py -mode binary/uniform (choose one) -batsize 30 -gpu $GPU -numbered -domain $DOMAIN
```

To run the BERT+Transformer baseline on domain $DOMAIN, run this:
```shell script
python overnight_baseline.py -gpu $GPU -mode tm -domain $DOMAIN -dropout 0.25 -lr 0.00005
```

To run the BERT+TreeGRU baseline on domain $DOMAIN, run this:
```shell script
python overnight_baseline.py -gpu $GPU -mode simpletree -domain $DOMAIN -dropout 0.25 -lr 0.001
```