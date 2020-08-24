# attributed-graphs

Setup VRG conda environment 
```shell
conda env create -f VRG_env.yml
```


To run the VRG code. 
```shell
conda activate VRG
cd VRG
usage: python runner.py [-h] [-g] [-c] [-m MU] [-t] [-o OUTDIR] [-n N] [-p] [-d]
                 [-a ATTR_NAME]

optional arguments:
  -h, --help            show this help message and exit
  -g , --graph          Name of the graph or path to gpickle file (default:
                        karate)
  -c , --clustering     Clustering method to use (default: leiden)
  -m MU, --mu MU        Size of RHS (mu) (default: 4)
  -t , --type           Grammar type (default: VRG)
  -o OUTDIR, --outdir OUTDIR
                        Name of the output directory (default: output)
  -n N                  Number of graphs to generate (default: 5)
  -p, --grammar-pickle  Use pickled grammar? (default: False)
  -d, --cluster-pickle  Use pickled dendrogram? (default: False)
  -a ATTR_NAME, --attr-name ATTR_NAME
                        Name of Attribute (default: )

```

## Grammar Types
* `VRG` is generic VRG, 
* `NCE` is for isomorphic graphs and for node correspondence.

## TODOs
* Add partial and full correpsondence flags
