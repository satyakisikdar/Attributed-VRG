# ENZYMES
python -m train --datadir=data --bmname=ENZYMES --max-nodes=100 --num-classes=6

# ENZYMES - Diffpool
python -m train --bmname=ENZYMES --assign-ratio=0.1 --hidden-dim=30 --output-dim=30 --num-classes=6 --method=soft-assign

# DD
python -m train --datadir=data --bmname=DD --max-nodes=500 --epochs=1000 --num-classes=2

# DD - Diffpool
python -m train --bmname=ENZYMES --assign-ratio=0.1 --hidden-dim=64 --output-dim=64 --num-classes=2 --method=soft-assign
