"""Select U forward test epochs from 20 completed validation sources per seed."""
import argparse,json
from pathlib import Path
from peas.ablations.selection import select_forward_epochs
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',required=True);p.add_argument('--smoke',action='store_true')
a=p.parse_args();print(json.dumps(select_forward_epochs(a.output,a.seeds,smoke=a.smoke),indent=2))
