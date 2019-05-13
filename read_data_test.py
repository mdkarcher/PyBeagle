from Bio import SeqIO
from Bio import Phylo
from ete3 import Tree

from loglikelihood import *

# data = SeqIO.parse("rbcl/rbcL.nex", 'nexus')
# data_dict = {rec.id:str(rec.seq) for rec in data}
# tree = Tree("rbcl/rbcLjc.tre")

# Phylo.convert("rbcl/rbcLjc.tre", 'nexus', "rbcl/rbcLjc.nwk", 'newick')
# Phylo.convert("rbcl/rbcl10.tre", 'nexus', "rbcl/rbcl10.nwk", 'newick')
# Phylo.convert("rbcl/rbcl738nj.tre", 'nexus', "rbcl/rbcl738.nwk", 'newick')

# nex = Phylo.read("rbcl/rbcLjc.tre", 'nexus')
rbcl = Tree("rbcl/rbcLjc.nwk")
data = SeqIO.parse("rbcl/rbcL.nex", 'nexus')
seqs = {rec.id: str(rec.seq) for rec in data}

# nex10 = Phylo.read("rbcl/rbcl10.tre", 'nexus')
rbcl10 = Tree("rbcl/rbcl10.nwk")
data10 = SeqIO.parse("rbcl/rbcl10.nex", 'nexus')
seqs10 = {rec.id: str(rec.seq) for rec in data10}

# nex738 = Phylo.read("rbcl/rbcl738nj.tre", 'nexus')
rbcl738 = Tree("rbcl/rbcl738.nwk")
data738 = SeqIO.parse("rbcl/rbcl738.nex", 'nexus')
seqs738 = {rec.id: str(rec.seq) for rec in data738}

# likelihood calculations
my_llik = loglikelihood(rbcl, seqs, id_attr="id", leaf_attr="name")
print(f"rbcl (my llik): {my_llik}")
bg_llik = loglikelihood_beagle(rbcl, seqs, id_attr="id", leaf_attr="name", scaling=True)
print(f"rbcl (beagle llik): {bg_llik}")
round(bg_llik - my_llik, 8)

my_llik10 = loglikelihood(rbcl10, seqs10, id_attr="id", leaf_attr="name")
print(f"rbcl (my llik): {my_llik10}")
bg_llik10 = loglikelihood_beagle(rbcl10, seqs10, id_attr="id", leaf_attr="name", scaling=True)
print(f"rbcl (beagle llik): {bg_llik10}")
round(bg_llik10 - my_llik10, 8)

my_llik738 = loglikelihood(rbcl738, seqs738, id_attr="id", leaf_attr="name")
print(f"rbcl (my llik): {my_llik738}")
bg_llik738 = loglikelihood_beagle(rbcl738, seqs738, id_attr="id", leaf_attr="name", scaling=True)
print(f"rbcl (beagle llik): {bg_llik738}")
round(bg_llik738 - my_llik738, 8)


