from dimod.generators import and_gate
from dwave.system import DWaveSampler, EmbeddingComposite, LazyFixedEmbeddingComposite

bqm = and_gate('in1', 'in2', 'out')
sampler = EmbeddingComposite(DWaveSampler())
sampler01 = LazyFixedEmbeddingComposite(DWaveSampler())

# sampleset = sampler.sample(bqm, num_reads=1000)
sampleset01 = sampler01.sample(bqm, num_reads=1000)

# print(sampleset)
print(sampleset01)
