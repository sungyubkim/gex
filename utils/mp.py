import jax
from flax.jax_utils import replicate


def cross_replica_mean(replicated):
    return jax.pmap(lambda x: jax.lax.pmean(x,'x'),'x')(replicated)

def unreplicate(tree, i=0):
  """Returns a single instance of a replicated array."""
  return jax.tree_util.tree_map(lambda x: x[i], tree)

def sync_bn(state):
  batch_stats = cross_replica_mean(state.batch_stats)
  return state.replace(batch_stats=batch_stats)