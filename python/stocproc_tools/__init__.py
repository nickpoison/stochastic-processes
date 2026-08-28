from .load_data import load
from .tsplot import tsplot
from .acf import acf1, acf2
from .astsa_col import astsa_col
from .sarima_sim import sarima_sim
from .adj_qstat import adj_qstat
from .qqnorm import qq_norm
from .ts_diag import ts_diag
from .hmm import hmm_fit, viterbi
from .mvspec import mvspec
from .arma_spec import arma_spec
from .timex import timex
from .scatter_hist import scatter_hist

__all__ = ["load", "tsplot", "acf1", "acf2", "astsa_col", "sarima_sim",
           "adj_qstat", "qq_norm", "ts_diag", "hmm_fit", "viterbi",
           "mvspec", "arma_spec", "timex", "scatter_hist"]

# sarima() needs statsmodels, which the rest of this package doesn't --
# import it lazily so `import stocproc_tools` still works without it.
try:
    from .sarima import sarima
    __all__.append("sarima")
except ImportError:
    pass
