from rbms.bernoulli_bernoulli.classes import BBRBM
from rbms.classes import EBM
from rbms.potts_bernoulli.classes import PBRBM
from rbms.ising_gaussian.classes import IGRBM
from rbms.bernoulli_gaussian.classes import BGRBM
from rbms.ising_ising.classes import IIRBM

map_model: dict[str, EBM] = {"BBRBM": BBRBM, "PBRBM": PBRBM, "BGRBM": BGRBM, "IGRBM": IGRBM, "IIRBM": IIRBM}
