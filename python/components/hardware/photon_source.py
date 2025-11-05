
import numpy as np
from typing import List, Any

__all__ = ['SinglePhotonSource', 'AtomPhotonSource', 'PhotonPhotonSource']

# single photon source parameters: duration, brightness, g2_0, fidelity, fidelity_variance, visibility
# atom photon source parameters: duration, brightness, g2_0, fidelity, fidelity variance, visibility
# photon photon source parameters: duration, brightness, stats, fidelity, fidelity variance, visibility
# fock photon source parameters: duration, bright state, bright state variance, visibility, emission_prob, leakage_split

SINGLE_PHOTON_SOURCE_MODELS = {'perfect': (0., 1., 1., 0.), 'standard': None} # needs the additional parameters
ATOM_PHOTON_SOURCE_MODELS = {'perfect': (0., 1., 1., 0.), 'standard': (3e-6, 0.8, 0.941, 0.0005), 'krutyanskiy': (3.5e-4, 0.75, 0.938, 0.04)} # needs the additional parameters
PHOTON_PHOTON_SOURCE_MODELS = {'perfect': (0., 1., 1., 0., 1.), 'standard': (126e-6, 0.901, 0.880, 0.002, 0.65)} # needs the additional parameters
FOCK_PHOTON_SOURCE = {'perfect': (0., 1., 0.5, 0.), 'standard': (3.8e-6, 0.939, 0.05, 0.)} # needs the additional parameters

# atom photon source:
#   standard: Entangling single atoms over 33 km telecom fibre
#   Krutyanskiy: Entanglement of trapped-ion qubits separated by 230 meters.

# photon photon source
#   standard: A solid-state source of strongly entangled photon pairs with high brightness and indistinguishability


class PhotonSource:
    
    def __init__(self, duration: float=0., fidelity: float=1., fidelity_variance: float=0., visibility: float=1.):
        
        if duration < 0.:
            raise ValueError(f'Duration should be positive. {duration}')
        
        if not (0. <= fidelity <= 1.):
            raise ValueError(f'Fidelity should be betwee 0 and 1: {fidelity}')
        
        if not (0. <= visibility <= 1.):
            raise ValueError(f'Visibility should be between 0 and 1: {visibility}')
        
        self._duration: float = duration
        self._fidelity: float = fidelity
        self._fidelity_variance: float = fidelity_variance
        self._visibility: float = visibility

        self._pmf: np.ndarray = np.array([0., 1., 0.])
        
    @property
    def emission_prob(self):
        
        return 1 - self._pmf[0]
        
class SinglePhotonSource(PhotonSource):
    
    def __standard_init(self, duration: float=0., brightness: float=1., g2_0: float=0., fidelity: float=1., fidelity_variance: float=0., visibility: float=1.):
        
        if not (0. <= brightness <= 1.):
            raise ValueError(f'Brightness should be between 0 and 1: {brightness}')
        
        if not (0. <= g2_0 <= 1.):
            raise ValueError(f'G2(0) should be between 0 and 1: {g2_0}')
        
        super(SinglePhotonSource, self).__init__(duration, fidelity, fidelity_variance, visibility)
        
        _p2 = 0.5 * g2_0 * brightness ** 2
        _p1 = brightness - 2 * _p2
        _p0 = 1 - brightness + _p2
        
        self._pmf: np.ndarray = np.array([_p0, _p1, _p2]) / (_p0 + _p1 + _p2)
        
    def __model_init(self, model: str='perfect'):
        
        _model = SINGLE_PHOTON_SOURCE_MODELS[model]
        
        if not (0. <= _model[1] <= 1.):
            raise ValueError(f'Brightness should be between 0 and 1: {_model[1]}')
        
        if not (0. <= _model[2] <= 1.):
            raise ValueError(f'G2(0) should be between 0 and 1: {_model[2]}')
        
        _model_red = (_model[0], _model[3], _model[4], _model[5])
        
        super(SinglePhotonSource, self).__init__(*_model_red)
        
        _p2 = 0.5 * _model[2] * _model[1] ** 2
        _p1 = _model[1] - 2 * _p2
        _p0 = 1 - _model[1] + _p2
        
        self._pmf: np.ndarray = np.array([_p0, _p1, _p2]) / (_p0 + _p1 + _p2)
        
    def __init__(self, *args: str | List[float], **kwargs: str | List[float]):
        
        if args and args[0] == 'model':
            self.__model_init(*args, **kwargs)
            return
        if kwargs and list(kwargs.keys())[0] == 'model':
            self.__model_init(*args, **kwargs)
            return
        
        self.__standard_init(*args, **kwargs)

class AtomPhotonSource(PhotonSource):
    
    def __standard_init(self, duration: float=0., brightness: float=1., g2_0: float=0., fidelity: float=1., fidelity_variance: float=0., visibility: float=1.):
        
        if not (0. <= brightness <= 1.):
            raise ValueError(f'Brightness should be between 0 and 1: {brightness}')
        
        if not (0. <= g2_0 <= 1.):
            raise ValueError(f'G2(0) should be between 0 and 1: {g2_0}')
        
        super(AtomPhotonSource, self).__init__(duration, fidelity, fidelity_variance, visibility)
        
        _p2 = 0.5 * g2_0 * brightness ** 2
        _p1 = brightness - 2 * _p2
        _p0 = 1 - brightness + _p2
        
        self._pmf: np.ndarray = np.array([_p0, _p1, _p2]) / (_p0 + _p1 + _p2)
        
    def __model_init(self, model: str='perfect'):
        
        _model = ATOM_PHOTON_SOURCE_MODELS[model]
        
        if not (0. <= _model[1] <= 1.):
            raise ValueError(f'Brightness should be between 0 and 1: {_model[1]}')
        
        if not (0. <= _model[2] <= 1.):
            raise ValueError(f'G2(0) should be between 0 and 1: {_model[2]}')
        
        _model_red = (_model[0], _model[3], _model[4], _model[5])
        
        super(AtomPhotonSource, self).__init__(*_model_red)
        
        _p2 = 0.5 * _model[2] * _model[1] ** 2
        _p1 = _model[1] - 2 * _p2
        _p0 = 1 - _model[1] + _p2
        
        self._pmf: np.ndarray = np.array([_p0, _p1, _p2]) / (_p0 + _p1 + _p2)
        
    def __init__(self, *args, **kwargs):
        
        if args and args[0] == 'model':
            self.__model_init(*args, **kwargs)
            return
        if kwargs and list(kwargs.keys())[0] == 'model':
            self.__model_init(*args, **kwargs)
            return
        
        self.__standard_init(*args, **kwargs)

class PhotonPhotonSource(PhotonSource):
    
    def __standard_init(self, duration: float=0., brightness: float=-1., stats: str='thermal', fidelity: float=1., fidelity_variance: float=0., visibility: float=1.):
        
        if not (0. <= brightness <= 1.) and brightness != -1:
            raise ValueError(f'Brightness should be between 0 and 1: {brightness}')
        
        if stats not in ['thermal', 'poisson']:
            raise ValueError(f'The Photon Model should be either thermal or poisson: {stats}')
        
        super(PhotonPhotonSource, self).__init__(duration, fidelity, fidelity_variance, visibility)
        
        if brightness == -1:
            
            self._pmf: np.ndarray = np.array([0., 1., 0.])
            
            return
        
        if stats == 'thermal':
            
            _p0 = 1 / (1 + brightness)
            _p1 = brightness / (1 + brightness) ** 2
            _p2 = 1 - _p0 - _p1
            
            self._pmf: np.ndarray = np.array([_p0, _p1, _p2]) / (_p0 + _p1 + _p2)
            return
        
        if stats == 'poisson':
            
            _p0 = np.exp(-brightness)
            _p1 = brightness * np.exp(-brightness)
            _p2 = 1 - _p0 - _p1
            
            self._pmf: np.ndarray = np.array([_p0, _p1 + _p2])
            
            return
    
    def __model_init(self, model: str='perfect'):
        
        _model = PHOTON_PHOTON_SOURCE_MODELS[model]
        
        if not (0. <= _model[1] <= 1.):
            raise ValueError(f'Brightness should be between 0 and 1: {_model[1]}')
        
        if _model[2] not in ['thermal', 'poisson']:
            raise ValueError(f'The Photon Model should be either thermal or poisson: {_model[2]}')
        
        _model_red = (_model[0], _model[3], _model[4], _model[5])
        
        super(PhotonPhotonSource, self).__init__(*_model_red)
        
        if _model[2] == 'thermal':
            
            _p0 = 1 / (1 + _model[1])
            _p1 = _model[1] / (1 + _model[1]) ** 2
            _p2 = 1 - _p0 - _p1
            
            self._pmf: np.ndarray = np.array([_p0, _p1, _p2]) / (_p0 + _p1 + _p2)
            return
        
        if _model[2] == 'poisson':
            
            _p0 = np.exp(-_model[1])
            _p1 = _model[1] * np.exp(-_model[1])
            _p2 = 1 - _p0 - _p1
            
            self._pmf: np.ndarray = np.array([_p0, _p1, _p2])
            
            return
    
    def __init__(self, *args, **kwargs):
        
        if args and args[0] == 'model':
            self.__model_init(*args, **kwargs)
            return
        if kwargs and list(kwargs.keys())[0] == 'model':
            self.__model_init(*args, **kwargs)
            return
        
        self.__standard_init(*args, **kwargs)
    