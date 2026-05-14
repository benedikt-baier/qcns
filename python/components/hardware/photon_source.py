
import numpy as np
from typing import List, Any

__all__ = ['SinglePhotonSource', 'AtomPhotonSource', 'PhotonPhotonSource']

# single photon source parameters: duration, brightness, g2_0, fidelity, fidelity_variance, visibility
# atom photon source parameters: duration, brightness, g2_0, fidelity, fidelity variance, visibility
# photon photon source parameters: duration, brightness, stats, fidelity, fidelity variance, visibility
# fock photon source parameters: duration, bright state, bright state variance, visibility, emission_prob, leakage_split

SINGLE_PHOTON_SOURCE_MODELS = {
    'perfect': (0., 1., 0., 1., 0., 1.),
    # [1] High-efficiency single-photon source above the loss-tolerant threshold for efficient linear optical quantum computing, 2025
    'ding2025high': (39.4e-9, 0.712, 0.0205, 0.9795, 0.01, 0.9856), # σ²_F aus [15]
    # [2] Deterministic and highly indistinguishable single photons in the telecom C-band, 2025     
    'hauser2026deterministic': (12.5e-9, 0.712, 0.017, 9795, 0.01, 0.950), # α, F aus [1]; σ²_F aus [15]
    # [15] Generation and characterization of polarization-entangled states using quantum dot single photon sources, 2024 
    # LA = Longitudinal Acoustic              
    'valeri2024generation_LA': (12e-9, 0.712, 0.012, 0.92, 0.01, 0.927), # α aus [1]
    # RF = Resonance Fluorescence
    'valeri2024generation_RF': (12e-9, 0.712, 0.016, 0.95, 0.01, 0.949), # α aus [1]
}

ATOM_PHOTON_SOURCE_MODELS = {
    'perfect': (0., 1., 0., 1., 0., 1.),
    # [3] Source of Heralded Atom-Photon Entanglement for Quantum Networking, 2025
    'Ref3': (40e-9, 0.68, 0.012, 0.87, 0.01, 0.9856), # g2, σ²_F aus [15]; ν aus [1]
}

PHOTON_PHOTON_SOURCE_MODELS = {
    'perfect': (0., 1., 'poisson', 1., 0., 1.),
    # [4] Passive demultiplexed two-photon state generation (2025)
    'karli2025passive': (12.5e-9, 1.0, 'poisson', 0.992, 0.001, 0.937), # F, σ²_F aus [16]
    # [5] Bright source of degenerate polarization-entangled photons (2025)
    # without iris filtering
    'bera2025bright': (12.5e-9, 1.0, 'thermal', 0.83, 0.001, 0.976), # d aus [4]; σ²_F aus [16]
    # with iris filtering
    'bera2025bright_iris': (12.5e-9, 1.0, 'thermal', 0.95, 0.001, 0.971), # d aus [4]; σ²_F aus [16]
    # [16] High-quality entangled photon source by symmetric beam displacement (2025)
    'paganini2025high': (12.5e-9, 1.0, 'thermal', 0.992, 0.001, 0.99) # d aus [4]
}

FOCK_PHOTON_SOURCE = {'perfect': (0., 1., 0.5, 0.), 'standard': (3.8e-6, 0.939, 0.05, 0.)}


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
    