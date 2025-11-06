__all__ = ['SinglePhotonSource', 'AtomPhotonSource', 'PhotonPhotonSource', 'FockPhotonSource']

# single photon source parameters: duration, fidelity, fidelity variance, emission prob
# atom photon source parameters: duration, fidelity, fidelity variance, emission prob
# photon photon source parameters: duration, fidelity, fidelity variance, emission prob, indistinguishability
# fock photon source parameters: duration, bright state, bright state variance, emission prob

SINGLE_PHOTON_SOURCE_MODELS = {'perfect': (0., 1., 0., 1.), 'standard': None}
ATOM_PHOTON_SOURCE_MODELS = {'perfect': (0., 1., 0., 1.), 'standard': (3e-6, 0.941, 0.0005, 0.8), 'krutyanskiy': (3.5e-4, 0.938, 0.04, 0.75)}
PHOTON_PHOTON_SOURCE_MODELS = {'perfect': (0., 1., 0., 1., 1.), 'standard': (126e-6, 0.880, 0.002, 0.65, 0.901)}
FOCK_PHOTON_SOURCE = {'perfect': (0., 0.5, 0., 1.), 'standard': (3.8e-6, 0.05, 0., 0.939)}

# atom photon source:
#   standard: Entangling single atoms over 33 km telecom fibre
#   Krutyanskiy: Entanglement of trapped-ion qubits separated by 230 meters.

# photon photon source
#   standard: A solid-state source of strongly entangled photon pairs with high brightness and indistinguishability

# fock photon source
#   standard: Realization of a multinode quantum network of remote solid-state qubits

class SinglePhotonSource: 
    
    """
    Represents a single photon source
    
    Attr:
        _duration (float): duration to emit photon
        _fidelity (float): fidelity of emitted state
        _fidelity_variance (float): variance of the fidelity of the emitted state
        _emission_prob (float): emission proability of source
    """
    
    def __init__(self, _model_name: str='perfect') -> None:
        
        """
        Initializes a single photon source
        
        Args:
            _model_name (str): single photon source model
            
        Returns:
            /
        """
        
        _model = SINGLE_PHOTON_SOURCE_MODELS[_model_name]
        
        self._duration: float = _model[0]
        self._fidelity: float = _model[1]
        self._fidelity_variance: float = _model[2]
        self._emission_prob: float = _model[3]
    
class AtomPhotonSource:
    
    """
    Represents a atom photon source
    
    Attr:
        _duration (float): duration to emit photon
        _fidelity (float): fidelity of emitted state
        _fidelity_variance (float): variance of the fidelity of the emitted state
        _emission_prob (float): emission probability of source
    """
    
    def __init__(self, _model_name: str='perfect') -> None:
        
        """
        Initializes a atom photon source
        
        Args:
            _model_name (str): atom photon source model
            
        Returns:
            /
        """
        
        _model = ATOM_PHOTON_SOURCE_MODELS[_model_name]
        self._duration: float = _model[0]
        self._fidelity: float = _model[1]
        self._fidelity_variance: float = _model[2]
        self._emission_prob: float = _model[3]
        
class PhotonPhotonSource:
    
    """
    Represents a photon photon source
    
    Attr:
        _duration (float): duration to emit photon
        _fidelity (float): fidelity of emitted state
        _fidelity_variance (float): variance of the fidelity of the emitted state
        _emission_prob (float): emission probability of source
        _visibility (float): indistinguishability of photons
    """
    
    def __init__(self, _model_name: str='perfect') -> None:
        
        """
        Initializes a photon photon source
        
        Args:
            _model_name (str): photon photon source model
            
        Returns:
            /
        """
        
        _model = PHOTON_PHOTON_SOURCE_MODELS[_model_name]
        
        self._duration: float = _model[0]
        self._fidelity: float = _model[1]
        self._fidelity_variance: float = _model[2]
        self._emission_prob: float = _model[3]
        self._visibility: float = _model[4]
        
class FockPhotonSource:
    
    """
    Represents a photon source for Fock state connection
    
    Attrs:
        _duration (float): duration to emit photon
        _alpha (float): bright state population
        _fidelity_variance (float): variance of bright state population
        _emission_prob (float): emission probability of source
    """
    
    def __init__(self, _model_name: str='perfect') -> None:
        
        _model = FOCK_PHOTON_SOURCE[_model_name]
        
        self._duration: float = _model[0]
        self._alpha: float = _model[1]
        self._alpha_variance: float = _model[2]
        self._emission_prob: float = _model[3]