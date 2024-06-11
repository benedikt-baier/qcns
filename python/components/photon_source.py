__all__ = ['SinglePhotonSource', 'AtomPhotonSource', 'PhotonPhotonSource']

SINGLE_PHOTON_SOURCE_MODELS = {'perfect': (0., 1., 1.), 'standard': None}
ATOM_PHOTON_SOURCE_MODELS = {'perfect': (0., 1., 1.), 'standard': (1.25e-3, 0.923, 1.), 'sheng_zhi_wang': (5e-2, 0.904, 1.)}
PHOTON_PHOTON_SOURCE_MODELS = {'perfect': (0., 1., 1., 1.), 'standard': (12e-9, 0.88, 0.65, 0.901), 'klaus_joens': (5e-6, 0.817, 1., None)}

class SinglePhotonSource: 
    
    """
    Represents a single photon source
    
    Attr:
        _duration (float): duration to emit photon
        _fidelity (float): fidelity of emitted state
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
        self._emission_prob: float = _model[2]
    
class AtomPhotonSource:
    
    """
    Represents a atom photon source
    
    Attr:
        _duration (float): duration to emit photon
        _fidelity (float): fidelity of emitted state
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
        self._emission_prob: float = _model[2]
        
class PhotonPhotonSource:
    
    """
    Represents a photon photon source
    
    Attr:
        _duration (float): duration to emit photon
        _fidelity (float): fidelity of emitted state
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
        self._emission_prob: float = _model[2]
        self._visibility: float = _model[3]