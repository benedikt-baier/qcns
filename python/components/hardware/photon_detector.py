
__all__ = ['PhotonDetector']

# detector parameters: duration, efficiency, dark_count_prob

PHOTON_DETECTOR_MODELS = {'perfect': (0., 1., 0.), 'standard': (138e-9, 0.85, 4.55e-6), 'krut': (5.5e-6, 0.87, 1.696e-4)}

# photon detector:
#   standard: Entangling single atoms over 33 km telecom fibre
#   Krutyanskiy: Entanglement of Trapped-Ion Qubits Separated by 230 Meters

class PhotonDetector:
    
    """
    Represents a photon detector
    
    Attr:
        _duration (float): duration to detect photon
        _efficiency (float): probability to detect photon
        _dark_count_prob (float): probability for a dark count
    """
    
    def __init__(self, _model_name: str='perfect') -> None:
        
        """
        Initializes a photon detector
        
        Args:
            _model_name (str): name of the model
            
        Returns:
            /
        """
        
        _model = PHOTON_DETECTOR_MODELS[_model_name]
        
        self._duration: float = _model[0]
        self._efficiency: float = _model[1]
        self._dark_count_prob: float = _model[2]