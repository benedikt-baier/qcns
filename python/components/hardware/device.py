
__all__ = ['Interaction_Device', 'BSM_Device', 'FS_Device']

# Sender Receiver: duration, interaction probability, state transfer fidelity
# Two photon source: duration, interaction probability, state transfer fidelity,
# Bell State measurement: duration, visibility, coin ph ph, coin ph dc, coin dc dc
# Fock State: duration, visibility, coherent phase, spin photon correlation

INTERACTION_DEVICE = {}

SENDER_RECEIVER_MODELS = {'perfect': (0., 1., 1.), 'standard': (8.95e-7, 0.14, 0.85)}
TWO_PHOTON_MODELS = {'standard': (8.95e-7, 0.37, 0.85)}


BSM_DEVICE = {'perfect': (0., 1., 1., 0., 0.), 'standard': (70e-9, 0.955, 1., 1., 1.), 'leent70ns': (70e-9, 0.955, 1., 1., 1.), 'krut3mus': (3e-6, 0.66, 0.313, 0.355, 0.313), 'krut5mus': (5e-6, 0.45, 0.487, 0.546, 0.49), 
              'krut7mus': (7e-6, 0.37, 0.635, 0.698, 0.64), 'krut10mus': (10e-6, 0.36, 0.811, 0.86, 0.816), 'krut15mus': (15e-6, 0.25, 0.98, 0.988, 0.98)}

FOCK_DEVICE = {'perfect': (0., 1., 0., 0.), 'standard': (3.8e-6, 0.9, 0., 0.01)}

class Interaction_Device:
    
    def __standard_init(self, duration: float=0., interaction_probability: float=1., state_transfer_fidelity: float=1.):
        
        if duration < 0.:
            raise ValueError(f'Duration should be positive: {duration}')
        
        if not (0. <= interaction_probability <= 1.):
            raise ValueError(f'Interaction Probability should be between 0 and 1: {interaction_probability}')
        
        if not (0. <= state_transfer_fidelity <= 1.):
            raise ValueError(f'State Transfer Fidelity should be between 0 and 1: {state_transfer_fidelity}')
        
        self._duration: float = duration
        self._interaction_probability: float = interaction_probability
        self._state_transfer_fidelity: float = state_transfer_fidelity
    
    def __model_init(self, model: str='perfect'):
        
        _model = INTERACTION_DEVICE[model]
        
        self.__standard_init(*_model)
    
    def __init__(self, *args, **kwargs):
        
        if args and args[0] == 'model':
            self.__model_init(*args, **kwargs)
            return
        if kwargs and list(kwargs.keys())[0] == 'model':
            self.__model_init(*args, **kwargs)
            return
        
        self.__standard_init(*args, **kwargs)
    
class BSM_Device:
    
    def __standard_init(self, duration: float=0., visibility: float=1., signal_prob: float=1.):
        
        if duration < 0.:
            raise ValueError(f'Duration should be positive')
        
        if not (0. <= visibility <= 1.):
            raise ValueError(f'Visibiility should be between 0 and 1: {visibility}')
        
        if not (0. <= signal_prob <= 1.):
            raise ValueError(f'Signal Probability should be between 0 and 1.: {signal_prob}')
        
        self._duration: float = duration
        self._visibility: float = visibility
        self._signal_prob: float = signal_prob
    
    def __model_init(self, model: str='perfect'):
        
        _model = BSM_DEVICE[model]
        
        self.__standard_init(*_model)
    
    def __init__(self, *args, **kwargs):
        
        if args and args[0] == 'model':
            self.__model_init(*args, **kwargs)
            return
        if kwargs and list(kwargs.keys())[0] == 'model':
            self.__model_init(*args, **kwargs)
            return
        
        self.__standard_init(*args, **kwargs)
    
class FS_Device:
    
    def __standard_init(self, duration: float=0., visibility: float=1., coherent_phase: float=0., spin_photon_correlation: float=1.):
        
        if duration < 0.:
            raise ValueError(f'Duration should be positive: {duration}')
        
        if not (0. <= visibility <= 1.):
            raise ValueError(f'Visibility should be between 0 and 1: {visibility}')
        
        if not (-1 <= coherent_phase <= 1.):
            raise ValueError(f'Coherent Phase should be between -1 and 1: {coherent_phase}')
        
        if not (0. <= spin_photon_correlation <= 1.):
            raise ValueError(f'Spin Photon interaction should be between 0 and 1: {spin_photon_correlation}')
                    
        self._duration: float = duration
        self._visibility: float = visibility
        self._coherent_phase: float = coherent_phase
        self._spin_photon_correlation: float = spin_photon_correlation
        
    def __model_init(self, model: str='perfect'):
        
        _model = FOCK_DEVICE[model]
        
        self.__standard_init(*_model)
    
    def __init__(self, *args, **kwargs):
        
        if args and args[0] == 'model':
            self.__model_init(*args, **kwargs)
            return
        if kwargs and list(kwargs.keys())[0] == 'model':
            self.__model_init(*args, **kwargs)
            return
        
        self.__standard_init(*args, **kwargs)