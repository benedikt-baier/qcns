
__all__ = ['Interaction_Device', 'BSM_Device', 'FS_Device']

# (duration, interaction_probability, state_transfer_fidelity)
INTERACTION_DEVICE = {
    'perfect': (0., 1., 1.), 
    # [9] Photon bound state dynamics from a single artificial atom (2023)
    # [10] High-fidelity remote entanglement of trapped atoms mediated by time-bin photons (2025)
    'tomm2023photon': (135e-12, 0.93, 0.97) # FST aus [10]
}

SENDER_RECEIVER_MODELS = {'perfect': (0., 1., 1.), 'standard': (8.95e-7, 0.14, 0.85)}
TWO_PHOTON_MODELS = {'standard': (8.95e-7, 0.37, 0.85)}

# (duration, visibility, signal_prob)
BSM_DEVICE = {
    'perfect': (0., 1., 1.),
    # [11] Bell-state measurement exceeding 50% success probability with linear optics (2023)
    'bayerbach2023bell': (13.2e-9, 0.9645, 0.579), # duration aus [12]  
    # [12] Boosted Bell-state measurements for photonic quantum computation (2025)
    'hauser2025boosted': (13.2e-9, 0.94, 0.693)
}

# (duration, visibility, coherent_phase, spin_photon_correlation)
FOCK_DEVICE = {
    'perfect': (0., 1., 0., 1.),
    # [13] Generation of multi-photon Fock states at telecommunication wavelength using picosecond pulsed light (2024)
    'sonoyama2024generation': (100e-9, 0.96, 0., 1.),    # V aus [14];
    'sonoyama2024generation_130ns': (130e-9, 0.96, 0., 1.), # V aus [14]
    # [14] Neural Network Enhanced Single-Photon Fock State Tomography (2024)
    'hsieh2024neural': (100e-9, 0.96, 0., 1.),    # duration aus [13]
}

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