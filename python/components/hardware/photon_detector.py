
__all__ = ['PhotonDetector', 'ThresholdDetector', 'PNRDetector']

PHOTON_DETECTOR_MODELS = {'perfect': (0., 1., 0.), 'standard': (138e-9, 0.85, 4.55e-6), 'krut': (5.5e-6, 0.87, 1.696e-4)} # need to be changed

THRESHOLD_DETECTOR_MODELS = {}
PNR_DETECTOR_MODELS = {}

# photon detector:
#   standard: Entangling single atoms over 33 km telecom fibre
#   Krutyanskiy: Entanglement of Trapped-Ion Qubits Separated by 230 Meters

class PhotonDetector:
    
    def __init__(self, duration: float=0., quench_time: float=0., dead_time: float=0., after_pulse_duration: float=0., efficiency: float=1., dark_count: float=0., after_pulse_prob: float=0., spectral_overlap: float=1., temporal_overlap: float=1., polarization_overlap: float=1.):
        
        if duration < 0.:
            raise ValueError(f'Duration should be positive: {duration}')
        
        if quench_time < 0.:
            raise ValueError(f'Quench Time should be positive: {quench_time}')
        
        if dead_time < 0.:
            raise ValueError(f'Dead Time should be positive: {dead_time}')
        
        if after_pulse_duration < 0.:
            raise ValueError(f'After pulse duration should be positive: {after_pulse_duration}')
        
        if not (0. <= efficiency <= 1.):
            raise ValueError(f'Efficiency should be between 0 and 1: {efficiency}')
        
        if not (0. <= dark_count <= 1.):
            raise ValueError(f'Dark Count Probability should be between 0 and 1: {dark_count}')
        
        if not (0. <= after_pulse_prob <= 1.):
            raise ValueError(f'After Pulse Probability should be between 0 and 1: {after_pulse_prob}')
        
        if not (0. <= spectral_overlap <= 1.):
            raise ValueError(f'Spectral Overlap should be between 0 and 1: {spectral_overlap}')
        
        if not (0. <= temporal_overlap <= 1.):
            raise ValueError(f'Temporal Overalp should be between 0 and 1: {temporal_overlap}')
        
        if not (0. <= polarization_overlap <= 1.):
            raise ValueError(f'Polarization Overlap should be between 0 and 1: {polarization_overlap}')
        
        self._duration: float = duration
        self._efficiency: float = efficiency * spectral_overlap * temporal_overlap * polarization_overlap
        self._dark_count: float = dark_count
        self._quench_time: float = quench_time
        self._dead_time: float = dead_time
        self._after_pulse_prob: float = after_pulse_prob
        self._after_pulse_duration: float = after_pulse_duration
        
class ThresholdDetector(PhotonDetector):
    
    def __standard_init(self, duration: float=0., efficiency: float=1., dark_count: float=0., quench_time: float=0., dead_time: float=0., after_pulse_prob: float=0., after_pulse_duration: float=0., spectral_overlap: float=1., temporal_overlap: float=1., polarization_overlap: float=1.):
        
        super(ThresholdDetector, self).__init__(duration, quench_time, dead_time, after_pulse_duration, efficiency, dark_count, after_pulse_prob, spectral_overlap, temporal_overlap, polarization_overlap)
        
    def __model_init(self, model='perfect'):
        
        _model = THRESHOLD_DETECTOR_MODELS[model]
        
        super(ThresholdDetector, self).__init__(*_model)
        
    def __init__(self, *args, **kwargs):
        
        if args and args[0] == 'model':
            self.__model_init(*args, **kwargs)
            return
        if kwargs and list(kwargs.keys())[0] == 'model':
            self.__model_init(*args, **kwargs)
            return
        
        self.__standard_init(*args, **kwargs)
        
class PNRDetector(PhotonDetector):
    
    def __standard_init(self, duration: float=0., efficiency: float=1., dark_count: float=0., quench_time: float=0., dead_time: float=0., after_pulse_prob: float=0., after_pulse_duration: float=0., spectral_overlap: float=1., temporal_overlap: float=1., polarization_overlap: float=1.):
        
        super(PNRDetector, self).__init__(duration, quench_time, dead_time, after_pulse_duration, efficiency, dark_count, after_pulse_prob, spectral_overlap, temporal_overlap, polarization_overlap)
        
    def __model_init(self, model='perfect'):
        
        _model = PNR_DETECTOR_MODELS[model]
        
        super(PNRDetector, self).__init__(*_model)
        
    def __init__(self, *args, **kwargs):
        
        if args and args[0] == 'model':
            self.__model_init(*args, **kwargs)
            return
        if kwargs and list(kwargs.keys())[0] == 'model':
            self.__model_init(*args, **kwargs)
            return
        
        self.__standard_init(*args, **kwargs)