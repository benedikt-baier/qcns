import numpy as np
from typing import List

from .photon_source import SinglePhotonSource, AtomPhotonSource, PhotonPhotonSource
from .photon_detector import PhotonDetector, ThresholdDetector, PNRDetector
from .device import Interaction_Device, BSM_Device, FS_Device

class QuantumError:
    
    pass

__all__ = ['QChannel_Model', 'PChannel_Model', 'SQC_Model', 'SRC_Model', 'EQC_Model', 'TPSC_Model', 'BSMC_Model', 'FSC_Model', 'L3C_Model']

SR_MODELS = {}
TPS_MODELS = {}
BMS_MODELS = {}
FOCK_MODELS = {}

class EQC_Model:
    
    def __init__(self):
        pass

class QChannel_Model:
    
    def __init__(self, length: float=0., attenuation: float=0., in_coupling: float=1., out_coupling: float=1.):
        
        if length < 0.:
            raise ValueError(f'Length of connection cannot be negative: {length}')
        
        if not (0. <= in_coupling <= 1.):
            raise ValueError(f'In coupling probability needs to be a probability: {in_coupling}')
        
        if not (0. <= out_coupling <= 1.):
            raise ValueError(f'In coupling probability needs to be a probability: {out_coupling}')
        
        self._length: float = length
        self._propagation_time: float = length * 5e-6
        self._attenuation: float = attenuation
        self._in_coupling: float = in_coupling
        self._out_coupling: float = out_coupling

class PChannel_Model:
    
    def __init__(self, length: float=0., data_rate: float=-1):
        
        if length < 0.:
            raise ValueError('Length of connection cannot be negative')
            
        if data_rate < -1:
            raise ValueError(f'Data rate cannot be negative: {data_rate}')
        
        self._length: float = length
        self._data_rate: float = data_rate

class SQC_Model:
    
    def __init__(self, source_model: SinglePhotonSource=SinglePhotonSource(), 
                 num_sources: int=-1, channel_model: QChannel_Model=QChannel_Model(), 
                 channel_errors: QuantumError | List[QuantumError]=None):
        
        if num_sources == 0 or num_sources < -1:
            raise ValueError(f'Number of Sources should be either -1 or positive: {num_sources}')
        
        self._source_model: SinglePhotonSource = source_model
        self._num_sources: int = num_sources
        self._qchannel_model: QChannel_Model = channel_model
        self._channel_errors: List[QuantumError] = channel_errors
        
        if self._channel_errors is None:
            self._channel_errors = []
            
        if not isinstance(self._channel_errors, list):
            self._channel_errors = [self._channel_errors]

class SRC_Model(EQC_Model):
    
    def __init__(self, source: AtomPhotonSource=AtomPhotonSource(), num_sources: int=-1,
                 detector: PhotonDetector=PNRDetector(), 
                 device: Interaction_Device=Interaction_Device(), 
                 qchannel: QChannel_Model=QChannel_Model(),
                 pchannel: PChannel_Model=PChannel_Model()):
        
        if num_sources == 0. or num_sources < -1:
            raise ValueError(f'Number of Sources should be either -1 or positive: {num_sources}')
        
        self._connection_type: str = 'sr'
        
        self._device: Interaction_Device = device
        self._source: AtomPhotonSource = source
        self._num_sources: int = num_sources
        self._detector: PhotonDetector = detector
        self._qchannel: QChannel_Model = qchannel
        self._pchannel: PChannel_Model = pchannel
        
class TPSC_Model(EQC_Model):
    
    def __init__(self, source: PhotonPhotonSource=PhotonPhotonSource(), num_sources: int=-1,
                 sender_detector: PhotonDetector=PNRDetector(), 
                 sender_qchannel: QChannel_Model=QChannel_Model(),
                 sender_pchannel: PChannel_Model=PChannel_Model(),
                 sender_device: Interaction_Device=Interaction_Device(),
                 receiver_detector: PhotonDetector=PNRDetector(),
                 receiver_qchannel: QChannel_Model=QChannel_Model(),
                 receiver_pchannel: PChannel_Model=PChannel_Model(),
                 receiver_device: Interaction_Device=Interaction_Device()):
        
        if num_sources == 0. or num_sources < -1:
            raise ValueError(f'Number of Sources should be either -1 or positive: {num_sources}')
        
        self._connection_type: str = 'tps'
        
        self._source: PhotonPhotonSource = source
        self._num_sources: int = num_sources
        self._sender_detector: PhotonDetector = sender_detector
        self._sender_qchannel: QChannel_Model = sender_qchannel
        self._sender_pchannel: PChannel_Model = sender_pchannel
        self._sender_device: Interaction_Device = sender_device
        self._receiver_detector: PhotonDetector = receiver_detector
        self._receiver_qchannel: QChannel_Model = receiver_qchannel
        self._receiver_pchannel: PChannel_Model = receiver_pchannel
        self._receiver_device: Interaction_Device = receiver_device
        
class BSMC_Model(EQC_Model):
    
    def __init__(self, device: BSM_Device=BSM_Device(), 
                 sender_source: AtomPhotonSource=AtomPhotonSource(), 
                 receiver_source: AtomPhotonSource=AtomPhotonSource(), 
                 num_sources: int=-1,
                 sender_detector: PhotonDetector=PNRDetector(), 
                 receiver_detector: PhotonDetector=PNRDetector(),
                 sender_qchannel: QChannel_Model=QChannel_Model(),
                 receiver_qchannel: QChannel_Model=QChannel_Model(),
                 sender_pchannel: PChannel_Model=PChannel_Model(),
                 receiver_pchannel: PChannel_Model=PChannel_Model()):
        
        if num_sources == 0. or num_sources < -1:
            raise ValueError(f'Number of Sources should be either -1 or positive: {num_sources}')
        
        self._connection_type: str = 'bsm'
        
        self._device: BSM_Device = device
        self._sender_source: AtomPhotonSource = sender_source
        self._receiver_source: AtomPhotonSource = receiver_source
        self._sender_detector: PhotonDetector = sender_detector
        self._receiver_detector: PhotonDetector = receiver_detector
        self._num_sources: int = num_sources
        self._sender_qchannel: QChannel_Model = sender_qchannel
        self._receiver_qchannel: QChannel_Model = receiver_qchannel
        self._sender_pchannel: PChannel_Model = sender_pchannel
        self._receiver_pchannel: PChannel_Model = receiver_pchannel
        
class FSC_Model(EQC_Model):
    
    def __init__(self, device: FS_Device=FS_Device(), 
                 sender_source: AtomPhotonSource=AtomPhotonSource(brightness=0.5), receiver_source: AtomPhotonSource=AtomPhotonSource(brightness=0.5), 
                 num_sources: int=-1,
                 sender_detector: PhotonDetector=PNRDetector(), receiver_detector: PhotonDetector=PNRDetector(),
                 sender_qchannel: QChannel_Model=QChannel_Model(),
                 receiver_qchannel: QChannel_Model=QChannel_Model(),
                 sender_pchannel: PChannel_Model=PChannel_Model(),
                 receiver_pchannel: PChannel_Model=PChannel_Model()):
        
        if num_sources == 0. or num_sources < -1:
            raise ValueError(f'Number of Sources should be either -1 or positive: {num_sources}')
        
        self._connection_type: str = 'fs'
        
        self._device: str = device
        self._sender_source: str = sender_source
        self._receiver_source: str = receiver_source
        self._sender_detector: str = sender_detector
        self._receiver_detector: str = receiver_detector
        self._num_sources: int = num_sources
        self._sender_qchannel: QChannel_Model = sender_qchannel
        self._receiver_qchannel: QChannel_Model = receiver_qchannel
        self._sender_pchannel: PChannel_Model = sender_pchannel
        self._receiver_pchannel: PChannel_Model = receiver_pchannel
        
class L3C_Model(EQC_Model):
    
    def __init__(self, qchannel: QChannel_Model=QChannel_Model(), pchannel: PChannel_Model=PChannel_Model(),
                 num_sources: int=-1, duration: float=0., success_prob: float=1., fidelity: float=1., fidelity_variance: float=0.):
        
        if num_sources == 0. or num_sources < -1:
            raise ValueError(f'Number of Sources should be either -1 or positive: {num_sources}')
        
        if -np.log10(success_prob) >= 6:
            raise ValueError('Too low success probability')
        
        self._connection_type: str = 'l3c'
        
        self._qchannel: QChannel_Model = qchannel
        self._pchannel: PChannel_Model = pchannel
        self._num_sources: int = num_sources
        self._duration: float = duration
        self._success_prob: float = success_prob
        self._fidelity: float = fidelity
        self._fidelity_variance: float = fidelity_variance
