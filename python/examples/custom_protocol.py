"""Run from the repository root: python python/examples/custom_protocol.py.

Transport-only extension example: no quantum operation or noise model is implied.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from qcns.python.components.packet.packet import Packet
from qcns.python.components.packet.protocol import L3_FRP, L7_Protocol
from qcns.python.components.network.node import Node
from qcns.python.components.network.qprotocol import L7_QProtocol, QProtocolStack
from qcns.python.components.simulation.simulation import Simulation


class MessageHeader(L7_Protocol):
    def __init__(self, message):
        super().__init__(0, 99, {'message': message})


class MessageProtocol(L7_QProtocol):
    def _handle_payload(self, packet):
        if not isinstance(packet.layer7, MessageHeader):
            raise ValueError('Expected a MessageHeader')
        return packet.payload['message']


class Sender(Node):
    async def run(self):
        packet = Packet(self.id, 2,
                        layer3=L3_FRP(self.id, 2, self.id, 2, 0, 0),
                        layer7=MessageHeader('hello QCNS'))
        packet.l3_set_cf()
        self.send_packet(packet)


class Receiver(Node):
    async def run(self):
        self.message = await self.l7_qprotocol.receive(sender=1, time_out=1)


def main():
    simulation = Simulation()
    sender = Sender(1, simulation)
    receiver = Receiver(2, simulation, qprotocols=QProtocolStack(
        l7_qprotocol=MessageProtocol()))
    sender.set_packet_connection(receiver)
    simulation.run()
    if receiver.message != 'hello QCNS':
        raise RuntimeError('Custom protocol did not deliver the expected message')
    print(receiver.message)


if __name__ == '__main__':
    main()
