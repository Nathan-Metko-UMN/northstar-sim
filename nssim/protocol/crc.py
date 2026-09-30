"""CRCs used by DJI serial framing.

Both ends of the MCB <-> Jetson link (taproot's ``tap::algorithms`` CRC and Northstar-CV's
``src/uart/crc/crc.cpp``) use the same table-driven, reflected CRCs:

- CRC8: polynomial 0x31 reflected (0x8C), init 0xFF, no final XOR.
- CRC16: polynomial 0x1021 reflected (0x8408), init 0xFFFF, no final XOR (CRC-16/MCRF4XX).
"""

CRC8_INIT = 0xFF
CRC16_INIT = 0xFFFF


def _reflected_table(poly: int, width: int) -> tuple[int, ...]:
    mask = (1 << width) - 1
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = ((crc >> 1) ^ poly) if crc & 1 else (crc >> 1)
        table.append(crc & mask)
    return tuple(table)


CRC8_TABLE = _reflected_table(0x8C, 8)
CRC16_TABLE = _reflected_table(0x8408, 16)


def crc8(data: bytes | bytearray | memoryview, init: int = CRC8_INIT) -> int:
    crc = init
    for byte in data:
        crc = CRC8_TABLE[crc ^ byte]
    return crc


def crc16(data: bytes | bytearray | memoryview, init: int = CRC16_INIT) -> int:
    crc = init
    for byte in data:
        crc = (crc >> 8) ^ CRC16_TABLE[(crc ^ byte) & 0xFF]
    return crc
