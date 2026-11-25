"""Architectural regressions beyond the supplied snapshot programs."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import unittest

MASK = (1 << 64) - 1
AOK, HLT, ADR, INS = 1, 2, 3, 4


def word(value):
    return (value & MASK).to_bytes(8, 'little')


def immediate(register, value):
    return bytes((0x30, 0xF0 | register)) + word(value)


def signed(value):
    value &= MASK
    return value if value < (1 << 63) else value - (1 << 64)


class CornerCases(unittest.TestCase):
    command = ['./cpu']

    def run_program(self, program, extra=()):
        segments = [(0, program), *extra]
        source = ''.join(f'0x{addr:x}: {data.hex()}\n' for addr, data in segments)
        result = subprocess.run(self.command, input=source, text=True,
                                capture_output=True, timeout=2, check=True)
        return json.loads(result.stdout)

    def test_push_and_pop_stack_pointer(self):
        states = self.run_program(immediate(4, 256) + b'\xa0\x4f\xb0\x4f\x00')
        self.assertEqual(states[1]['REG']['rsp'], 248)
        self.assertEqual(states[1]['MEM']['248'], 256)
        self.assertEqual(states[2]['REG']['rsp'], 256)
        self.assertEqual(states[-1]['STAT'], HLT)

    def test_memory_faults_still_write_back(self):
        cases = [
            (b'\xa0\x4f', 0, -8, None),
            (b'\x80' + word(64), 0, -8, None),
            (b'\xb0\x0f', 8192, 8200, 0),
            (b'\xb0\x4f', 8192, 0, None),
            (b'\x90', 8192, 8200, None),
            (b'\x50\x04' + word(0), 8192, 8192, 0),
            (b'\x40\x04' + word(0), 8192, 8192, 123),
        ]
        for instruction, stack, expected_stack, expected_rax in cases:
            with self.subTest(instruction=instruction.hex()):
                prefix = immediate(0, 123) + immediate(4, stack)
                states = self.run_program(prefix + instruction)
                before, fault = states[-2:]
                self.assertEqual(fault['STAT'], ADR)
                self.assertEqual(fault['PC'], len(prefix))
                self.assertEqual(fault['REG']['rsp'], expected_stack)
                self.assertEqual(fault['MEM'], before['MEM'])
                self.assertEqual(fault['CC'], before['CC'])
                if expected_rax is not None:
                    self.assertEqual(fault['REG']['rax'], expected_rax)

    def test_memory_boundary_and_unaligned_access(self):
        for address in (257, 8184):
            with self.subTest(address=address):
                program = (immediate(0, -123) + immediate(3, address)
                           + b'\x40\x03' + word(0)
                           + b'\x50\x13' + word(0) + b'\x00')
                state = self.run_program(program)[-1]
                self.assertEqual(state['STAT'], HLT)
                self.assertEqual(state['REG']['rcx'], -123)
        # An eight-byte access starting here crosses the final byte of memory.
        states = self.run_program(immediate(3, 8185) + b'\x40\x03' + word(0))
        self.assertEqual(states[-1]['STAT'], ADR)
        self.assertEqual(states[-1]['MEM'], states[-2]['MEM'])

    def test_fetch_boundaries(self):
        cases = [(8192, b''), (MASK, b''), (8191, b'\x20'),
                 (8191, b'\x70'), (8183, b'\x30\xf0')]
        for target, tail in cases:
            with self.subTest(target=target, tail=tail.hex()):
                extra = [(target, tail)] if tail else []
                states = self.run_program(b'\x70' + word(target), extra)
                self.assertEqual(states[-1]['STAT'], ADR)
                self.assertEqual(states[-1]['PC'], target)
                self.assertEqual(states[-1]['REG'], states[-2]['REG'])
        state = self.run_program(b'\x70' + word(8191), [(8191, b'\x00')])[-1]
        self.assertEqual((state['STAT'], state['PC']), (HLT, 8191))

    def test_invalid_instruction_encodings(self):
        maximum = [0, 0, 6, 0, 0, 0, 3, 6, 0, 0, 0, 0]
        for opcode in range(256):
            icode, ifun = opcode >> 4, opcode & 15
            if icode < len(maximum) and ifun <= maximum[icode]:
                continue
            with self.subTest(opcode=hex(opcode)):
                states = self.run_program(bytes((opcode,)))
                self.assertEqual(len(states), 1)
                self.assertEqual(states[0]['STAT'], INS)
                self.assertEqual(states[0]['PC'], 0)
                self.assertTrue(all(v == 0 for v in states[0]['REG'].values()))
        invalid = ['20f0', '200f', '3000', '30ff', '40f0', '400f',
                   '50f0', '500f', '60f0', '600f', 'a000', 'a0ff',
                   'b000', 'b0ff']
        for instruction in invalid:
            with self.subTest(registers=instruction):
                state = self.run_program(bytes.fromhex(instruction) + word(0))[-1]
                self.assertEqual((state['STAT'], state['PC']), (INS, 0))

    def test_alu_extremes(self):
        values = [0, 1, -1, 2, -2, (1 << 63) - 1, -(1 << 63)]
        for op in range(4):
            for a in values:
                for b in values:
                    with self.subTest(op=op, a=a, b=b):
                        mathematical = [b + a, b - a, b & a, b ^ a][op]
                        expected = signed(mathematical)
                        overflow = int(op < 2 and not -(1 << 63) <= mathematical < (1 << 63))
                        program = (immediate(0, a) + immediate(3, b)
                                   + bytes((0x60 | op, 0x03)) + b'\x00')
                        state = self.run_program(program)[-1]
                        self.assertEqual(state['REG']['rbx'], expected)
                        self.assertEqual(state['CC'], dict(ZF=int(expected == 0),
                                                          SF=int(expected < 0), OF=overflow))

    def test_conditions_and_flag_preservation(self):
        for a, b in [(0, 0), (1, 0), (0, 1), (-1, (1 << 63) - 1),
                     (1, -(1 << 63))]:
            difference = b - a
            zf = int(signed(difference) == 0)
            sf = int(signed(difference) < 0)
            of = int(not -(1 << 63) <= difference < (1 << 63))
            less = bool(sf ^ of)
            conditions = [True, less or zf, less, bool(zf), not zf,
                          not less, not less and not zf]
            for condition, taken in enumerate(conditions):
                with self.subTest(a=a, b=b, condition=condition):
                    prefix = immediate(0, a) + immediate(3, b) + b'\x61\x03'
                    # Moves and nop between the ALU and the branch must retain CC.
                    prefix += immediate(1, 123) + b'\x10' + bytes((0x20 | condition, 0x12))
                    jump_pc = len(prefix)
                    target = jump_pc + 10
                    states = self.run_program(prefix + bytes((0x70 | condition,))
                                              + word(target) + b'\x00\x00')
                    self.assertEqual(states[-1]['PC'], target if taken else jump_pc + 9)
                    self.assertEqual(states[-1]['REG']['rdx'], 123 if taken else 0)
                    self.assertEqual(states[-1]['CC'], dict(ZF=zf, SF=sf, OF=of))

    def test_call_return_address(self):
        program = immediate(4, 256) + b'\x80' + word(32) + b'\x00'
        states = self.run_program(program, [(32, immediate(0, 42) + b'\x90')])
        self.assertEqual(states[1]['PC'], 32)
        self.assertEqual(states[1]['MEM']['248'], 19)
        self.assertEqual(states[-1]['PC'], 19)
        self.assertEqual(states[-1]['REG']['rsp'], 256)
        self.assertEqual(states[-1]['REG']['rax'], 42)
        self.assertEqual(states[-1]['STAT'], HLT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin', default=str(Path(__file__).resolve().with_name('cpu')))
    args, remaining = parser.parse_known_args()
    CornerCases.command = shlex.split(args.bin)
    unittest.main(argv=[__file__, *remaining])
