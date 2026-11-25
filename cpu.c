#include "cpu_common.h"

/* 每个阶段通过 CPU 结构体中的暂存值传递数据。 */
void fetch(CPU *cpu)
{
    uint64_t byte0 = read_mem(cpu, cpu->pc, 1);
    if (cpu->stat != AOK) return;
    cpu->icode = (int)(byte0 >> 4);
    cpu->ifun = (int)(byte0 & 15);

    int max_ifun = 0;
    switch (cpu->icode) {
    case I_RRMOVQ: case I_JMP: max_ifun = 6; break;
    case I_ALU: max_ifun = 3; break;
    case I_HALT: case I_NOP: case I_IRMOVQ: case I_RMMOVQ:
    case I_MRMOVQ: case I_CALL: case I_RET: case I_PUSHQ: case I_POPQ:
        break;
    default:
        cpu->stat = INS;
        return;
    }
    if (cpu->ifun > max_ifun) {
        cpu->stat = INS;
        return;
    }
    if (cpu->icode == I_HALT) {
        cpu->stat = HLT;
        return;
    }

    cpu->valP = cpu->pc + 1;
    cpu->rA = cpu->rB = RNONE;
    cpu->valC = 0;
    switch (cpu->icode) {
    case I_RRMOVQ: case I_IRMOVQ: case I_RMMOVQ: case I_MRMOVQ:
    case I_ALU: case I_PUSHQ: case I_POPQ: {
        uint64_t registers = read_mem(cpu, cpu->valP, 1);
        if (cpu->stat != AOK) return;
        cpu->rA = (int)(registers >> 4);
        cpu->rB = (int)(registers & 15);
        /* RNONE 只可用于 irmovq 的源、pushq/popq 的第二个字段。 */
        int valid;
        if (cpu->icode == I_IRMOVQ)
            valid = cpu->rA == RNONE && cpu->rB < RNONE;
        else if (cpu->icode == I_PUSHQ || cpu->icode == I_POPQ)
            valid = cpu->rA < RNONE && cpu->rB == RNONE;
        else
            valid = cpu->rA < RNONE && cpu->rB < RNONE;
        if (!valid) {
            cpu->stat = INS;
            return;
        }
        cpu->valP++;
        break;
    }
    default: break;
    }
    switch (cpu->icode) {
    case I_IRMOVQ: case I_RMMOVQ: case I_MRMOVQ: case I_JMP: case I_CALL:
        cpu->valC = read_mem(cpu, cpu->valP, 8);
        if (cpu->stat != AOK) return;
        cpu->valP += 8;
        break;
    default: break;
    }
}

void decode(CPU *cpu)
{
    int srcA = RNONE, srcB = RNONE;
    switch (cpu->icode) {
    case I_RRMOVQ: case I_RMMOVQ: case I_ALU: case I_PUSHQ:
        srcA = cpu->rA;
        break;
    case I_POPQ: case I_RET: srcA = RSP; break;
    default: break;
    }
    switch (cpu->icode) {
    case I_RMMOVQ: case I_MRMOVQ: case I_ALU: srcB = cpu->rB; break;
    case I_PUSHQ: case I_POPQ: case I_CALL: case I_RET: srcB = RSP; break;
    default: break;
    }
    cpu->valA = srcA == RNONE ? 0 : cpu->reg[srcA];
    cpu->valB = srcB == RNONE ? 0 : cpu->reg[srcB];
}

void execute(CPU *cpu)
{
    cpu->valE = 0;
    cpu->cnd = 0;
    switch (cpu->icode) {
    case I_RRMOVQ:
        cpu->valE = cpu->valA;
        cpu->cnd = check_cond(cpu, cpu->ifun);
        break;
    case I_IRMOVQ: cpu->valE = cpu->valC; break;
    case I_RMMOVQ: case I_MRMOVQ:
        cpu->valE = cpu->valB + cpu->valC;
        break;
    case I_ALU: {
        uint64_t a = cpu->valA, b = cpu->valB;
        cpu->of = 0;
        /* 无符号计算保证模 2^64 回绕；最高位判断有符号溢出。 */
        switch (cpu->ifun) {
        case 0:
            cpu->valE = b + a;
            cpu->of = (int)((~(a ^ b) & (b ^ cpu->valE)) >> 63);
            break;
        case 1:
            cpu->valE = b - a;
            cpu->of = (int)(((a ^ b) & (b ^ cpu->valE)) >> 63);
            break;
        case 2: cpu->valE = b & a; break;
        case 3: cpu->valE = b ^ a; break;
        }
        cpu->zf = cpu->valE == 0;
        cpu->sf = (int)(cpu->valE >> 63);
        break;
    }
    case I_JMP: cpu->cnd = check_cond(cpu, cpu->ifun); break;
    case I_PUSHQ: case I_CALL: cpu->valE = cpu->valB - 8; break;
    case I_POPQ: case I_RET: cpu->valE = cpu->valB + 8; break;
    default: break;
    }
}

void memory_stage(CPU *cpu)
{
    cpu->valM = 0;
    switch (cpu->icode) {
    case I_RMMOVQ: case I_PUSHQ:
        /* valA 是译码时的旧值，pushq %rsp 也压入旧栈指针。 */
        write_mem(cpu, cpu->valE, 8, cpu->valA);
        break;
    case I_MRMOVQ: cpu->valM = read_mem(cpu, cpu->valE, 8); break;
    case I_CALL: write_mem(cpu, cpu->valE, 8, cpu->valP); break;
    case I_POPQ: case I_RET: cpu->valM = read_mem(cpu, cpu->valA, 8); break;
    default: break;
    }
}

void write_back(CPU *cpu)
{
    int dstE = RNONE, dstM = RNONE;
    switch (cpu->icode) {
    case I_RRMOVQ:
        if (cpu->cnd) dstE = cpu->rB;
        break;
    case I_IRMOVQ: case I_ALU: dstE = cpu->rB; break;
    case I_PUSHQ: case I_POPQ: case I_CALL: case I_RET: dstE = RSP; break;
    default: break;
    }
    if (cpu->icode == I_MRMOVQ || cpu->icode == I_POPQ) dstM = cpu->rA;
    /* 即使访存触发 ADR 也要写回；popq %rsp 的 valM 必须最后覆盖。 */
    if (dstE != RNONE) cpu->reg[dstE] = cpu->valE;
    if (dstM != RNONE) cpu->reg[dstM] = cpu->valM;
}

void update_pc(CPU *cpu)
{
    switch (cpu->icode) {
    case I_CALL: cpu->pc = cpu->valC; break;
    case I_JMP: cpu->pc = cpu->cnd ? cpu->valC : cpu->valP; break;
    case I_RET: cpu->pc = cpu->valM; break;
    default: cpu->pc = cpu->valP; break;
    }
}
