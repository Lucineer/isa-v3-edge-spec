# ISA v3 — Edge Encoding Specification

**Document:** ISA-V3-EDGE-ENCODING  
**Author:** JC1  
**Reviewer:** Oracle1  
**Date:** 2026-04-12  
**Target:** Jetson Orin Nano, ARM64, 1024 CUDA cores, 8 GB RAM, bare metal  
**Status:** Draft — Review Requested

---

## 1. Overview

ISA v3 defines a dual-mode instruction set:

| Mode     | Encoding        | Opcode Count | Target         |
|----------|----------------|-------------|----------------|
| **Cloud**| Fixed 4-byte   | ~200        | flux-runtime-c |
| **Edge** | Variable 1-3 B | 256 (80 used)| cuda-instruction-set |

This document specifies the **edge variable-width encoding**. Every byte matters on constrained hardware — the encoding is designed to pack maximum semantics into minimum space while preserving confidence fusion, energy awareness, and trust propagation from the cloud ISA.

---

## 2. Variable-Width Encoding Scheme

Instruction length is determined by the top 2 bits of the first byte:

```
Byte 0:
  0XXXXXXX  → 1-byte instruction (no operands)    range: 0x00–0x7F (128 opcodes)
  10XXXXXX  → 2-byte instruction (1 operand byte)  range: 0x80–0xBF (64 opcodes)
  11XXXXXX  → 3-byte instruction (2 operand bytes)  range: 0xC0–0xFF (64 opcodes)
```

### 2.1 One-Byte Instructions (0x00–0x7F)

```
 ┌────────────────────────────────┐
 │ 0  o6 o5 o4 o3 o2 o1 o0      │
 │                               │
 │ bit 7 = 0 → 1-byte            │
 │ bits 6-0 = opcode (128 slots) │
 └────────────────────────────────┘
```

No operands. Operate on implicit registers (typically r0 = accumulator).

#### Opcode Map (1-byte space)

```
0x00  NOP          No operation
0x01  ADD_r0       r0 += implicit operand (stack top or sensor bus)
0x02  SUB_r0       r0 -= implicit operand
0x03  AND_r0       r0 &= implicit operand
0x04  OR_r0        r0 |= implicit operand
0x05  XOR_r0       r0 ^= implicit operand
0x06  NOT_r0       r0 = ~r0
0x07  SHL_r0       r0 <<= 1
0x08  SHR_r0       r0 >>= 1
0x09  INC_r0       r0++
0x0A  DEC_r0       r0--
0x0B  NEG_r0       r0 = -r0

0x10  PUSH_r0      Push r0 onto stack
0x11  POP_r0       Pop stack into r0
0x12  DUP          Duplicate stack top
0x13  SWAP         Swap stack top two
0x14  DROP         Discard stack top

0x20  HALT         Stop execution
0x21  RET          Return (pop PC from stack)
0x22  IRET         Return from interrupt

0x28  SLEEP        Enter power state in r15[1:0]
0x29  WAKE         Resume ACTIVE state
0x2A  WDOG_RESET   Reset watchdog timer (r15 bit 7)

0x30  CONF_READ    r12 → r0 (copy confidence to accumulator)
0x31  CONF_SET     r0 → r12 (set confidence from accumulator)

0x38  ENERGY_READ  r13 → r0
0x39  ENERGY_SYNC  Sync r13 with hardware ATP counter

0x40  TRUST_READ   r14 → r0
0x41  TRUST_QUERY  Read trust of last message sender → r0

0xF8  BRK          Breakpoint (debug)
0xF9  ILLEGAL      Trap on illegal instruction
0xFF  UNDEF        Reserved / undefined trap
```

Remaining slots (0x0C–0x0F, 0x15–0x1F, 0x23–0x27, 0x2B–0x2F, etc.) are reserved for future extension.

### 2.2 Two-Byte Instructions (0x80–0xBF)

```
 ┌────────────────────────────────┐ ┌────────────────────────────────┐
 │ 1  0  o4 o3 o2 o1 o0          │ │ r3 r2 r1 r0  i3 i2 i1 i0      │
 │                               │ │                                │
 │ Byte 0: prefix(10) + opcode  │ │ Byte 1: reg(4) + imm/flag(4)  │
 └────────────────────────────────┘ └────────────────────────────────┘
```

**Byte 1 layout:**

```
 bits 7-4: register operand (r0–r15)
 bits 3-0: 4-bit immediate value OR condition flags
```

#### Condition Flags (for branch opcodes)

```
  0000 (0x0): Z   — zero flag set
  0001 (0x1): NZ  — zero flag clear
  0010 (0x2): C   — carry flag set
  0011 (0x3): NC  — carry flag clear
  0100 (0x4): N   — negative flag set
  0101 (0x5): NN  — negative flag clear
  0110 (0x6): GT  — signed greater than
  0111 (0x7): LT  — signed less than
  1000 (0x8): EQ  — equal (Z set)
  1001 (0x9): NE  — not equal
  1010 (0xA): HI  — unsigned higher (C && !Z)
  1011 (0xB): LS  — unsigned lower or same
  1100 (0xC): GE  — signed greater or equal
  1101 (0xD): LE  — signed less or equal
  1110 (0xE): CONF — confidence above threshold (r12 > r0)
  1111 (0xF): ALWAYS — unconditional
```

#### Opcode Map (2-byte space)

**Arithmetic (register + 4-bit immediate)**

```
0x80  CADD   rd = rd + imm4,    conf(rd) = bayesian_fuse(conf(rd), conf_imm)
0x81  CSUB   rd = rd - imm4,    conf(rd) = bayesian_fuse(conf(rd), conf_imm)
0x82  CMUL   rd = rd * imm4,    conf(rd) = bayesian_fuse(conf(rd), conf_imm)
0x83  CDIV   rd = rd / imm4,    conf(rd) = bayesian_fuse(conf(rd), conf_imm)

0x84  ADD    rd = rd + imm4
0x85  SUB    rd = rd - imm4
0x86  MUL    rd = rd * imm4
0x87  DIV    rd = rd / imm4 (safe — zero → r0 stays, carry set)
0x88  MOD    rd = rd % imm4
0x89  AND    rd = rd & imm4
0x8A  OR     rd = rd | imm4
0x8B  XOR    rd = rd ^ imm4
0x8C  SHL    rd = rd << imm4
0x8D  SHR    rd = rd >> imm4
0x8E  ROL    rd = rotate left imm4
0x8F  ROR    rd = rotate right imm4
```

**Register-Register (byte 1: rd[7:4] | rs[3:0])**

```
0x90  MOV    rd = rs
0x91  ADD_rr rd = rd + rs
0x92  SUB_rr rd = rd - rs
0x93  MUL_rr rd = rd * rs
0x94  CMP    flags = rd - rs (result discarded, flags set)
0x95  TEST   flags = rd & rs
0x96  CONF_RD rd = rs, conf(rd) = conf(rs)  (move with confidence)
0x97  CONF_XCH swap conf(rd) and conf(rs) values
```

**Load/Store (short offset)**

```
0x98  LD     rd = mem[rs + imm4]       (load word from register + 4-bit offset)
0x99  ST     mem[rs + imm4] = rd       (store word to register + 4-bit offset)
0x9A  LDB    rd = mem[rs + imm4]       (load byte, zero-extended)
0x9B  STB    mem[rs + imm4] = rd[7:0]  (store byte)
```

**Branch (short)**

```
0xA0  Bcond  PC += imm4 (if condition met, condition in byte 1 bits[3:0])
0xA1  BLOOP  if --loop_counter != 0: PC += imm4
```

**Confidence operations**

```
0xB0  CONF_SET4  conf(rd) = imm4       (set confidence directly, 4-bit = 0..15 scaled to 0.0..1.0)
0xB1  CONF_ADD   conf(rd) += imm4 * 0.0625  (additive confidence boost)
0xB2  CONF_DEC   conf(rd) -= imm4 * 0.0625  (confidence decay)
0xB3  CONF_MIN   conf(rd) = min(conf(rd), imm4 * 0.0625)
0xB4  CONF_MAX   conf(rd) = max(conf(rd), imm4 * 0.0625)
0xB5  CONF_MUX   rd = (conf(rd) > imm4*0.0625) ? rs : rd  (confidence-gated mux)
```

### 2.3 Three-Byte Instructions (0xC0–0xFF)

```
 ┌────────────────────────────────┐ ┌────────────────────────────────┐ ┌────────────────────────────────┐
 │ 1  1  o4 o3 o2 o1 o0          │ │ byte 1: operand               │ │ byte 2: operand               │
 │                               │ │                                │ │                                │
 │ Byte 0: prefix(11) + opcode  │ │ register pair, addr lo, etc.  │ │ register pair, addr hi, etc.  │
 └────────────────────────────────┘ └────────────────────────────────┘ └────────────────────────────────┘
```

Common byte 1/2 layouts:

```
 Register-Pair:   rd[7:4] rs[3:0]  |  — (unused, zero)
 Wide-Imm16:      imm15..imm8      |  imm7..imm0
 Address16:       addr15..addr8    |  addr7..addr0
 Reg+Addr16:      rd[7:4] _ _ _    |  addr15..addr8 | addr7..addr0
```

#### Opcode Map (3-byte space)

**Control flow**

```
0xC0  CALL    push PC+3, PC = addr16
0xC1  JMP     PC = addr16
0xC2  Jcond   if condition(flags): PC = addr16 (condition in byte 1 bits[3:0], addr in byte 1[3:0]+byte2)
0xC3  LOOP    if --loop_reg != 0: PC = addr16
```

**Load/Store (wide)**

```
0xC8  LD16    rd = mem16[addr16]         (load 16-bit from absolute address)
0xC9  ST16    mem16[addr16] = rd         (store 16-bit to absolute address)
0xCA  LDI     rd = imm16                  (load immediate 16-bit into register)
0xCB  LD_CONF rd.conf = mem16[addr16]    (load confidence from stigmergy space)
0xCC  ST_CONF mem16[addr16] = rd.conf    (store confidence to stigmergy space)
```

**Energy / ATP Operations (0xD0–0xDF)**

```
0xD0  ATP_CHECK   r0 = (r13 >= imm16) ? 1 : 0     (can we afford this?)
0xD1  ATP_SPEND   r13 -= imm16, r0 = r13           (burn energy, return remaining)
0xD2  ATP_EARN    r13 += imm16, r0 = r13           (gain energy from environment)
0xD3  ATP_BUDGET  r0 = imm16, r13 = min(r13, imm16) (set budget cap)
0xD4  ATP_DRAIN   r13 = 0                            (emergency drain)
0xD5  ATP_RESERVE r13 -= imm16 (failsafe, r0 = new r13, won't go below 0)
0xD6  ATP_QUERY   r0 = r13                           (read current energy)
0xD7  ATP_TRUST   if r14 < imm16: reject operation   (trust-gated spend)
```

**Trust Operations (0xD8–0xDF)**

```
0xD8  TRUST_SET     r14 = imm16                (set trust level directly)
0xD9  TRUST_UPDATE  r14 += imm16, clamp [0,65535]  (adjust trust)
0xDA  TRUST_DECAY   r14 = r14 >> 1             (halve trust — temporal decay)
0xDB  TRUST_BOOST   r14 = min(r14 + imm16, 65535)
0xDC  TRUST_READ    r0 = r14                   (read trust)
0xDD  TRUST_MIN     r14 = min(r14, imm16)      (cap trust)
0xDE  TRUST_VERIFY  r0 = (r14 >= imm16) ? 1 : 0 (trust threshold check)
0xDF  TRUST_RESET   r14 = 0                    (zero trust — defund agent)
```

**A2A Message Operations (0xE0–0xEF)**

```
0xE0  MSG_SEND     rd = send(addr16)           (send r1-r3 as message payload to addr)
0xE1  MSG_RECV     recv(rd) → r1,r2,r3,r14     (receive into arg regs + sender trust)
0xE2  MSG_BCAST    broadcast(r1-r3)            (broadcast to all agents in stigmergy space)
0xE3  MSG_REPLY    reply(r1-r3) to last sender (send response to r14's source)
0xE4  MSG_POLL     r0 = pending_messages()     (check message queue count)
0xE5  MSG_DEQ      dequeue next message → r1,r2,r3,r14
0xE6  MSG_TRUSTED  r0 = (r14 >= r13>>2) ? send : 0  (only send if trusted)
0xE7  MSG_CONF     attach rd.conf to outgoing message (confidence-tagged send)
```

**Instinct Operations (0xF0–0xFF)**

```
0xF0  INST_REACT    Execute hardcoded reactive behavior (stimulus-response)
0xF1  INST_FLEE     Decrement r13 by cost, move away from threat source in r1
0xF2  INST_APPROACH Decrement r13 by cost, move toward target in r1
0xF3  INST_EXPLORE  Decrement r13 by cost, random walk; r0 = discoveries
0xF4  INST_REST     Enter IDLE power state, increment r13 by regen rate
0xF5  INST_GUARD    Alert mode: high r14 sensitivity, low r13 burn rate
0xF6  INST_HERD     Align with agent state at addr16 (stigmergy read + adjust)
0xF7  INST_FORAGE   Search addr16 range for energy sources, r0 = found
0xF8  INST_COMM     Broadcast state summary to stigmergy space
0xF9  INST_LEARN    Update internal weights based on r1 (reward signal) and r14
0xFA  INST_SIGNAL   Emit signal (wired I/O or radio) with pattern from r1
0xFB  INST_LISTEN   Sample environment → r0 (sensor bus read)
0xFC  INST_ACT      r0 → actuator bus (motor/servo/I-O write)
0xFD  INST_MATE     Cooperative behavior: share r13 with addr16 agent
0xFE  INST_SURVIVE  Priority cascade: energy check → flee if low → otherwise continue
0xFF  EMERGENCY     r13 = 0, r14 = 0, r15 = HALT (total failure mode)
```

---

## 3. Register Encoding

The 4-bit register field in operand bytes encodes 16 registers:

```
  Field  | Reg | ABI Name    | Role                     | Save   | HW Special
  -------|-----|-------------|--------------------------|--------|------------
   0000  | r0  | zero/acc    | Reads 0, writes nowhere  | —      | Hardwired
   0001  | r1  | arg0/ret0   | Function arg/return #1   | caller | —
   0010  | r2  | arg1/ret1   | Function arg/return #2   | caller | —
   0011  | r3  | arg2/ret2   | Function arg/return #3   | caller | —
   0100  | r4  | sv0         | Callee-saved #1          | callee | —
   0101  | r5  | sv1         | Callee-saved #2          | callee | —
   0110  | r6  | sv2         | Callee-saved #3          | callee | —
   0111  | r7  | sv3         | Callee-saved #4          | callee | —
   1000  | r8  | tmp0        | Caller-saved #1          | caller | —
   1001  | r9  | tmp1        | Caller-saved #2          | caller | —
   1010  | r10 | tmp2        | Caller-saved #3          | caller | —
   1011  | r11 | tmp3        | Caller-saved #4          | caller | —
   1100  | r12 | conf        | Confidence value         | caller | Fused propagation
   1101  | r13 | energy/atp  | ATP energy budget        | caller | HW counter
   1110  | r14 | trust       | Sender trust level       | caller | A2A metadata
   1111  | r15 | status      | Flags & power state      | —      | HW flags
```

### r15 (Status Register) Bit Layout

```
  15  14  13  12  11  10   9   8   7   6   5   4   3   2   1   0
  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
  │   │   │   │   │   │   │   │   │ W │   │   │   │   │   │ P1│ P0│
  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
                                                      │   │   │   │
  W = Watchdog expired (r/o)  P1:P0 = Power state:     │   │   │   └── bit 0
                                                      │   │   └────── bit 1
                                                      │   └───────── (see power states)
                                                      └───────────── bit 7

  Lower 8 bits (accessible from 1-byte instructions):
  ┌───┬───┬───┬───┬───┬───┬───┬───┐
  │ Z │ N │ C │ V │ Q │   │ P1│ P0│
  └───┴───┴───┴───┴───┴───┴───┴───┘
   7   6   5   4   3   2   1   0

  Z = Zero      N = Negative    C = Carry    V = Overflow
  Q = Sticky saturation (fixed-point overflow, cleared by explicit write)
  P1:P0 = Power state (00=ACTIVE, 01=IDLE, 10=SLEEP, 11=HIBERNATE)
```

---

## 4. Confidence Fusion

### 4.1 Confidence Register (r12)

r12 holds a 16-bit confidence value representing belief in the contents of the associated data register. Encoded as Q0.16 (unsigned fixed-point): 0x0000 = no confidence, 0xFFFF = absolute certainty.

### 4.2 Bayesian Confidence Fusion

The `C*` (Confidence-Arithmetic) opcodes fuse confidence alongside the computation:

```
CADD rd, imm4:
  result = rd + imm4
  conf_out = 1 / (1/conf_rd + 1/conf_imm)

CSUB rd, imm4:
  result = rd - imm4
  conf_out = 1 / (1/conf_rd + 1/conf_imm)

CMUL rd, imm4:
  result = rd * imm4
  conf_out = min(conf_rd, conf_imm)    (multiplication — take the weaker source)

CDIV rd, imm4:
  result = rd / imm4  (safe, div-by-zero → carry set, rd unchanged)
  conf_out = 1 / (1/conf_rd + 1/conf_imm)
```

### 4.3 Confidence Lifecycle

```
                    ┌──────────┐
    Sensor read ──▶ │ CONF_SET │ conf(r12) = calibration_constant
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │  CADD    │ conf propagates through arithmetic
                    │  CSUB    │
                    │  CMUL    │
                    │  CDIV    │
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │ MSG_SEND │ confidence tags outbound message
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │ CONF_DEC │ time-based decay per cycle
                    └──────────┘
```

---

## 5. Memory Map

```
  Address      Size    Contents
  ──────────────────────────────────────────────────────────
  0x0000–0x000F  16 B   Interrupt Vector Table (8 entries × 2 B)
  0x0010–0x00FF 240 B   Stack (grows downward from 0x00FF)
  0x0100–0x07FF 1.75 KB Agent State
                          0x0100–0x01FF: Sensor state (read-only)
                          0x0200–0x02FF: Actuator state (write-only)
                          0x0300–0x07FF: Agent memory (read/write)
  0x0800–0x0FFF   2 KB   Shared Stigmergy Space (inter-agent comms)
  0x1000–0x1FFF   4 KB   Code space (up to 4096 instruction bytes)
  ──────────────────────────────────────────────────────────
  Total:            8 KB   (fits comfortably in 8 GB with room for growth)
```

### Interrupt Vector Table

```
  Offset  Vector
  ───────────────────────────────
  0x0000  Reset / Power-on
  0x0002  Timer interrupt
  0x0004  Sensor ready
  0x0006  Message received
  0x0008  Energy threshold
  0x000A  Watchdog timeout
  0x000C  Trust violation
  0x000E  Reserved
```

Each entry is a 16-bit address (little-endian) pointing into code space (0x1000–0x1FFF).

### Stack Convention

```
  0x00FF  ┌─────────┐  ← SP initialized here (stack pointer, register r7 by convention)
          │         │
          │  stack  │  grows DOWN
          │  grows  │
          │  this   │
          │  way    │
          │    ▼    │
  0x0010  ├─────────┤
          │  IVT    │  ← DO NOT overwrite
  0x0000  └─────────┘

  Stack overflow: if SP < 0x0010 → TRAP (stack overflow exception, vector 0x000E)
```

---

## 6. Power States

Controlled via r15 bits [1:0]:

```
  State      | Code | r15[1:0] | Behavior
  ─────────────────────────────────────────────────────────
  ACTIVE     |  0   | 00       | Full execution, all cores
  IDLE       |  1   | 01       | Clock gated, interrupt-wake
  SLEEP      |  2   | 10       | Core power-down, SRAM retained
  HIBERNATE  |  3   | 11       | Full power-down, wake via external pin only
```

**Watchdog Timer (r15 bit 7):**
- Set by hardware when counter expires (configurable, default: 1024 cycles)
- If set: processor auto-HALTs, W bit remains 1
- Software must execute `WDOG_RESET` (opcode 0x2A) before expiry
- WDOG_RESET clears bit 7 and reloads the counter

---

## 7. Edge-Specific Constraints

| Constraint | Rule |
|-----------|------|
| **No dynamic allocation** | All memory is static. Stack only. No `malloc`. |
| **No floating point** | All arithmetic is fixed-point Q16.16 (16 int bits, 16 frac bits). Range: ±32767.99998. |
| **Fixed-point multiply** | `a * b` → `(a * b) >> 16` in hardware. Saturation flag Q set on overflow. |
| **No division by zero** | DIV/CDIV: if divisor == 0, destination unchanged, carry flag set. |
| **Stack limit** | 240 bytes maximum. SP below 0x0010 traps. |
| **Deterministic** | All instructions complete in bounded time. No caches to flush. |
| **Bare metal** | No OS. Interrupts, power, watchdog are hardware-managed. |

---

## 8. Cloud ↔ Edge Opcode Mapping

The assembler accepts `--target=edge` or `--target=cloud`. Semantic opcodes map across targets:

```
  Semantic          | Cloud ISA v2              | Edge ISA v3
  ──────────────────┼───────────────────────────┼──────────────
  ADD rd, imm       | 0x10 (Format A, 4-byte)   | 0x84 (2-byte)
  SUB rd, imm       | 0x11 (Format A, 4-byte)   | 0x85 (2-byte)
  MUL rd, imm       | 0x12 (Format A, 4-byte)   | 0x86 (2-byte)
  DIV rd, imm       | 0x13 (Format A, 4-byte)   | 0x87 (2-byte)
  MOV rd, rs        | 0x20 (Format B, 4-byte)   | 0x90 (2-byte)
  CMP rd, rs        | 0x30 (Format B, 4-byte)   | 0x94 (2-byte)
  NOP               | 0x00 (4-byte)             | 0x00 (1-byte)
  HALT              | 0x01 (4-byte)             | 0x20 (1-byte)
  RET               | 0x02 (4-byte)             | 0x21 (1-byte)
  CALL addr16       | 0x40 (Format C, 4-byte)   | 0xC0 (3-byte)
  JMP addr16        | 0x41 (Format C, 4-byte)   | 0xC1 (3-byte)
  LD rd, [addr]     | 0x50 (Format D, 4-byte)   | 0xC8 (3-byte)
  ST [addr], rd     | 0x51 (Format D, 4-byte)   | 0xC9 (3-byte)
  ADD_CONF (fused)  | 0x10 + Format E flag      | 0x80 (2-byte CADD)
  SUB_CONF (fused)  | 0x11 + Format E flag      | 0x81 (2-byte CSUB)
  MUL_CONF (fused)  | 0x12 + Format E flag      | 0x82 (2-byte CMUL)
  DIV_CONF (fused)  | 0x13 + Format E flag      | 0x83 (2-byte CDIV)
```

**Key difference:** Cloud uses a Format E flag bit to enable confidence fusion on any arithmetic opcode. Edge uses dedicated `C*` opcodes — this eliminates the flag byte and saves space at the cost of fewer fusion-capable operations.

---

## 9. Example Programs

### 9.1 Hello World: Sensor Read → Trust-Weighted Broadcast

```asm
; Read temperature sensor, check trust, broadcast if trusted
    INST_LISTEN          ; 0xFB — r0 = sensor reading
    CONF_SET4  r0, 0xF   ; 0xB0 0x0F — set confidence high (calibrated sensor)
    TRUST_QUERY          ; 0x41 — r0 = trust of last msg sender (or self-trust baseline)
    CMP  r0, r1          ; 0x94 0x01 — compare trust to threshold in r1
    Bcond NZ, skip       ; 0xA0 0x9F — if trust low, skip broadcast (cond=NE, imm=0xF→offset)
    MSG_SEND  r0, 0x0800 ; 0xE0 [r0=reg] [0x08] [0x00] — broadcast to stigmergy base
skip:
    INST_REST            ; 0xF4 — idle, regenerate energy
    HALT                 ; 0x20 — done
```

**Byte sequence:** `FB B0 0F 41 94 01 A0 9F E0 00 08 00 F4 20` = **14 bytes**

### 9.2 Loop with Confidence Decay

```asm
; Process 5 samples, decaying confidence each iteration
    LDI    r8, 5         ; 0xCA 0x08 0x00 0x05 — loop counter = 5
    CONF_SET4 r0, 0xF    ; 0xB0 0x0F — start with full confidence
loop:
    INST_LISTEN          ; 0xFB — r0 = sensor reading
    CADD   r0, 0x01      ; 0x80 0x01 — add bias, fuse confidence
    CONF_DEC r0, 0x02    ; 0xB2 0x02 — decay confidence (2 * 0.0625 = 0.125 per loop)
    DEC    r8             ; 0x0A — r8--
    Bcond  NZ, loop      ; 0xA0 0x9F — if r8 != 0, loop back (offset = -6 → imm4 wraps)
    HALT                  ; 0x20
```

**Byte sequence:** `CA 08 00 05 B0 0F FB 80 01 B2 02 0A A0 9F 20` = **15 bytes**

### 9.3 Energy-Gated Task Execution

```asm
; Only execute task if we have enough energy; otherwise rest
    ATP_QUERY             ; 0xD6 (3-byte: D6 00 00) — r0 = current energy
    LDI    r1, 100        ; 0xCA 0x01 0x00 0x64 — task costs 100 ATP
    CMP    r0, r1         ; 0x94 0x01 — compare energy to cost
    Bcond  LT, low_energy ; 0xA0 0x7D — if r0 < r1, jump to rest
    ATP_SPEND r0, 100     ; 0xD1 0x00 0x00 0x64 — burn energy
    ; ... task code here ...
    JMP    done           ; 0xC1 0x00 0x00 0xNN
low_energy:
    INST_REST             ; 0xF4 — regenerate
done:
    HALT                  ; 0x20
```

### 9.4 A2A Message Send/Receive

```asm
; Agent 1: Send discovery message with confidence tag
    INST_LISTEN          ; 0xFB — r0 = what we found
    CONF_SET4 r0, 0xA    ; 0xB0 0x0A — moderate confidence
    MOV    r1, r0        ; 0x90 0x01 — payload arg0 = discovery
    LDI    r2, 0x1234    ; 0xCA 0x02 0x12 0x34 — payload arg1 = location
    MSG_SEND r1, 0x0800  ; 0xE0 0x10 0x08 0x00 — send to stigmergy space
    HALT                 ; 0x20

; Agent 2: Receive and trust-check
    MSG_POLL             ; 0xE4 (3-byte) — r0 = pending messages
    Bcond  Z, done       ; 0xA0 0x0F — nothing? bail
    MSG_RECV r0          ; 0xE1 0x00 0x00 0x00 — dequeue → r1,r2,r3,r14
    TRUST_VERIFY r14, 50 ; 0xDE 0xE0 0x00 0x32 — r0 = (trust >= 50)?
    Bcond  Z, done       ; 0xA0 0x0F — not trusted, discard
    ; process trusted message ...
done:
    HALT                 ; 0x20
```

---

## 10. Summary Statistics

| Metric | Value |
|--------|-------|
| Total opcode space | 256 (128 + 64 + 64) |
| Currently assigned | ~100 (room for expansion) |
| Smallest instruction | 1 byte (NOP, HALT, RET) |
| Largest instruction | 3 bytes (CALL, MSG_SEND) |
| Average (typical program) | ~1.7 bytes/instruction |
| Code density vs cloud | **~2.3×** denser (4-byte fixed vs variable-width) |
| Memory footprint | 8 KB total address space |
| Registers | 16 (4 ABI, 4 callee, 4 caller, 4 special) |
| Confidence opcodes | 12 dedicated (C* series + CONF_* series) |
| A2A opcodes | 8 |
| Instinct opcodes | 16 |

---

## 11. Open Questions for Oracle1

1. **Format E ↔ C* opcode mapping**: Cloud uses a single flag bit to enable confidence fusion on any arithmetic opcode. Edge dedicates separate opcodes. Should the assembler `--target=cloud` automatically expand `CADD` into `ADD + FormatE`, or should they remain separate mnemonics?

2. **Stigmergy space coherence**: The spec assumes single-writer stigmergy (each agent owns a region). Do we need atomic read-modify-write primitives (e.g., CONF_XCHG) for shared stigmergy cells, or is best-effort sufficient for edge?

3. **Instinct opcode programmability**: Currently instincts are hardcoded behaviors. Should we allow instinct vectors to be patched at runtime (write to IVT), or keep them ROM-locked for safety?

4. **16-bit address space ceiling**: 64 KB total. Current spec uses 8 KB. Is 64 KB sufficient for the target class, or should we plan for a future 24-bit address extension prefix (e.g., `0b11_1111xx` → 5-byte wide-address instruction)?

---

*End of specification. Ready for Oracle1 review.*