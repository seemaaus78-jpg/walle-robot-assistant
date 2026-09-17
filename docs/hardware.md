# Hardware

## Bill of materials

| Part | Role | Notes |
|---|---|---|
| Radxa Cubie A7Z | Compute | 1 GB or 2 GB LPDDR4. The 2 GB variant is strongly preferred — see the RAM budget in [architecture.md](architecture.md). |
| 2×20 male pin header, 2.54 mm | GPIO header | **Check your board.** The A7Z ships with the 40-pin header soldered or unpopulated depending on SKU ([CNX Software](https://www.cnx-software.com/2025/08/25/pi-zero-sized-radxa-cubie-a7z-sbc-features-allwinner-a733-cortex-a76-a55-soc-up-to-16gb-ram-wifi-6/)). On an unpopulated board it is 40 bare through-holes and nothing can be plugged in until a header is soldered on. |
| Soldering iron + 0.8 mm rosin-core solder | Assembly | Needed for the header above and for the copper/brass frame. Not optional for this build. |
| MicroSD 64 GB | OS, models, city database | A1/A2 rated. Cheap cards are the single most common cause of a robot that boots slowly and stutters mid-sentence. |
| INMP441 | I²S MEMS microphone | Digital output, no analogue noise pickup. |
| MAX98357A | I²S class-D amplifier | 3.2 W into 4 Ω, ~2 W into 8 Ω. |
| 8 Ω 2 W speaker | Audio out | |
| 4 × SG90 | Neck pan/tilt, both arms | ~650 mA stall each. |
| 3.7 V 6000 mAh LiPo | Battery | 22.2 Wh nominal. |
| TP4056 **with protection** | Charging | Must be the DW01+FS8205 protected variant. See the warning below. |
| 5 V 3 A boost converter | Rail | Two of them is better than one — see "Power". |

## The 40-pin header

The Cubie A7Z is sold both with the 40-pin GPIO header soldered on and with it left as
bare through-holes. Both variants have the identical 40-position pinout; the unpopulated
one simply has nothing to plug into.

If yours is unpopulated, solder a standard 2×20 male header (2.54 mm pitch) into it before
wiring anything. Everything else in this document — the display on SPI1, the I²S audio on
pins 12/35/38/40, the drive GPIOs — assumes the header is fitted.

Verify the pitch before ordering: ten holes along a row should span 25.4 mm. Radxa
describes the connector as a standard 40-pin GPIO compatible with common accessories,
which implies 2.54 mm, but this has not been measured on hardware here.

Soldering the eight display wires directly into the holes also works and is what the
reference build does, but it makes every mistake a desoldering job. Prefer the header.

## Wiring

```
                    [ 3.7V 6000mAh LiPo ]
                              |
                    [ TP4056 (protected) ]
                              |
              +---------------+---------------+
              |                               |
   [ 5V boost - logic ]              [ 5V boost - servos ]
              |                               |
     [ Radxa Cubie A7Z ]              [ 4x SG90 V+ ]
              |                               |
              |   GPIO signal x4              |
              +-------------------------------+
              |
              |   I2S bus (shared clocks)
              +---> BCLK  --+--> INMP441 SCK
              |             +--> MAX98357A BCLK
              +---> LRCLK --+--> INMP441 WS
              |             +--> MAX98357A LRC
              +---> SDIN  <----- INMP441 SD      (capture)
              +---> SDOUT ------> MAX98357A DIN  (playback)
                                        |
                                  [ 8 ohm speaker ]

              All grounds tied together at one point.
```

### I²S is one bus, not two

The microphone and the amplifier share `BCLK` and `LRCLK`. Only the data lines
are separate: the INMP441 drives the SoC's input line, the SoC drives the
amplifier's input line. This is normal full-duplex I²S and it is why both
devices can run at once from a single peripheral.

Two pin-strap details that are easy to miss and produce silence rather than an
error:

- **INMP441 `L/R`** selects which half of the frame the microphone speaks in.
  Tie it to GND for the left channel and capture the left channel, or you get a
  stream of zeros.
- **MAX98357A `SD`** is a mode pin, not a shutdown pin, despite the name. Left
  floating the amplifier is off. The resistor to GND selects left, right, or
  (left+right)/2. For a mono robot, use the mono setting.

The Radxa's I²S peripheral must be enabled in the device tree before either
device appears. See [setup.md](setup.md).

## Servo signalling

SG90s expect a 50 Hz pulse train: 1000 µs is roughly 0°, 1500 µs centre, 2000 µs
roughly 180°, with usable travel typically between 500 µs and 2500 µs. They do
**not** respond to a line simply held high.

Two caveats worth designing around:

1. **3.3 V logic into a 5 V servo is marginal.** An SG90's input threshold is
   specified against its own supply. Many units latch onto 3.3 V pulses fine;
   some twitch, and some ignore them entirely. If yours is unreliable, put a
   level shifter or a 74AHCT125 buffer on the four signal lines. This is the
   first thing to check when a servo works on the bench at 3.3 V and misbehaves
   in the robot.
2. **Software PWM jitters.** `walle/motion.py` generates the pulse train from a
   Python thread. Linux scheduling noise of a few hundred microseconds is
   normal, which reads as a slight tremble in the horn. That is acceptable for
   gestures and not acceptable for a steady hold. If you want steady holds, put
   a **PCA9685** on the I²C bus and drive all four servos from its hardware
   timers; the module's `LineBackend` protocol is the seam to implement it
   against.

The code detaches a servo (stops pulsing) `hold_s` seconds after it reaches a
pose. That is deliberate: a continuously pulsed SG90 buzzes, heats up, and holds
its position by actively fighting gravity, which is the largest avoidable draw
on the battery.

## Power

### Give the servos their own rail

Four SG90s can momentarily pull ~2.6 A between them at stall. Sharing one boost
converter with the board means every arm movement dips the logic rail, and an
SBC that browns out mid-write corrupts the SD card. Use a second boost converter
for the servo rail, tie the grounds together at a single point, and put a
1000 µF electrolytic across the servo rail close to the servos to absorb the
inrush.

### Charging while running needs a load-sharing board

A bare TP4056 has one battery terminal and no load-sharing circuitry. Wiring the
robot to `BAT+` while it charges means the charger sees battery current *plus*
load current, so it never sees the taper current that tells it to terminate. In
practice this ranges from "the charge LED never goes green" to overcharging a
LiPo, which is a fire risk rather than an inconvenience.

If you want the robot to run while charging, use a TP4056 module that
explicitly advertises **load sharing** (an ideal-diode or P-MOSFET path from
`OUT+`), or an integrated PMIC. Otherwise treat charging and running as separate
states.

Separately: buy the TP4056 variant carrying a **DW01 + FS8205 protection pair**.
The bare charger has no over-discharge or short-circuit protection, and a 6000
mAh cell taken below ~2.5 V is damaged and can vent on the next charge.

### Runtime

22.2 Wh nominal, less boost-converter losses (85–90% is typical for these
modules) gives roughly 19–20 Wh at the 5 V rail.

The design target of **4.5–5.5 hours** is reachable, but only while the servos
are idle nearly all the time — which is what the detach-after-move behaviour in
`walle/motion.py` is for. Continuous gesturing pulls that figure down sharply.
Treat 4.5–5.5 h as "listening and answering", not as a duty-cycle-independent
number.

I have not measured any of this. Put a USB power meter on the 5 V rail and
record your own figures before trusting the estimate.
