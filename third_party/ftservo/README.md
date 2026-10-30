# FTServo serial read patch

`bounded-serial-read.patch` applies to FTServo_Linux commit
`064a6dbf814477b970959b5b01b2fc9bc594afa3`.

Both read overloads share one implementation. EOF, read errors, and select
errors return the bytes already received; they never decrement the buffer
position or spin on a disconnected descriptor. Each select call receives fresh
arguments, and a steady-clock deadline bounds the entire read, including partial
responses. Invalid descriptors and nonpositive lengths return zero.

The upstream revision and MIT license are unchanged.

Regression coverage lives in `common/so101_driver/tests/test_ftservo_serial.cc`.
