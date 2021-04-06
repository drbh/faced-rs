# faced-rs

A Rust port of https://github.com/iitzco/faced and "🚀 😏 CPU (Near) Real Time face detection"

```bash
git clone https://github.com/drbh/faced-rs.git
```

```
TF_CPP_MIN_LOG_LEVEL=2 cargo run -- images/people.jpg
```

```bash
cargo build --release
./target/release/faced-rs images/people.jpg
```

### Compare to original

```bash
./bin/faced --input ../faced-rs/images/people.jpg
# [
#    [
#       2373,
#       1150,
#       481,
#       614,
#       0.95099664
#    ],
#    [
#       3860,
#       1219,
#       512,
#       546,
#       0.8763757
#    ],
#    [
#       5678,
#       3765,
#       467,
#       643,
#       0.88071346
#    ]
# ]
```

```bash
TF_CPP_MIN_LOG_LEVEL=2 cargo run -- images/people.jpg
# [
#    [
#       2377.0,
#       1167.0,
#       478.0,
#       624.0,
#       0.94749
#    ],
#    [
#       3868.0,
#       1199.0,
#       499.0,
#       517.0,
#       0.87994
#    ],
#    [
#       5660.0,
#       3755.0,
#       471.0,
#       642.0,
#       0.87556
#    ]
# ]
```
