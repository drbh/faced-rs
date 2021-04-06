#![allow(non_snake_case)]
mod facedrs;

use facedrs::FaceDetector;
use std::env;

use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let d = FaceDetector::init();
    let start = Instant::now();
    let results = d.predict(&args[1]);
    let elapsed = start.elapsed();
    println!("Milis elapsed - {:?}", elapsed.as_millis());
    println!("{:?}", results);
}
