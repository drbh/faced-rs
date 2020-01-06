#![allow(non_snake_case)]
mod facedrs;

use facedrs::FaceDetector;

fn main() {
    let d = FaceDetector::init();
    let mut results = d.predict("./images/people.jpg");
    println!("{:#?}", results);
    results = d.predict("./images/phelps.jpg");
    println!("{:#?}", results);
}
