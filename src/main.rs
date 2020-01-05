#![allow(non_snake_case, unused_variables)]

extern crate image;
extern crate tensorflow;
use std::fs::File;
use std::io::Read;
use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Tensor;

// const YOLO_SIZE: i32 = 288;
// const CORRECTOR_SIZE: i32 = 50;
// const YOLO_TARGET: i32 = 9;

fn main() {
    let filename = "./src/models/face_corrector.pb";
    let mut graph = Graph::new();
    let mut proto = Vec::new();
    File::open(filename)
        .unwrap()
        .read_to_end(&mut proto)
        .unwrap();
    graph
        .import_graph_def(&proto, &ImportGraphDefOptions::new())
        .unwrap();
    let session = Session::new(&SessionOptions::new(), &graph).unwrap();

    // pull operations
    let img = graph.operation_by_name("img").unwrap().unwrap();
    let training = graph.operation_by_name("training").unwrap().unwrap();
    let X = graph.operation_by_name("X").unwrap().unwrap();
    let Y = graph.operation_by_name("Y").unwrap().unwrap();
    let W = graph.operation_by_name("W").unwrap().unwrap();
    let H = graph.operation_by_name("H").unwrap().unwrap();

    let tensor_training = <Tensor<bool>>::new(&[1]).with_values(&[false]).unwrap();
    let tensor_img = <Tensor<f32>>::new(&[1]).with_values(&[0.0]).unwrap();

    // read image
    // let image = image::open("./images/phelps.jpg").unwrap();
    let empty_img = <Tensor<f32>>::new(&[1, 50, 50, 3]);
    let mut step = SessionRunArgs::new();

    step.add_feed(&img, 0, &empty_img);
    step.add_feed(&training, 0, &tensor_training);

    let X_output_token = step.request_fetch(&X, 0);
    let Y_output_token = step.request_fetch(&Y, 0);
    let W_output_token = step.request_fetch(&W, 0);
    let H_output_token = step.request_fetch(&H, 0);

    session.run(&mut step).unwrap();

    println!("{:#?}", X_output_token);
}
