extern crate tensorflow;

use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process::exit;
use std::result::Result;
use tensorflow::Code;
use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;
use tensorflow::Output;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Status;
use tensorflow::Tensor;

const YOLO_SIZE: i32 = 288;
const CORRECTOR_SIZE: i32 = 50;
const YOLO_TARGET: i32 = 9;

extern crate image;

use image::GenericImageView;

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

    let mut my_runtime = SessionRunArgs::new();

    let img = Output {
        operation: graph.operation_by_name("img").unwrap().unwrap(),
        index: 0,
    };
    let training = Output {
        operation: graph.operation_by_name("training").unwrap().unwrap(),
        index: 0,
    };
    let X = Output {
        operation: graph.operation_by_name("X").unwrap().unwrap(),
        index: 0,
    };
    let Y = Output {
        operation: graph.operation_by_name("Y").unwrap().unwrap(),
        index: 0,
    };
    let W = Output {
        operation: graph.operation_by_name("W").unwrap().unwrap(),
        index: 0,
    };
    let H = Output {
        operation: graph.operation_by_name("H").unwrap().unwrap(),
        index: 0,
    };
    // let w_ix = my_runtime.request_fetch(&W, 0);
    // session.run(&mut my_runtime).unwrap();

    // let wd: Result<Tensor<f32>, Status> = my_runtime.fetch(w_ix);
    // let wd: Vec<Tensor<f32>> = H.get_attr_tensor_list("").unwrap();

    // session.run(&mut my_runtime);
    // println!("{:#?}", img.num_inputs());

    // for a in graph.operation_iter() {
    //     // unimplemented!();
    //     println!("{:#?}", a.name());
    // }

    let image = image::open("./images/phelps.jpg").unwrap();

    // let david: Tensor<f32> = img.get_attr_tensor("shape").unwrap();

    println!("{:#?}", W);
    // println!("{:#?}", W.control_outputs());
}

// fn predict(self, frame) {
//     // Preprocess
//     input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
//     input_img = cv2.resize(input_img, (CORRECTOR_SIZE, CORRECTOR_SIZE)) / 255.
//     input_img = np.reshape(input_img, [1, CORRECTOR_SIZE, CORRECTOR_SIZE, 3])

//     x, y, w, h = self.sess.run([self.x, self.y, self.w, self.h], feed_dict={self.training: False, self.img: input_img})

//     img_h, img_w, _ = frame.shape

//     x = int(x*img_w)
//     w = int(w*img_w)

//     y = int(y*img_h)
//     h = int(h*img_h)

//     return x, y, w, h
// }
