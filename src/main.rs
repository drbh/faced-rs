#![allow(non_snake_case, unused_variables)]

extern crate image;
extern crate tensorflow;
use std::fs::File;
use std::io::Read;
use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;
use tensorflow::Operation;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Tensor;

const YOLO_SIZE: u64 = 288;
const CORRECTOR_SIZE: u64 = 50;
// const YOLO_TARGET: i32 = 9;

#[derive(Debug)]
struct FaceDetector {
    graph: Graph,
    sess: Session,
    prob: Operation,
    img: Operation,
    training: Operation,
    x_center: Operation,
    y_center: Operation,
    w: Operation,
    h: Operation,

    cols: i32,
    rows: i32,
}

impl FaceDetector {
    fn init() -> FaceDetector {
        let filename = "./src/models/face_yolo.pb";
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
        let prob = graph.operation_by_name("prob").unwrap().unwrap();
        let x_center = graph.operation_by_name("x_center").unwrap().unwrap();
        let y_center = graph.operation_by_name("y_center").unwrap().unwrap();
        let w = graph.operation_by_name("w").unwrap().unwrap();
        let h = graph.operation_by_name("h").unwrap().unwrap();

        // fn load_model(x: f64, y: f64) -> (f64,f64,f64,f64) {
        FaceDetector {
            graph: graph,
            sess: session,

            img: img,
            training: training,
            prob: prob,

            x_center: x_center,
            y_center: y_center,

            w: w,
            h: h,

            cols: 0,
            rows: 0,
        }
    }

    fn predict(&self) -> (f64, f64, f64, f64) {
        let tensor_training = <Tensor<bool>>::new(&[]);
        let empty_img = <Tensor<f32>>::new(&[1, YOLO_SIZE, YOLO_SIZE, 3]);

        let mut step = SessionRunArgs::new();

        step.add_feed(&self.img, 0, &empty_img);
        step.add_feed(&self.training, 0, &tensor_training);

        let prob_output_token = step.request_fetch(&self.prob, 0);
        let x_center_output_token = step.request_fetch(&self.x_center, 0);
        let y_center_output_token = step.request_fetch(&self.y_center, 0);
        let w_output_token = step.request_fetch(&self.w, 0);
        let h_output_token = step.request_fetch(&self.h, 0);

        self.sess.run(&mut step).unwrap();

        let prob_output_tensor = step.fetch::<f32>(prob_output_token).unwrap();
        let x_center_output_tensor = step.fetch::<f32>(x_center_output_token).unwrap();
        let y_center_output_tensor = step.fetch::<f32>(y_center_output_token).unwrap();
        let w_output_tensor = step.fetch::<f32>(w_output_token).unwrap();
        let h_output_tensor = step.fetch::<f32>(h_output_token).unwrap();

        // img_h, img_w, _ = frame.shape
        // resize the outputs

        println!("{:#?}", prob_output_tensor);
        println!("{:#?}", x_center_output_tensor);
        println!("{:#?}", y_center_output_tensor);
        println!("{:#?}", w_output_tensor);
        println!("{:#?}", h_output_tensor);

        // bboxes = self._absolute_bboxes(pred, frame, thresh)
        // bboxes = self._correct(frame, bboxes)
        // bboxes = self._nonmax_supression(bboxes)

        (0.0, 0.0, 0.0, 0.0)
    }
}

#[derive(Debug)]
struct FaceCorrector {
    graph: Graph,
    sess: Session,
    img: Operation,
    training: Operation,
    X: Operation,
    Y: Operation,
    W: Operation,
    H: Operation,
}

impl FaceCorrector {
    fn init() -> FaceCorrector {
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

        // fn load_model(x: f64, y: f64) -> (f64,f64,f64,f64) {
        FaceCorrector {
            graph: graph,
            sess: session,
            img: img,
            training: training,
            X: X,
            Y: Y,
            W: W,
            H: H,
        }
    }

    fn predict(&self) -> (f64, f64, f64, f64) {
        let tensor_training = <Tensor<bool>>::new(&[]);
        let empty_img = <Tensor<f32>>::new(&[1, CORRECTOR_SIZE, CORRECTOR_SIZE, 3]);

        let mut step = SessionRunArgs::new();

        step.add_feed(&self.img, 0, &empty_img);
        step.add_feed(&self.training, 0, &tensor_training);

        let X_output_token = step.request_fetch(&self.X, 0);
        let Y_output_token = step.request_fetch(&self.Y, 0);
        let W_output_token = step.request_fetch(&self.W, 0);
        let H_output_token = step.request_fetch(&self.H, 0);

        self.sess.run(&mut step).unwrap();

        let X_output_tensor = step.fetch::<f32>(X_output_token).unwrap();
        let Y_output_tensor = step.fetch::<f32>(Y_output_token).unwrap();
        let W_output_tensor = step.fetch::<f32>(W_output_token).unwrap();
        let H_output_tensor = step.fetch::<f32>(H_output_token).unwrap();

        // img_h, img_w, _ = frame.shape
        // resize the outputs

        println!("{:#?}", X_output_tensor);
        println!("{:#?}", Y_output_tensor);
        println!("{:#?}", W_output_tensor);
        println!("{:#?}", H_output_tensor);
        (0.0, 0.0, 0.0, 0.0)
    }
}

fn main() {
    let d = FaceDetector::init();
    let fc = FaceCorrector::init();
    fc.predict();
}
