extern crate image;
extern crate tensorflow;

use image::{imageops, GenericImageView};
use std::fs::File;
use std::io::Read;
use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;
use tensorflow::Operation;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Tensor;

mod utils;
use crate::facedrs::utils::iou;
// use crate::utils::iou;

// set some constants (for tensor size and etc)
const YOLO_SIZE: u64 = 288;
const CORRECTOR_SIZE: u64 = 50;
const YOLO_TARGET: i32 = 9;
const MARGIN: f32 = 0.5;
const FLT_TO_INT_SCALAR: f32 = 100_000.0;

// the main app struct
#[derive(Debug)]
pub struct FaceDetector {
    graph: Graph,
    corrector: FaceCorrector,
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

// the second - correction model struct
#[derive(Debug)]
pub struct FaceCorrector {
    graph: Graph,
    sess: Session,
    img: Operation,
    training: Operation,
    X: Operation,
    Y: Operation,
    W: Operation,
    H: Operation,
}

impl FaceDetector {
    pub fn init() -> FaceDetector {
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

            corrector: FaceCorrector::init(),

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

    // pub fn predict(&self, img: &str) -> Vec<(f32, f32, f32, f32, f32)> {
    pub fn predict(&self, img: &str) -> Vec<Vec<f32>> {
        // pub fn predict(&self, img: &str) -> f32 {
        // RBG effected by filter on resize - does not match CV2 perfectly
        let raw_im = image::open(img).unwrap();

        let im = raw_im.resize_exact(
            YOLO_SIZE as u32,
            YOLO_SIZE as u32,
            image::FilterType::Nearest,
        );
        let rgb = im.to_rgb();
        // let mut counter = 0;

        let mut values: Vec<f32> = vec![];

        for _rb in rgb.pixels() {
            let r = _rb[0] as f32 / 255.0;
            let g = _rb[1] as f32 / 255.0;
            let b = _rb[2] as f32 / 255.0;
            values.push(r);
            values.push(g);
            values.push(b);
            // counter += 1;
        }

        let tensor_train: Tensor<f32> = <Tensor<f32>>::new(&[1, YOLO_SIZE, YOLO_SIZE, 3])
            .with_values(&values)
            .unwrap();

        let tensor_training = <Tensor<bool>>::new(&[]);
        let mut step = SessionRunArgs::new();

        step.add_feed(&self.img, 0, &tensor_train);
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

        let absolute_bboxes = self._absolute_bboxes(
            prob_output_tensor,
            x_center_output_tensor,
            y_center_output_tensor,
            w_output_tensor,
            h_output_tensor,
            raw_im.clone(),
            0.83,
        );
        let corrected_bboxes = self._correct(raw_im.clone(), absolute_bboxes);
        let non_suppressed_boxes = self._nonmax_supression(corrected_bboxes.clone(), 0.2);
        non_suppressed_boxes
    }

    pub fn _correct(
        &self,
        mut raw_im: image::DynamicImage,
        scaled_bboxes: Vec<Vec<i32>>,
    ) -> Vec<Vec<f32>> {
        let mut results = vec![];
        // let mut counter = 0;
        for _box in scaled_bboxes {
            let xmin = (_box[0] as f32 - _box[2] as f32 / 2.0) - MARGIN * _box[2] as f32;
            let xmax = (_box[0] as f32 + _box[2] as f32 / 2.0) + MARGIN * _box[2] as f32;

            let ymin = (_box[1] as f32 - _box[3] as f32 / 2.0) - MARGIN * _box[3] as f32;
            let ymax = (_box[1] as f32 + _box[3] as f32 / 2.0) + MARGIN * _box[3] as f32;

            // get face and pass it to the correcter
            let sub_image = imageops::crop(
                &mut raw_im,
                xmin as u32,
                ymin as u32,
                (xmax - xmin) as u32,
                (ymax - ymin) as u32,
            );

            let (_x, _y, _w, _h) = self.corrector.predict(sub_image);

            results.push(vec![
                _x as f32 + xmin,
                _y as f32 + ymin,
                _w as f32,
                _h as f32,
                _box[4] as f32 / FLT_TO_INT_SCALAR,
            ]);
        }
        results
    }

    pub fn _absolute_bboxes(
        &self,
        prob_output_tensor: tensorflow::Tensor<f32>,
        x_center_output_tensor: tensorflow::Tensor<f32>,
        y_center_output_tensor: tensorflow::Tensor<f32>,
        w_output_tensor: tensorflow::Tensor<f32>,
        h_output_tensor: tensorflow::Tensor<f32>,
        raw_im: image::DynamicImage,
        thresh: f32,
    ) -> Vec<Vec<i32>> {
        // ) -> f32 {
        let mut _bboxes = vec![];
        for n in 1..prob_output_tensor.len() {
            let g = prob_output_tensor[n];
            if g > thresh {
                let row = ((n / 9) as f64).floor() as f32;
                let col = (n % 9) as f32;

                let bbox = vec![
                    row + x_center_output_tensor[n],
                    col + y_center_output_tensor[n],
                    w_output_tensor[n],
                    h_output_tensor[n],
                    g,
                ];
                _bboxes.push(bbox);
            }
        }

        let img_w = raw_im.width() as f32;
        let img_h = raw_im.height() as f32;
        // println!("{} {}", img_w, img_h);
        let mut scaled_bboxes = vec![];
        for _box in _bboxes {
            let new_box = vec![
                ((_box[0] / YOLO_TARGET as f32) * img_w).floor() as i32,
                ((_box[1] / YOLO_TARGET as f32) * img_h).floor() as i32,
                (_box[2] * img_w).floor() as i32,
                (_box[3] * img_h).floor() as i32,
                (_box[4] * FLT_TO_INT_SCALAR) as i32,
            ];

            scaled_bboxes.push(new_box);
        }
        scaled_bboxes
    }

    pub fn _nonmax_supression(&self, bboxes: Vec<Vec<f32>>, thresh: f32) -> Vec<Vec<f32>> {
        let NONE_VALUE = 0;
        let SUPPRESSED = 1;
        let NON_SUPPRESSED = 2;

        let N = bboxes.len();

        let mut status = vec![NONE_VALUE; N];
        for i in 0..N {
            // unimplemented!();
            if status[i] != NONE_VALUE {
                continue;
            }

            let mut curr_max_p = bboxes[i][bboxes[i].len() - 1];
            let mut curr_max_index = i;

            for j in i + 1..N {
                if status[j] != NONE_VALUE {
                    continue;
                }
                let metric = iou(bboxes[i].clone(), bboxes[j].clone());

                if metric > thresh {
                    if bboxes[j][bboxes[j].len() - 1] > curr_max_p {
                        status[curr_max_index] = SUPPRESSED;
                        curr_max_p = bboxes[j][bboxes[j].len() - 1];
                        curr_max_index = j;
                    } else {
                        status[j] = SUPPRESSED;
                    }
                }
            }
            status[curr_max_index] = NON_SUPPRESSED
        }

        let mut non_suppressed_boxes = vec![];
        for i in 0..N {
            if status[i] == NON_SUPPRESSED {
                non_suppressed_boxes.push(bboxes[i].clone());
            }
        }
        non_suppressed_boxes
    }
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

    fn predict(&self, frame: image::SubImage<&mut image::DynamicImage>) -> (i32, i32, i32, i32) {
        let og = frame.to_image();

        let img_w = og.width() as f32;
        let img_h = og.height() as f32;

        let rgb = imageops::resize(
            &og,
            CORRECTOR_SIZE as u32,
            CORRECTOR_SIZE as u32,
            image::FilterType::Nearest,
        );

        // let mut counter = 0;
        let mut values: Vec<f32> = vec![];
        for _rb in rgb.pixels() {
            let r = _rb[0] as f32 / 255.0;
            let g = _rb[1] as f32 / 255.0;
            let b = _rb[2] as f32 / 255.0;
            values.push(r);
            values.push(g);
            values.push(b);
            // counter += 1;
        }

        let empty_img: Tensor<f32> = <Tensor<f32>>::new(&[1, CORRECTOR_SIZE, CORRECTOR_SIZE, 3])
            .with_values(&values)
            .unwrap();

        let tensor_training = <Tensor<bool>>::new(&[]);

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

        // resize the outputs
        let _x = (X_output_tensor[0] * img_w).floor() as i32;
        let _y = (Y_output_tensor[0] * img_h).floor() as i32;
        let _w = (W_output_tensor[0] * img_w).floor() as i32;
        let _h = (H_output_tensor[0] * img_h).floor() as i32;

        // these are centers and sizes
        (_x, _y, _w, _h)
    }
}
