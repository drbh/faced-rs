// pub fn iou(bbox1: Vec<(f32, f32, f32, f32, f32)>, bbox2: Vec<(f32, f32, f32, f32, f32)>) -> f32 {
pub fn iou(bbox1: Vec<f32>, bbox2: Vec<f32>) -> f32 {
    // determine the (x, y)-coordinates of the intersection rectangle
    let boxA = (
        bbox1[0] - bbox1[2] / 2.0 as f32,
        bbox1[1] - bbox1[3] / 2.0 as f32,
        bbox1[0] + bbox1[2] / 2.0 as f32,
        bbox1[1] + bbox1[3] / 2.0 as f32,
    );

    let boxB = (
        bbox2[0] - bbox2[2] / 2.0 as f32,
        bbox2[1] - bbox2[3] / 2.0 as f32,
        bbox2[0] + bbox2[2] / 2.0 as f32,
        bbox2[1] + bbox2[3] / 2.0 as f32,
    );

    let xA = f32::max(boxA.0, boxB.0);
    let yA = f32::max(boxA.1, boxB.1);
    let xB = f32::min(boxA.2, boxB.2);
    let yB = f32::min(boxA.3, boxB.3);

    let interArea = f32::max(0.0, xB - xA + 1.0) * f32::max(0.0, yB - yA + 1.0);

    let boxAArea = (boxA.2 - boxA.0 + 1.0) * (boxA.3 - boxA.1 + 1.0);
    let boxBArea = (boxB.2 - boxB.0 + 1.0) * (boxB.3 - boxB.1 + 1.0);

    let ret = interArea / (boxAArea + boxBArea - interArea);
    ret
}
