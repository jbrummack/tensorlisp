//! (postprocess ...): detections and clustering on model outputs, per example.
use std::path::PathBuf;

use ndarray::{Array2, ArrayD, Axis, s};
use tensorlisp::{
    Device, Model, Program, RawInput, RunOptions,
    autopro::{
        cluster::{Dbscan, Metric, centroids},
        detect::{Detector, unletterbox},
    },
    gguf::GgufWriter,
};

fn write(name: &str, program: &str) -> PathBuf {
    let mut w = GgufWriter::new();
    w.set_program(&Program::Text(program.into())).unwrap();
    let path = std::env::temp_dir().join(format!("tensorlisp-post-{}-{name}.gguf", std::process::id()));
    w.write(&path).unwrap();
    path
}

/// Deterministic values in [0, 1).
fn noise(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// A YOLOv8-style head [4 + classes, n]: cxcywh boxes on a 64x64 input, then class probabilities.
fn yolo_head(n: usize, classes: usize, seed: u64) -> Array2<f32> {
    let mut r = noise(seed);
    let mut head = Array2::zeros((4 + classes, n));
    for j in 0..n {
        let (cx, cy) = (8.0 + 48.0 * r(), 8.0 + 48.0 * r());
        head.column_mut(j).slice_mut(s![..4]).assign(&ndarray::arr1(&[cx, cy, 4.0 + 12.0 * r(), 4.0 + 12.0 * r()]));
        for c in 0..classes {
            head[[4 + c, j]] = r().powi(3);
        }
    }
    head
}

#[test]
fn detections_on_the_original_image() {
    // Identity "detector": the raw head passes through the model; the image
    // only provides its size, the letterbox the model input geometry.
    let program = r#"
      (preprocess ([photo image] [head array])
        (model-inputs [pixels (image->array (image-letterbox photo 64 64))] [head head]))
      (model (inputs [pixels f32 (64 64 3 batch)] [head f32 (_ 7 batch)])
        (outputs [head (ggml-scale head 1.0) 3] [mean (ggml-mean pixels) 4]))
      (postprocess (head photo)
        (let ([rows (array-transpose head)])
          (let-values ([(boxes scores classes rows) (detect (array-slice rows 0 4 'axis 1) (array-slice rows 4 #f 'axis 1)
                                                            'score-threshold 0.2 'iou 0.5)])
            (let ([size (image-size photo)])
              (results [boxes (boxes-unletterbox boxes 64 64 (car size) (cadr size))]
                       [scores scores] [classes classes] [rows rows]
                       [count (array-length scores)] [size size])))))
    "#;
    let path = write("detect", program);
    let model = Model::load(&path, Device::Cpu).unwrap();
    assert_eq!(model.postprocess_args().unwrap(), ["head", "photo"]);

    let sizes = [(200u32, 100u32), (90, 120)];
    let heads: Vec<Array2<f32>> = (0..2).map(|i| yolo_head(50, 3, i + 1)).collect();
    let examples = sizes
        .iter()
        .zip(&heads)
        .map(|(&(w, h), head)| {
            let photo = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(w, h, image::Rgb([200, 30, 30])));
            vec![("photo".to_string(), RawInput::Image(photo)), ("head".to_string(), RawInput::Array(head.clone().into_dyn()))]
        })
        .collect();
    let inference = model.infer(examples, &RunOptions::default()).unwrap();
    assert_eq!(inference.run.outputs[0].1.shape(), &[2, 7, 50]);
    assert_eq!(inference.examples.len(), 2);

    for ((result, head), &(w, h)) in inference.examples.iter().zip(&heads).zip(&sizes) {
        let get = |name: &str| result.iter().find(|(n, _)| n == name).unwrap().1.as_array().unwrap();
        let rows = head.t();
        let want = Detector { score_threshold: 0.2, iou_threshold: 0.5, ..Detector::default() }
            .detect(rows.slice(s![.., ..4]), rows.slice(s![.., 4..]));
        assert!(!want.is_empty());
        let boxes = unletterbox(want.boxes.view(), (64, 64), (w, h));
        assert_eq!(get("boxes"), &boxes.into_dyn());
        assert_eq!(get("scores"), &want.scores.clone().into_dyn());
        let classes: Vec<f32> = want.classes.iter().map(|&c| c as f32).collect();
        assert_eq!(get("classes").iter().copied().collect::<Vec<_>>(), classes);
        let rows: Vec<f32> = want.indices.iter().map(|&i| i as f32).collect();
        assert_eq!(get("rows").iter().copied().collect::<Vec<_>>(), rows);
        assert_eq!(get("count"), &ndarray::arr0(want.len() as f32).into_dyn());
        assert_eq!(get("size").iter().copied().collect::<Vec<_>>(), [w as f32, h as f32]);
    }

    // Without raw inputs, the postprocess can't see the photo.
    let err = model.postprocess(&inference.run.outputs).unwrap_err().to_string();
    assert!(err.contains("needs raw input \"photo\""), "{err}");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn clustering_patch_embeddings() {
    // Tokens [cls + 64 patches, 16]; the postprocess drops the CLS token and clusters the patches.
    let program = r#"
      (model (inputs [tokens f32 (16 65 batch)]) (outputs [tokens (ggml-scale tokens 1.0) 3]))
      (postprocess (tokens)
        (let* ([patches (array-slice tokens 1 #f)]
               [labels (dbscan patches 0.1 4 'metric 'cosine)])
          (results [labels labels]
                   [centroids (cluster-centroids patches labels)]
                   [grid (array-reshape labels 8 8)]
                   [best (array-argmax patches)]
                   [cls (array-take tokens '(0))])))
    "#;
    let path = write("cluster", program);
    let model = Model::load(&path, Device::Cpu).unwrap();

    // Two regions of the 8x8 grid with their own direction, plus noise.
    let mut r = noise(7);
    let protos: Vec<Vec<f32>> = (0..2).map(|_| (0..16).map(|_| r() - 0.5).collect()).collect();
    let tokens = ArrayD::from_shape_fn(ndarray::IxDyn(&[2, 65, 16]), |idx| {
        let (b, t, d) = (idx[0], idx[1], idx[2]);
        let region = if t == 0 { 0 } else { ((t - 1) % 8 >= 4) as usize ^ b };
        protos[region][d] * (1.0 + t as f32 / 65.0) + 0.02 * (r() - 0.5)
    });
    let out = model.run_with(&[("tokens", tokens.view())], &RunOptions::default()).unwrap();
    let results = model.postprocess(&out.outputs).unwrap();
    assert_eq!(results.len(), 2);

    for (b, result) in results.iter().enumerate() {
        let get = |name: &str| result.iter().find(|(n, _)| n == name).unwrap().1.as_array().unwrap();
        let patches = tokens.index_axis(Axis(0), b).slice(s![1.., ..]).to_owned();
        let want = Dbscan { eps: 0.1, min_samples: 4, metric: Metric::Cosine }.fit(patches.view());
        assert_eq!(want.n_clusters, 2);
        let labels: Vec<f32> = want.labels.iter().map(|&l| l as f32).collect();
        assert_eq!(get("labels").iter().copied().collect::<Vec<_>>(), labels);
        assert_eq!(get("centroids"), &centroids(patches.view(), &want.labels, 2).into_dyn());
        assert_eq!(get("grid").shape(), &[8, 8]);
        assert_eq!(get("best").shape(), &[64]);
        assert_eq!(get("cls"), &tokens.slice(s![b, 0..1, ..]).to_owned().into_dyn());
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn postprocess_errors() {
    let program = r#"
      (preprocess ([x array]) (model-inputs [x x]))
      (model (inputs [x f32 (4 batch)]) (outputs [y (ggml-scale x 2.0) 2]))
      (postprocess (y missing) (results [z y]))
    "#;
    let path = write("errors", program);
    let model = Model::load(&path, Device::Cpu).unwrap();
    let x = ndarray::arr2(&[[1.0f32, 2.0, 3.0, 4.0]]).into_dyn();
    let out = model.run_with(&[("x", x.view())], &RunOptions::default()).unwrap();
    let err = model.postprocess(&out.outputs).unwrap_err().to_string();
    assert!(err.contains("\"missing\" is neither an output (y) nor a raw input"), "{err}");

    let bad_value = write("bad-value", "(model (inputs [x f32 (4 batch)]) (outputs [y x 2])) (postprocess (y) (results [z (list \"text\")]))");
    let model = Model::load(&bad_value, Device::Cpu).unwrap();
    let err = model.postprocess(&[("y".into(), x.clone())]).unwrap_err().to_string();
    assert!(err.contains("z must be an array, a number, a string or a list of numbers"), "{err}");
    let err = model.postprocess(&[("y".into(), ndarray::arr1(&[1.0f32, 2.0]).into_dyn()), ("w".into(), x.clone())]).unwrap_err().to_string();
    assert!(err.contains("into 2 examples") && err.contains("\"w\" has shape [1, 4]"), "{err}");

    // The same array twice, a boxes shape error from the host.
    let twice = write(
        "twice",
        "(model (inputs [x f32 (4 batch)]) (outputs [y x 2])) (postprocess (y) (results [a y] [b y] [c (nms (array-reshape y 1 4) (list->array '(1)))]))",
    );
    let model = Model::load(&twice, Device::Cpu).unwrap();
    let r = model.postprocess(&[("y".into(), x.clone())]).unwrap();
    assert_eq!(r[0][0].1, r[0][1].1);
    assert_eq!(r[0][2].1.as_array().unwrap().iter().copied().collect::<Vec<_>>(), [0.0]);
    let bad_boxes = write("bad-boxes", "(model (inputs [x f32 (4 batch)]) (outputs [y x 2])) (postprocess (y) (results [c (nms (array-reshape y 2 2) y)]))");
    let model = Model::load(&bad_boxes, Device::Cpu).unwrap();
    let err = model.postprocess(&[("y".into(), x)]).unwrap_err().to_string();
    assert!(err.contains("boxes must be [n, 4], got [2, 2]"), "{err}");

    for p in [path, bad_value, twice, bad_boxes] {
        std::fs::remove_file(p).unwrap();
    }
}
