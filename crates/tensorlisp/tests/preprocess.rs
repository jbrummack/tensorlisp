//! (preprocess ...) with tokenizers, images and audio from the program.
use std::path::PathBuf;


use tensorlisp::{
    Device, LoadOptions, Model, Program, RawInput, RunOptions,
    autopro::{
        audio::{Audio, MelSpectrogram},
        image::{Filter, ImageProcessor, Size},
    },
    gguf::GgufWriter,
};

const TOKENIZER_JSON: &str = r#"{
  "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
  "normalizer": {"type": "Lowercase"}, "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null, "decoder": null,
  "model": {"type": "WordLevel", "vocab": {"[PAD]": 0, "[UNK]": 1, "a": 2, "photo": 3, "of": 4, "dog": 5}, "unk_token": "[UNK]"}
}"#;

fn write(name: &str, program: &str, assets: &[(&str, &[u8])]) -> PathBuf {
    let mut w = GgufWriter::new();
    w.set_program(&Program::Text(program.into())).unwrap();
    for (n, bytes) in assets {
        w.set_asset(n, bytes).unwrap();
    }
    let path = std::env::temp_dir().join(format!("tensorlisp-pre-{}-{name}.gguf", std::process::id()));
    w.write(&path).unwrap();
    path
}

fn raw(name: &str, input: RawInput) -> Vec<(String, RawInput)> {
    vec![(name.to_string(), input)]
}

#[test]
fn text_through_an_embedded_tokenizer() {
    let program = r#"
      (define tok (tokenizer (asset "tokenizer.json")))
      (preprocess ([text string])
        (let-values ([(ids mask) (tokenize tok text 'pad-to 6)])
          (model-inputs [ids ids] [mask mask])))
      (model (inputs [ids i32 (6 batch)] [mask f32 (6 batch)])
        (outputs [ids (ggml-scale (ggml-cast ids GGML_TYPE_F32) 1.0) 2] [mask (ggml-scale mask 1.0) 2]))
    "#;
    let path = write("text", program, &[("tokenizer.json", TOKENIZER_JSON.as_bytes())]);
    let model = Model::load(&path, Device::Cpu).unwrap();
    assert_eq!(model.raw_inputs().unwrap()[0].name, "text");

    let examples = vec![raw("text", RawInput::Text("A photo of a DOG".into())), raw("text", RawInput::Text("dog cat".into()))];
    let out = model.run_raw(examples, &RunOptions::default()).unwrap();
    let ids = &out.outputs[0].1;
    assert_eq!(ids.shape(), &[2, 6]);
    assert_eq!(ids.iter().copied().collect::<Vec<_>>(), [2., 3., 4., 2., 5., 0., 5., 1., 0., 0., 0., 0.]);
    assert_eq!(out.outputs[1].1.iter().copied().collect::<Vec<_>>(), [1., 1., 1., 1., 1., 0., 1., 1., 0., 0., 0., 0.]);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn images_match_autopro() {
    let program = r#"
      (preprocess ([photo image])
        (model-inputs [x (image->array (image-center-crop (image-resize-shortest photo 16 'bicubic) 16 12)
                                       'mean '(0.5 0.5 0.5) 'std '(0.25 0.25 0.25))]))
      (model (inputs [x f32 (_ _ 3 batch)]) (outputs [y (ggml-scale x 1.0) 4]))
    "#;
    let path = write("image", program, &[]);
    let model = Model::load(&path, Device::Cpu).unwrap();
    let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(40, 30, |x, y| image::Rgb([(x * 6) as u8, (y * 8) as u8, ((x + y) * 3) as u8])));

    let arrays = model.preprocess(raw("photo", RawInput::Image(img.clone()))).unwrap();
    let expected = ImageProcessor {
        resize: Some((Size::ShortestEdge { edge: 16, max_longest: None }, Filter::Bicubic)),
        center_crop: Some((16, 12)),
        normalize: Some(([0.5; 3], [0.25; 3])),
        ..ImageProcessor::default()
    }
    .process(&img);
    assert_eq!(arrays[0].1, expected.into_dyn());

    let out = model.run_raw(vec![raw("photo", RawInput::Image(img))], &RunOptions::default()).unwrap();
    assert_eq!(out.outputs[0].1.shape(), &[1, 3, 12, 16]);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn audio_features() {
    let program = r#"
      (preprocess ([clip audio])
        (let ([clip (audio-resample clip 16000)])
          (model-inputs [mel (log-mel clip 'mels 40 'n-fft 400 'hop 160)])))
      (model (inputs [mel f32 (_ 40 batch)]) (outputs [y (ggml-scale mel 1.0) 3]))
    "#;
    let path = write("audio", program, &[]);
    let model = Model::load(&path, Device::Cpu).unwrap();
    let samples: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
    let arrays = model.preprocess(raw("clip", RawInput::Audio(Audio::new(samples.clone(), 16000)))).unwrap();
    let mut mel = MelSpectrogram::whisper(40);
    mel.f_max = 8000.0;
    assert_eq!(arrays[0].1, mel.compute(&samples).into_dyn());
    assert_eq!(arrays[0].1.shape(), &[40, 101]);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn preprocessing_errors() {
    let program = r#"
      (preprocess ([clip audio] [text string])
        (model-inputs [x (whisper-features clip 80)]))
      (model (inputs [x f32]) (outputs [y (ggml-scale x 1.0)]))
    "#;
    let path = write("errors", program, &[]);
    let model = Model::load(&path, Device::Cpu).unwrap();
    let clip = || RawInput::Audio(Audio::new(vec![0.0; 800], 8000));

    let err = model.preprocess(vec![("clip".into(), clip())]).unwrap_err().to_string();
    assert!(err.contains("missing raw input \"text\""), "{err}");
    let err = model
        .preprocess(vec![("clip".into(), RawInput::Text("x".into())), ("text".into(), RawInput::Text("y".into()))])
        .unwrap_err()
        .to_string();
    assert!(err.contains("\"clip\" must be Audio"), "{err}");
    let err = model.preprocess(vec![("clip".into(), clip()), ("text".into(), RawInput::Text("y".into()))]).unwrap_err().to_string();
    assert!(err.contains("need 16000 Hz") && err.contains("whisper-features"), "{err}");

    // A program reading a missing asset fails to load, naming it.
    let path2 = write("asset", "(define t (tokenizer (asset \"nope.json\"))) (model (inputs [x f32]) (outputs [y x]))", &[]);
    let err = Model::load(&path2, Device::Cpu).err().unwrap().to_string();
    assert!(err.contains("no asset") && err.contains("nope.json"), "{err}");
    // ... unless it is supplied at load time.
    let options = LoadOptions { assets: vec![("nope.json".into(), TOKENIZER_JSON.as_bytes().to_vec())], ..Default::default() };
    assert!(Model::load_with(&path2, Device::Cpu, options).is_ok());

    std::fs::remove_file(path).unwrap();
    std::fs::remove_file(path2).unwrap();
}
