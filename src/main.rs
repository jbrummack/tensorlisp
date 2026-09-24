fn main() {
    /*let program = std::fs::read_to_string("example.tl").unwrap();
    let mut parser = Parser::new(&program).unwrap();
    let result = parser.parse();
    println!("{result:#?}");
    parser.spurdump();*/
    //stupid_lisp().unwrap();
    //load_lisp().unwrap();
    //tensorlisp::backend::hashcons::hashed_lisp();
    //tensorlisp::backend::loader::load_model().unwrap();
    //tensorlisp::onnx::loader::load_onnx().unwrap();
    tensorlisp::coreml::load_cml().unwrap();
}
