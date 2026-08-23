use tensorlisp::{
    backend::{
        hashcons::hashed_lisp,
        stupid_lisp::{load_lisp, stupid_lisp},
    },
    parser::Parser,
};

fn main() {
    /*let program = std::fs::read_to_string("example.tl").unwrap();
    let mut parser = Parser::new(&program).unwrap();
    let result = parser.parse();
    println!("{result:#?}");
    parser.spurdump();*/
    //stupid_lisp().unwrap();
    //load_lisp().unwrap();
    hashed_lisp();
}
