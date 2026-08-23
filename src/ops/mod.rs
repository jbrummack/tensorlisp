pub enum Op {
    IsExpr, //is expr?
    IsList, //is list?
    Car,    //Rest
    Cdr,    //First
    Cons,   //Construct list
    Quote,
    Eq,
    Cond,
    Lambda, //λ/fn
    If,
    Def,
    Math(Math),
}
pub enum Math {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}
